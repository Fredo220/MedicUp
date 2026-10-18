"""Stage 1 — Query Enhancement.

Runs in two phases:
  Phase A (concurrent): HyDE + Translate
  Phase B (sequential, uses Phase A results): Keyword/search query generation + Embedding
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from openoe.llm.client import LLMClient
from openoe.llm.prompts import PromptManager
from openoe.utils.text import extract_keywords_heuristic

log = logging.getLogger(__name__)


async def enhance_query(
    question: str,
    cfg: Dict[str, Any],
    llm: LLMClient,
    prompts: PromptManager,
    history: Optional[List[Dict[str, str]]] = None,
    embed_fn=None,
) -> Dict[str, Any]:
    """Return dict with keys: hyde, translated, keywords, search_query, embedding."""

    hyde_cfg = cfg.get("hyde", {})
    trans_cfg = cfg.get("translate", {})
    kw_cfg = cfg.get("keywords", {})

    # ── Phase A: HyDE + Translate (concurrent) ────────────
    async def _hyde() -> Optional[str]:
        if not hyde_cfg.get("enabled"):
            return None
        try:
            r = await llm.chat(
                step="hyde", prompt=prompts.render("generate_hyde", question=question),
                system="You are a medical search assistant.",
                model=hyde_cfg["model"], provider=hyde_cfg.get("provider"),
                max_tokens=8192, reasoning_effort="medium", history=history,
            )
            text = r.content.strip() or None
            log.info("HyDE result: %s", text[:80] if text else "empty")
            return text
        except Exception as e:
            log.warning("HyDE failed: %s", e)
            return None

    async def _translate() -> Optional[str]:
        if not trans_cfg.get("enabled"):
            return None
        try:
            r = await llm.chat(
                step="translate", prompt=prompts.render("translate_to_search_english", question=question),
                system="You are a medical search rewriting assistant.",
                model=trans_cfg["model"], provider=trans_cfg.get("provider"),
                max_tokens=4096, reasoning_effort="low", history=history,
            )
            text = r.content.strip() or None
            log.info("Translate result: %s", text[:80] if text else "empty")
            return text
        except Exception as e:
            log.warning("Translate failed: %s", e)
            return None

    hyde, translated = await asyncio.gather(_hyde(), _translate())

    # ── Phase B: Keywords (uses hyde + translated for context) ──
    keywords: List[str] = []
    search_query: Optional[str] = None
    if kw_cfg.get("enabled"):
        max_kw = int(kw_cfg.get("max_keywords", 10))
        try:
            prompt = prompts.render(
                "keyword_extraction",
                question=question,
                max_keywords=max_kw,
                hyde_text=hyde or "(none)",
                translated=translated or "(none)",
            )
            r = await llm.chat(
                step="keywords",
                prompt=prompt,
                system="You are a medical literature search expert. Output only the search query.",
                model=kw_cfg["model"], provider=kw_cfg.get("provider"),
                max_tokens=4096, reasoning_effort="low",
            )
            raw = r.content.strip()
            # The model outputs a single-line search expression
            search_query = raw
            # Also split into individual keywords for other uses
            keywords = [w.strip() for w in raw.replace(",", " ").split() if w.strip() and len(w.strip()) > 1]
            log.info("Search query: %s", search_query[:80])
            log.info("Keywords parsed: %s", keywords[:8])
        except Exception as e:
            log.warning("Keyword extraction failed: %s — using heuristic", e)
            keywords = extract_keywords_heuristic(question, max_kw)
            search_query = " ".join(keywords)

    # ── Embedding (Full mode) ──
    embedding = None
    if embed_fn:
        try:
            text_to_embed = hyde or translated or question
            embedding = await embed_fn(text_to_embed)
        except Exception as e:
            log.warning("Embedding failed: %s", e)

    return {
        "hyde": hyde,
        "translated": translated,
        "keywords": keywords,
        "search_query": search_query,
        "embedding": embedding,
    }