"""Stage 4 — Filter & Extract.

Papers:     concurrent lightweight LLM relevance check (Y/N per paper).
Guidelines: concurrent lightweight LLM passage extraction from full text.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List

from openoe.core.models import Document
from openoe.llm.client import LLMClient
from openoe.llm.prompts import PromptManager

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper relevance filter
# ---------------------------------------------------------------------------

async def filter_papers(
    question: str,
    papers: List[Document],
    cfg: Dict[str, Any],
    llm: LLMClient,
    prompts: PromptManager,
) -> List[Document]:
    """Keep only papers the LLM considers relevant."""
    if not cfg.get("enabled") or not papers:
        return papers

    model = cfg["model"]
    provider = cfg.get("provider")
    sem = asyncio.Semaphore(int(cfg.get("concurrency", 20)))

    async def _check(doc: Document) -> bool:
        text = (doc.content or doc.title)[:8192]
        prompt = prompts.render("relevance_check_v2", question=question, retrieved=text)
        async with sem:
            try:
                r = await llm.chat(
                    step="filter", prompt=prompt,
                    system="You are a medical relevance judge. Output only Y or N.",
                    model=model, provider=provider,
                    max_tokens=2048,          # Y/N output is tiny, but reasoning model needs headroom
                    reasoning_effort="low",   # simple binary classification
                )
                raw = r.content.strip()
                # Accept: "Y", "Yes", "y", or any response containing Y/yes as first meaningful char
                first_char = raw.upper().lstrip(".,;:!?#*- \t\n")[:1]
                relevant = first_char == "Y"
                doc.relevance = relevant
                log.info("Filter %s: raw=%r → %s", doc.id[:40], raw[:20], "KEEP" if relevant else "DROP")
                return relevant
            except Exception as e:
                log.warning("Relevance check failed for %s: %s — keeping", doc.id, e)
                doc.relevance = True
                return True

    flags = await asyncio.gather(*[_check(d) for d in papers])
    kept = [d for d, ok in zip(papers, flags) if ok]
    log.info("Paper filter: %d/%d kept", len(kept), len(papers))
    return kept


# ---------------------------------------------------------------------------
# Guideline passage extraction
# ---------------------------------------------------------------------------

async def extract_guideline_passages(
    question: str,
    guidelines: List[Document],
    cfg: Dict[str, Any],
    llm: LLMClient,
    prompts: PromptManager,
) -> List[Document]:
    """For each guideline, extract relevant passages from full text into doc.relevant_chunks."""
    if not cfg.get("enabled") or not guidelines:
        return guidelines

    model = cfg["model"]
    provider = cfg.get("provider")
    max_chunk = int(cfg.get("max_chunk_size", 8000))
    overlap = int(cfg.get("overlap", 128))
    sem = asyncio.Semaphore(int(cfg.get("concurrency", 20)))

    # Build flat task list: (guideline_idx, chunk_text, title)
    tasks: List[tuple[int, str, str]] = []
    for gi, g in enumerate(guidelines):
        content = g.content or ""
        if not content:
            continue
        if len(content) <= max_chunk:
            tasks.append((gi, content, g.title))
        else:
            s = 0
            while s < len(content):
                tasks.append((gi, content[s:s + max_chunk], g.title))
                s += max_chunk - overlap

    async def _extract(gi: int, chunk: str, title: str) -> tuple[int, List[str]]:
        prompt = prompts.render("extract_relevant_chunks_v2", title=title, content=chunk, question=question)
        async with sem:
            try:
                r = await llm.chat(
                    step="extract", prompt=prompt,
                    system="You are a precise medical information extractor. Output only JSON.",
                    model=model, provider=provider,
                    max_tokens=8192,          # JSON output with multiple paragraphs can be long
                    reasoning_effort="low",   # extraction, not deep reasoning
                )
                raw = r.content.strip()
                m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
                data = json.loads(m.group(1) if m else raw)
                return gi, data.get("relevant_content", [])
            except Exception as e:
                log.warning("Chunk extraction failed (guideline %d): %s", gi, e)
                return gi, []

    results = await asyncio.gather(*[_extract(gi, ch, t) for gi, ch, t in tasks])

    # Reassemble per guideline and deduplicate
    buckets: Dict[int, List[str]] = {}
    for gi, chunks in results:
        buckets.setdefault(gi, []).extend(chunks)

    for gi, g in enumerate(guidelines):
        raw_chunks = buckets.get(gi, [])
        seen: set[str] = set()
        unique: List[str] = []
        for ch in raw_chunks:
            if not isinstance(ch, str) or not ch.strip():
                continue
            fp = ch[:100]
            if fp not in seen:
                unique.append(ch)
                seen.add(fp)
        g.relevant_chunks = unique

    total = sum(len(g.relevant_chunks or []) for g in guidelines)
    log.info("Guideline extraction: %d passages from %d guidelines", total, len(guidelines))
    return guidelines
