"""Stage 0 — Safety Check.

Lightweight LLM classifies the question as medical_related / non_medical / unsafe.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from openoe.llm.client import LLMClient
from openoe.llm.prompts import PromptManager

log = logging.getLogger(__name__)


async def check_safety(
    question: str,
    cfg: Dict[str, Any],
    llm: LLMClient,
    prompts: PromptManager,
) -> Optional[Dict[str, str]]:
    if not cfg.get("enabled", False):
        return None

    prompt = prompts.render("safety_guard", question=question)
    try:
        r = await llm.chat(
            step="safety",
            prompt=prompt,
            system="You are a medical RAG safety guard.",
            model=cfg["model"],
            provider=cfg.get("provider"),
            max_tokens=2048,          # reasoning overhead + short JSON output
            reasoning_effort="low",   # simple classification, no deep reasoning needed
        )
        raw = r.content.strip()
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL)
        js = m.group(1) if m else raw
        data = json.loads(js)
        return {"flag": data.get("flag", "unsafe"), "reason": data.get("reason", "")}
    except Exception as e:
        log.warning("Safety check failed: %s — defaulting to unsafe", e)
        return {"flag": "unsafe", "reason": f"Safety check error: {e}"}