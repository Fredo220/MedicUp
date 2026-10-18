"""Evidence-grounded follow-up answers for discovery result sets."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from openoe.llm.client import LLMClient
from openoe.llm.prompts import PromptManager

_BARE_CITATION_RE = re.compile(r"(?<![\^\w])\[\d+(?:\s*[,\-–]\s*\d+)*\]")
_FOOTNOTE_RE = re.compile(r"\[\^(\d+)\]")


async def answer_followup(
    llm: LLMClient,
    prompts: PromptManager,
    cfg: Dict[str, Any],
    question: str,
    history: List[Dict[str, str]],
    papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    numbered = []
    by_number: Dict[int, Dict[str, Any]] = {}
    for index, paper in enumerate(papers, 1):
        number = int(paper.get("n", index))
        by_number[number] = paper
        numbered.append({
            "n": number,
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "abstract": paper.get("abstract", ""),
            "openalex_id": paper.get("openalex_id"),
            "links": paper.get("links", {}),
        })
    response = await llm.chat(
        step="discovery_followup",
        prompt=prompts.render(
            "discovery_followup",
            question=question,
            papers=json.dumps(numbered, ensure_ascii=False, indent=2),
        ),
        system="Answer only from the supplied papers and cite factual claims.",
        model=cfg["strong_model"],
        max_tokens=1200,
        history=history,
    )
    answer = _BARE_CITATION_RE.sub("", response.content.strip())
    cited_numbers = []
    for value in _FOOTNOTE_RE.findall(answer):
        number = int(value)
        if number in by_number and number not in cited_numbers:
            cited_numbers.append(number)
    citations = []
    for number in cited_numbers:
        paper = by_number[number]
        links = paper.get("links") or {}
        citations.append({
            "n": number,
            "openalex_id": paper.get("openalex_id"),
            "title": paper.get("title", ""),
            "url": links.get("openalex") or links.get("doi"),
        })
    return {"answer": answer, "citations": citations}
