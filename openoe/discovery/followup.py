"""Evidence-grounded follow-up answers for discovery result sets."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from openoe.llm.client import LLMClient
from openoe.llm.prompts import PromptManager

_BARE_CITATION_RE = re.compile(r"(?<![\^\w])\[\d+(?:\s*[,\-–]\s*\d+)*\]")
_FOOTNOTE_RE = re.compile(r"\[\^(\d+)\]")
log = logging.getLogger(__name__)


async def suggest_followups(
    llm: LLMClient,
    prompts: PromptManager,
    cfg: Dict[str, Any],
    question: str,
    answer: str,
    titles: List[str],
) -> List[str]:
    try:
        response = await llm.chat(
            step="discovery_suggestions",
            prompt=prompts.render(
                "discovery_suggestions",
                question=question,
                answer=answer[:2500],
                titles="\n".join(f"- {title}" for title in titles[:20]),
            ),
            system="Return only a JSON list of 3 short follow-up questions.",
            model=cfg.get("small_model") or cfg["strong_model"],
            max_tokens=300,
            reasoning_effort="low",
        )
        match = re.search(r"\[[\s\S]*?\]", response.content)
        items = json.loads(match.group(0)) if match else []
        return [str(item).strip() for item in items if str(item).strip()][:3]
    except Exception as exc:
        log.warning("Follow-up suggestions failed: %s", type(exc).__name__)
        return []


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
    suggestions = await suggest_followups(
        llm, prompts, cfg, question, answer, [item["title"] for item in numbered],
    )
    return {"answer": answer, "citations": citations, "suggestions": suggestions}
