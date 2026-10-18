"""Evidence-grounded follow-up answers for discovery result sets."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from openoe.llm.client import LLMClient, LLMError
from openoe.llm.prompts import PromptManager

_BARE_CITATION_RE = re.compile(r"(?<![\^\w])\[\d+(?:\s*[,\-–]\s*\d+)*\]")
_FOOTNOTE_RE = re.compile(r"\[\^(\d+)\]")
log = logging.getLogger(__name__)


def _eligible_evidence(paper: Dict[str, Any]) -> bool:
    decision = str(paper.get("decision") or "").casefold()
    return not (
        paper.get("is_retracted") is True
        or str(paper.get("flag") or "").casefold() == "retracted"
        or decision in {"r", "rej", "reject", "rejected"}
        or paper.get("rejected") is True
        or paper.get("assessment_status") in {"unassessed", "excluded"}
    )


def _cited_passages(text: str, allowed: set[int]) -> tuple[str, int]:
    """Keep only passages whose citations resolve to the current evidence set.

    This checks citation membership, not whether a source entails a claim. Never
    turn an unknown citation into an uncited claim by deleting only its marker.
    """
    # Models sometimes place a footnote immediately after sentence punctuation.
    text = re.sub(r"([.!?])\s*((?:\[\^\d+\]\s*)+)",
                  lambda m: " " + m.group(2).strip() + m.group(1) + " ", text.strip())
    units = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept = []
    removed = 0
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        refs = {int(value) for value in _FOOTNOTE_RE.findall(unit)}
        has_claim_text = re.search(r"[A-Za-z]", _FOOTNOTE_RE.sub("", unit))
        if not refs or not refs <= allowed or _BARE_CITATION_RE.search(unit) or not has_claim_text:
            removed += 1
            continue
        kept.append(unit)
    return " ".join(kept), removed


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
    except LLMError as exc:
        if exc.systemic:
            raise
        log.warning("Follow-up suggestions failed: %s", exc.code)
        return []
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
        if not _eligible_evidence(paper):
            continue
        try:
            number = int(paper.get("n", index))
        except (ValueError, TypeError):
            continue
        if number < 1 or number in by_number:
            continue
        by_number[number] = paper
        numbered.append({
            "n": number,
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "abstract": paper.get("abstract", ""),
            "openalex_id": paper.get("openalex_id"),
            "links": paper.get("links", {}),
        })
    if not numbered:
        return {
            "answer": "There are no eligible papers available to answer this question.",
            "citations": [], "suggestions": [],
            "assessment_status": "no_eligible_evidence", "errors": [],
        }
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
    answer, removed = _cited_passages(response.content, set(by_number))
    errors = ["Some passages were omitted because their citations were missing or invalid."] if removed else []
    if not answer:
        return {
            "answer": "The supplied papers did not yield an answer with valid source citations.",
            "citations": [], "suggestions": [], "errors": errors,
            "assessment_status": "insufficient_cited_evidence",
        }
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
    return {"answer": answer, "citations": citations, "suggestions": suggestions,
            "assessment_status": "answered", "citation_validation": "source_membership_only",
            "errors": errors}
