"""Embedding-based candidate ranking and LLM evidence judging."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple

from openoe.discovery.models import Paper
from openoe.llm.client import LLMClient, LLMError
from openoe.llm.prompts import PromptManager

log = logging.getLogger(__name__)
_BARE_CITATION_RE = re.compile(r"(?<![\^\w])\[\d+(?:\s*[,\-–]\s*\d+)*\]")
_RANDOMIZED_RE = re.compile(r"(?<!non-)(?<!non )\brandomi[sz]ed\b", re.I)
_NON_RANDOMIZED_RE = re.compile(r"\bnon[- ]?randomi[sz]ed\b", re.I)
_STUDY_DESIGNS = (
    "meta-analysis",
    "systematic review",
    "RCT",
    "non-randomized trial",
    "cohort",
    "case-control",
    "cross-sectional",
    "case report",
    "narrative review",
    "guideline",
    "animal",
    "in vitro",
    "computational",
    "unclear",
)


def normalize_study_design(value: Any) -> str:
    if not isinstance(value, str):
        return "unclear"
    normalized = value.strip().casefold()
    return next(
        (design for design in _STUDY_DESIGNS if design.casefold() == normalized),
        "unclear",
    )


def reconcile_study_design(design: str, title: str, abstract: str) -> str:
    if design != "non-randomized trial":
        return design
    text = f"{title} {abstract}"
    if _RANDOMIZED_RE.search(text) and not _NON_RANDOMIZED_RE.search(text):
        return "RCT"
    return design


def score_to_pct(score: float) -> int:
    return max(1, min(100, round(score * 100)))


def parse_judge_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escaped = False
    end = -1
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    if end < 0:
        raise ValueError("unterminated JSON object")
    result = json.loads(text[start:end])
    if not isinstance(result, dict):
        raise ValueError("judge response is not a JSON object")
    result["study_design"] = normalize_study_design(result.get("study_design"))
    return result


def _normalize_quote(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().strip().strip("\"'“”‘’"))


def verify_quote(quote: str, abstract: str) -> bool:
    normalized_quote = _normalize_quote(quote)
    normalized_abstract = re.sub(r"\s+", " ", abstract.lower().strip())
    return bool(normalized_quote) and normalized_quote in normalized_abstract


def _cosine(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _seed_link_count(signals: List[Dict[str, Any]]) -> int:
    linked: set[int] = set()
    for signal in signals:
        if signal.get("type") in {"cited_by_seeds", "cites_seeds"}:
            linked.update(int(index) for index in signal.get("seeds", []))
    return len(linked)


def calculate_pre_score(
    similarity: float,
    min_similarity: float,
    max_similarity: float,
    seed_links: int,
    n_seeds: int,
    n_signal_types: int,
    has_s2_recommendation: bool,
) -> float:
    if max_similarity > min_similarity:
        normalized_similarity = (similarity - min_similarity) / (max_similarity - min_similarity)
    else:
        normalized_similarity = 0.0
    seed_component = 0.3 * min(seed_links / n_seeds, 1) if n_seeds else 0.0
    return (
        0.5 * normalized_similarity
        + seed_component
        + 0.1 * min((n_signal_types - 1) / 3, 1)
        + 0.1 * bool(has_s2_recommendation)
    )


def calculate_final_score(relevance: float, pre_score: float) -> float:
    return 0.6 * relevance / 10 + 0.4 * pre_score


def _candidate_text(paper: Paper) -> str:
    return f"{paper.title}\n{paper.abstract[:1500]}"


async def rank_candidates(
    question: str,
    seeds: List[Paper],
    candidates: List[Dict[str, Any]],
    llm: LLMClient,
    prompts: PromptManager,
    cfg: Dict[str, Any],
    limit: int = 20,
    progress: Any = None,
) -> Tuple[List[Dict[str, Any]], int, List[str], Dict[str, float], List[Dict[str, Any]]]:
    timings: Dict[str, float] = {}
    assessments: List[Dict[str, Any]] = []

    def assessment(item: Dict[str, Any], status: str, reason: str) -> Dict[str, Any]:
        return {"openalex_id": item["paper"].openalex_id, "title": item["paper"].title,
                "assessment_status": status, "reason": reason}

    eligible = []
    for item in candidates:
        if item["paper"].is_retracted:
            assessments.append(assessment(item, "excluded", "Retracted publication."))
        else:
            eligible.append(item)
    candidates = eligible

    async def notify(stage: str, message: str, detail: Dict[str, Any]) -> None:
        if progress:
            value = progress(stage, message, detail)
            if inspect.isawaitable(value):
                await value

    rank_started = time.perf_counter()
    if not candidates:
        await notify("rank", "Embedding seeds and candidate papers", {
            "n_seeds": len(seeds), "n_candidates": 0,
        })
        timings["rank"] = (time.perf_counter() - rank_started) * 1000
        await notify("judge", "Judging the highest-ranked candidates", {"n_judged": 0})
        timings["judge"] = 0.0
        return [], 0, [], timings, assessments

    texts = [_candidate_text(paper) for paper in seeds]
    texts.extend(_candidate_text(item["paper"]) for item in candidates)
    question_index = None
    if question.strip():
        question_index = len(texts)
        texts.append(question)

    await notify("rank", "Embedding seeds and candidate papers", {
        "n_seeds": len(seeds), "n_candidates": len(candidates),
    })
    errors: List[str] = []
    vectors: List[List[float]]
    try:
        vectors = await llm.embed(texts, cfg["embedding_model"])
    except LLMError as exc:
        if exc.systemic:
            raise
        errors.append(f"Embedding unavailable ({exc.code}); ranking stopped.")
        assessments.extend(assessment(item, "unassessed", "Embedding stage unavailable.") for item in candidates)
        timings["rank"] = (time.perf_counter() - rank_started) * 1000
        timings["judge"] = 0.0
        return [], 0, errors, timings, assessments
    except Exception as exc:
        errors.append(f"Embedding unavailable ({type(exc).__name__}); ranking stopped.")
        log.warning("Discovery embedding failed: %s", type(exc).__name__)
        assessments.extend(assessment(item, "unassessed", "Embedding stage unavailable.") for item in candidates)
        timings["rank"] = (time.perf_counter() - rank_started) * 1000
        timings["judge"] = 0.0
        return [], 0, errors, timings, assessments

    seed_vectors = vectors[:len(seeds)]
    candidate_start = len(seeds)
    candidate_vectors = vectors[candidate_start:candidate_start + len(candidates)]
    if seed_vectors and seed_vectors[0]:
        dim = len(seed_vectors[0])
        centroid = [
            sum(vector[index] for vector in seed_vectors if len(vector) == dim) /
            max(1, sum(1 for vector in seed_vectors if len(vector) == dim))
            for index in range(dim)
        ]
    else:
        centroid = []
    question_vector = vectors[question_index] if question_index is not None and question_index < len(vectors) else []

    similarities: List[float] = []
    for vector in candidate_vectors:
        seed_similarity = _cosine(centroid, vector)
        if question_index is not None and not seeds:
            similarities.append(_cosine(question_vector, vector))
        elif question_index is not None:
            similarities.append(0.5 * _cosine(question_vector, vector) + 0.5 * seed_similarity)
        else:
            similarities.append(seed_similarity)
    min_similarity = min(similarities, default=0.0)
    max_similarity = max(similarities, default=0.0)

    scored = []
    n_seeds = len(seeds)
    for index, item in enumerate(candidates):
        signals = item.get("signals", [])
        signal_types = {signal.get("type") for signal in signals}
        links = _seed_link_count(signals)
        has_s2 = "s2_recommendation" in signal_types
        pre_score = calculate_pre_score(
            similarities[index], min_similarity, max_similarity, links, n_seeds,
            len(signal_types), has_s2,
        )
        scored.append({
            **{key: item[key] for key in ("metadata_sources", "metadata_conflicts") if key in item},
            "paper": item["paper"],
            "signals": signals,
            "similarity": similarities[index],
            "seed_links": links,
            "n_signals": len(signal_types),
            "pre_score": pre_score,
        })
    scored.sort(key=lambda entry: entry["pre_score"], reverse=True)
    judge_k = int(cfg.get("judge_k", 25))
    to_judge = scored[:max(0, judge_k)]
    assessments.extend(assessment(item, "unassessed", "Outside the configured judge budget.")
                       for item in scored[len(to_judge):])
    timings["rank"] = (time.perf_counter() - rank_started) * 1000
    await notify("judge", "Judging the highest-ranked candidates", {"n_judged": len(to_judge)})
    seed_titles = "\n".join(f"S{index + 1}: {paper.title}" for index, paper in enumerate(seeds)) or "(none — judge against the research question only; related_seeds must be [])"
    semaphore = asyncio.Semaphore(10)
    judge_started = time.perf_counter()
    judge_model = cfg.get("judge_model") or cfg.get("strong_model") or cfg["small_model"]
    min_relevance = max(1, min(10, int(cfg.get("min_relevance", 1))))

    async def judge(entry: Dict[str, Any]) -> Dict[str, Any]:
        paper: Paper = entry["paper"]
        fallback = {
            "assessment_status": "unassessed",
            "reason": "Relevance assessment unavailable.",
        }
        prompt = prompts.render(
            "discovery_judge",
            question=question or "No separate research question was provided; assess relevance to the seed papers.",
            seed_titles=seed_titles,
            title=paper.title,
            year=paper.year or "",
            abstract=paper.abstract,
        )
        async with semaphore:
            try:
                response = await llm.chat(
                    step="discovery_judge",
                    prompt=prompt,
                    system="Return only the requested JSON object.",
                    model=judge_model,
                    max_tokens=900,
                    reasoning_effort="low",
                )
                parsed = parse_judge_json(response.content)
                raw_relevance = parsed.get("relevance")
                if isinstance(raw_relevance, bool) or not isinstance(raw_relevance, (int, float)):
                    raise ValueError("relevance must be an integer from 0 to 10")
                if raw_relevance not in range(11):
                    raise ValueError("relevance must be an integer from 0 to 10")
                relevance = int(raw_relevance)
                why = str(parsed.get("why") or "").strip()
                if relevance >= min_relevance and not why:
                    raise ValueError("relevant recommendations require an explanation")
                quote = str(parsed.get("quote") or "")
                if quote and not verify_quote(quote, paper.abstract):
                    raise ValueError("supporting quote is not present in the candidate abstract")
                raw_related = parsed.get("related_seeds") or []
                if not isinstance(raw_related, list):
                    raise ValueError("related_seeds must be a list")
                related = []
                for seed_ref in raw_related:
                    match = re.fullmatch(r"\s*S(\d+)\s*", str(seed_ref), flags=re.I)
                    if match:
                        index = int(match.group(1)) - 1
                        if 0 <= index < len(seeds) and index not in related:
                            related.append(index)
                return {
                    "assessment_status": "assessed",
                    "relevance": relevance,
                    "why": why,
                    "quote": quote,
                    "related_seeds": related,
                    "study_design": parsed["study_design"],
                }
            except LLMError as exc:
                if exc.systemic:
                    raise
                errors.append(f"Judge {paper.openalex_id}: assessment unavailable ({exc.code}).")
                return fallback
            except Exception as exc:
                log.warning("Discovery judge failed for %s: %s", paper.openalex_id, type(exc).__name__)
                errors.append(f"Judge {paper.openalex_id}: assessment unavailable ({type(exc).__name__}).")
                return fallback

    tasks = [asyncio.create_task(judge(entry)) for entry in to_judge]
    try:
        judged_values = await asyncio.gather(*tasks)
    except LLMError:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    timings["judge"] = (time.perf_counter() - judge_started) * 1000
    results = []
    for entry, judged in zip(to_judge, judged_values):
        paper: Paper = entry["paper"]
        if judged["assessment_status"] != "assessed":
            assessments.append(assessment(entry, "unassessed", judged["reason"]))
            continue
        if judged["relevance"] < min_relevance:
            assessments.append(assessment(entry, "excluded", "Below the configured relevance gate."))
            continue
        assessments.append(assessment(entry, "assessed", "Passed the relevance gate."))
        final_score = calculate_final_score(judged["relevance"], entry["pre_score"])
        results.append({
            **paper.to_api_dict(),
            **{key: entry[key] for key in ("metadata_sources", "metadata_conflicts") if key in entry},
            "assessment_status": "assessed",
            "judge_model": judge_model,
            "score": final_score,
            "score_pct": score_to_pct(final_score),
            "components": {
                "similarity": entry["similarity"],
                "seed_links": entry["seed_links"],
                "n_signals": entry["n_signals"],
                "llm_relevance": judged["relevance"],
                "pre_score": entry["pre_score"],
            },
            "why": judged["why"],
            "quote": judged["quote"],
            "quote_verified": verify_quote(judged["quote"], paper.abstract),
            "related_seeds": judged["related_seeds"],
            "study_design": reconcile_study_design(
                judged["study_design"], paper.title, paper.abstract,
            ),
            "signals": entry["signals"],
        })
    results.sort(key=lambda entry: entry["score"], reverse=True)
    results = results[:max(0, limit)]
    for rank, item in enumerate(results, 1):
        item["rank"] = rank
    return results, len(to_judge), errors, timings, assessments
