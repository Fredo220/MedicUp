"""Orchestration for seeded paper discovery."""
from __future__ import annotations

import inspect
import time
from typing import Any, Dict, List, Optional

from openoe.discovery.candidates import gather_candidates
from openoe.discovery.models import DiscoverRequest, DiscoverResponse
from openoe.discovery.openalex import OpenAlexClient
from openoe.discovery.pubmed import PubMedFetchError, fetch_abstracts
from openoe.discovery.rank import rank_candidates
from openoe.discovery.seeds import resolve_seeds
from openoe.llm.client import LLMClient
from openoe.llm.prompts import PromptManager


async def discover(
    req: DiscoverRequest,
    progress: Any = None,
    llm: Optional[LLMClient] = None,
    prompts: Optional[PromptManager] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if llm is None or prompts is None or cfg is None:
        raise ValueError("discover requires configured llm, prompts, and discovery config")
    if isinstance(llm, LLMClient):
        llm.ensure_configured()
    timings: Dict[str, float] = {}
    errors: List[str] = []

    async def notify(stage: str, message: str, detail: Dict[str, Any]) -> None:
        if progress:
            value = progress(stage, message, detail)
            if inspect.isawaitable(value):
                await value

    started = time.perf_counter()
    openalex_cfg = cfg.get("openalex", {})
    client = OpenAlexClient(
        base_url=openalex_cfg.get("base_url", "https://api.openalex.org"),
        email=openalex_cfg.get("email", ""),
        api_key=openalex_cfg.get("api_key"),
    )
    await notify("seeds", "Resolving seed papers", {"n_seeds": len(req.seeds)})
    seeds, unresolved = await resolve_seeds(req.seeds, client)
    timings["seeds"] = (time.perf_counter() - started) * 1000

    candidates_started = time.perf_counter()
    await notify("candidates", "Gathering candidate papers", {"n_seeds": len(seeds)})
    candidates, web_context, candidate_errors = await gather_candidates(
        question=req.question,
        seeds=seeds,
        disliked=req.disliked,
        client=client,
        llm=llm,
        prompts=prompts,
        cfg=cfg,
    )
    errors.extend(candidate_errors)
    timings["candidates"] = (time.perf_counter() - candidates_started) * 1000

    papers_to_backfill = [*seeds, *(candidate["paper"] for candidate in candidates)]
    pmids = list(dict.fromkeys(
        paper.pmid for paper in papers_to_backfill if paper.pmid and not paper.abstract
    ))
    if pmids:
        try:
            pubmed_abstracts = await fetch_abstracts(pmids)
        except PubMedFetchError as exc:
            pubmed_abstracts = exc.abstracts
            errors.append(str(exc))
        except Exception as exc:
            pubmed_abstracts = {}
            errors.append(f"PubMed abstract fetch failed: {type(exc).__name__}")
        for paper in papers_to_backfill:
            if not paper.abstract and paper.pmid and pubmed_abstracts.get(paper.pmid):
                paper.abstract = pubmed_abstracts[paper.pmid]

    results, n_judged, rank_errors, rank_timings, assessments = await rank_candidates(
        question=req.question,
        seeds=seeds,
        candidates=candidates,
        llm=llm,
        prompts=prompts,
        cfg=cfg,
        limit=req.limit or int(cfg.get("limit", 20)),
        progress=progress,
    )
    errors.extend(rank_errors)
    timings.update(rank_timings)
    stats: Dict[str, Any] = {"signals": {}, "n_candidates": len(candidates), "n_judged": n_judged,
                             "assessments": assessments,
                             "n_assessed": sum(row.get("assessment_status") == "assessed" for row in assessments)}
    for candidate in candidates:
        for signal_type in {signal.get("type") for signal in candidate.get("signals", [])}:
            if signal_type:
                stats["signals"][signal_type] = stats["signals"].get(signal_type, 0) + 1
    if client.rate_limited:
        errors = [e for e in errors if not e.startswith("OpenAlex GET")]
        errors.insert(0, "OpenAlex daily limit reached (free tier, resets at midnight UTC) - searched PubMed instead")
    response = DiscoverResponse(
        seeds=[paper.to_api_dict() for paper in seeds],
        unresolved=unresolved,
        results=results,
        web_context=web_context,
        stats=stats,
        timings=timings,
        errors=list(dict.fromkeys(errors)),
    ).model_dump()
    timings["done"] = (time.perf_counter() - started) * 1000
    response["timings"] = timings
    await notify("done", "Discovery complete", {
        "n_candidates": len(candidates),
        "n_results": len(results),
    })
    return response
