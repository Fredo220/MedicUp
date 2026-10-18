"""Candidate gathering from OpenAlex and optional recommendation/web sources."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

from openoe.discovery.models import Paper, normalize_doi, paper_from_work, short_openalex_id
from openoe.discovery.openalex import OpenAlexClient
from openoe.discovery.pubmed import search_pubmed
from openoe.discovery.seeds import parse_seed
from openoe.llm.client import LLMClient
from openoe.llm.prompts import PromptManager

log = logging.getLogger(__name__)
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)


def _enabled(cfg: Dict[str, Any], signal_type: str) -> bool:
    value = cfg.get("signals", {}).get(signal_type, {})
    return value.get("enabled", True) if isinstance(value, dict) else bool(value)


def _extract_json_list(text: str) -> List[str]:
    match = re.search(r"\[[\s\S]*?\]", text)
    if not match:
        return []
    try:
        value = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()][:2]


async def gather_candidates(
    question: str,
    seeds: List[Paper],
    disliked: List[str],
    client: OpenAlexClient,
    llm: LLMClient,
    prompts: PromptManager,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]], List[str]]:
    errors: List[str] = []
    web_context: List[Dict[str, str]] = []
    candidate_map: Dict[str, Dict[str, Any]] = {}
    seed_ids = {paper.openalex_id for paper in seeds}
    seed_dois = {paper.doi.lower() for paper in seeds if paper.doi}
    disliked_ids = set()
    disliked_dois = set()
    for value in disliked:
        kind, parsed = parse_seed(value)
        if kind == "openalex":
            disliked_ids.add(parsed)
        elif kind == "doi":
            disliked_dois.add(parsed.lower())

    seed_pmids = {paper.pmid for paper in seeds if paper.pmid}
    key_by_doi: Dict[str, str] = {}
    key_by_pmid: Dict[str, str] = {}

    def add_paper(paper: Optional[Paper], signal: Dict[str, Any]) -> None:
        if not paper or paper.openalex_id in seed_ids or paper.openalex_id in disliked_ids:
            return
        doi = (paper.doi or "").lower()
        if doi and doi in (seed_dois | disliked_dois):
            return
        if paper.pmid and paper.pmid in seed_pmids:
            return
        key = paper.openalex_id
        if key not in candidate_map:
            key = key_by_doi.get(doi) or key_by_pmid.get(paper.pmid or "") or key
        record = candidate_map.get(key)
        if record is None:
            if len(candidate_map) >= 200:
                return
            record = {"paper": paper, "signals": []}
            candidate_map[key] = record
            if doi:
                key_by_doi[doi] = key
            if paper.pmid:
                key_by_pmid[paper.pmid] = key
        if signal not in record["signals"]:
            record["signals"].append(signal)

    def add_work(work: Dict[str, Any], signal: Dict[str, Any]) -> None:
        add_paper(paper_from_work(work), signal)

    async def cited_by_seeds() -> None:
        if not _enabled(cfg, "cited_by_seeds"):
            return
        linked: Dict[str, Set[int]] = {}
        for index, paper in enumerate(seeds):
            for wid in paper.referenced_works:
                if wid:
                    linked.setdefault(wid, set()).add(index)
        top_ids = sorted(linked, key=lambda wid: (-len(linked[wid]), wid))[:60]
        works = await client.get_works(top_ids)
        by_id = {short_openalex_id(work.get("id")): work for work in works}
        for wid in top_ids:
            if wid in by_id:
                add_work(by_id[wid], {"type": "cited_by_seeds", "seeds": sorted(linked[wid])})

    async def cites_seeds() -> None:
        if not _enabled(cfg, "cites_seeds"):
            return
        batches = await asyncio.gather(*(client.citing(p.openalex_id) for p in seeds))
        for index, works in enumerate(batches):
            for work in works:
                add_work(work, {"type": "cites_seeds", "seeds": [index]})

    async def related() -> None:
        if not _enabled(cfg, "related"):
            return
        linked: Dict[str, Set[int]] = {}
        for index, paper in enumerate(seeds):
            for wid in paper.related_works:
                if wid:
                    linked.setdefault(wid, set()).add(index)
        works = await client.get_works(linked)
        by_id = {short_openalex_id(work.get("id")): work for work in works}
        for wid, seed_indices in linked.items():
            if wid in by_id:
                add_work(by_id[wid], {"type": "related", "seeds": sorted(seed_indices)})

    async def semantic_scholar() -> None:
        if not _enabled(cfg, "s2_recommendation"):
            return
        positive = [f"DOI:{paper.doi}" for paper in seeds if paper.doi]
        negative = [f"DOI:{parse_seed(value)[1]}" for value in disliked if parse_seed(value)[0] == "doi"]
        if not positive:
            return
        try:
            async with httpx.AsyncClient(timeout=30.0) as session:
                response = await session.post(
                    "https://api.semanticscholar.org/recommendations/v1/papers",
                    params={"limit": 40, "fields": "externalIds,title"},
                    json={"positivePaperIds": positive, "negativePaperIds": negative},
                )
                response.raise_for_status()
                data = response.json()
            recommended = data.get("recommendedPapers", data.get("papers", []))
            dois: List[str] = []
            doi_ranks: Dict[str, int] = {}
            for rank, item in enumerate(recommended, 1):
                doi = normalize_doi((item.get("externalIds") or {}).get("DOI"))
                if doi:
                    dois.append(doi)
                    doi_ranks[doi.lower()] = rank
            for work in await client.by_dois(dois):
                doi = normalize_doi(work.get("doi"))
                rank = doi_ranks.get((doi or "").lower())
                if rank:
                    add_work(work, {"type": "s2_recommendation", "rank": rank})
        except Exception as exc:
            message = f"Semantic Scholar recommendations: {exc}"
            log.warning("%s", message)
            errors.append(message)

    async def search_candidates() -> None:
        if not _enabled(cfg, "search"):
            return
        try:
            generated = await llm.chat(
                step="discovery_queries",
                prompt=prompts.render(
                    "discovery_queries",
                    question=question,
                    seed_titles="\n".join(f"- {paper.title}" for paper in seeds),
                ),
                system="Return only the requested JSON list of search queries.",
                model=cfg["small_model"],
                max_tokens=500,
                reasoning_effort="low",
            )
            queries = _extract_json_list(generated.content)
        except Exception as exc:
            errors.append(f"Discovery query generation: {exc}")
            queries = []
        if not queries:
            queries = [paper.title for paper in seeds[:2]] or [question]
        if not seeds and question.strip() and question not in queries:
            queries.insert(0, question)
        n_queries, per_query = (3, 25) if not seeds else (2, 15)
        queries = queries[:n_queries]
        batches = await asyncio.gather(*(client.search(query, per_query) for query in queries))
        for query, works in zip(queries, batches):
            for work in works:
                add_work(work, {"type": "search", "query": query})
        for query in [query for query, works in zip(queries, batches) if not works]:
            try:
                papers = await search_pubmed(query, per_query)
            except Exception as exc:
                errors.append(f"PubMed search: {type(exc).__name__}")
                continue
            for paper in papers:
                add_paper(paper, {"type": "search", "query": query, "source": "pubmed"})

    async def web_search() -> None:
        nonlocal web_context
        if not _enabled(cfg, "web"):
            return
        key = os.getenv("TAVILY_API_KEY", "")
        if not key:
            return
        query = question.strip() or " ".join(paper.title for paper in seeds) + " research"
        try:
            async with httpx.AsyncClient(timeout=30.0) as session:
                response = await session.post(
                    "https://api.tavily.com/search",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"query": query, "max_results": 8, "search_depth": "basic"},
                )
                response.raise_for_status()
                data = response.json()
            results = data.get("results", [])
            web_context = [
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "snippet": str(item.get("content") or "")[:300],
                }
                for item in results
            ]
            dois: List[str] = []
            doi_urls: Dict[str, str] = {}
            for item in results:
                text = f"{item.get('url', '')} {item.get('content', '')}"
                for match in _DOI_RE.findall(text):
                    doi = match.rstrip(".,;:!?)]}>\"'")
                    if doi:
                        dois.append(doi)
                        doi_urls.setdefault(doi.lower(), str(item.get("url") or ""))
            for work in await client.by_dois(dois):
                doi = normalize_doi(work.get("doi"))
                url = doi_urls.get((doi or "").lower(), "")
                add_work(work, {"type": "web", "url": url})
        except Exception as exc:
            message = f"Tavily search: {exc}"
            log.warning("%s", message)
            errors.append(message)

    await asyncio.gather(
        cited_by_seeds(),
        cites_seeds(),
        related(),
        semantic_scholar(),
        search_candidates(),
        web_search(),
    )
    errors.extend(client.errors)
    return list(candidate_map.values()), web_context, errors
