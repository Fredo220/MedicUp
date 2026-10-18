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
from openoe.discovery.pubmed import lookup_pubmed, search_pubmed
from openoe.discovery.seeds import parse_seed
from openoe.llm.client import LLMClient, LLMError
from openoe.llm.prompts import PromptManager

log = logging.getLogger(__name__)
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"<>]+", re.I)


_PUBMED_URL_RE = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
_SCHOLARLY_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "doi.org", "nejm.org", "thelancet.com",
    "jamanetwork.com", "bmj.com", "nature.com", "sciencedirect.com", "springer.com", "wiley.com",
    "ascopubs.org", "ahajournals.org", "medrxiv.org", "biorxiv.org", "cochranelibrary.com",
]


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


def _admission_order(record: Dict[str, Any]) -> tuple:
    """Bound the complete fused pool deterministically before expensive ranking."""
    signals = record["signals"]
    linked_seeds = {
        index for signal in signals
        if signal.get("type") in {"cited_by_seeds", "cites_seeds", "related"}
        for index in signal.get("seeds", [])
    }
    types = {signal.get("type") for signal in signals}
    recommendation_rank = min(
        (signal.get("rank", 41) for signal in signals if signal.get("type") == "s2_recommendation"),
        default=41,
    )
    return -len(linked_seeds), -len(types), recommendation_rank, record["paper"].openalex_id


def _paper_aliases(paper: Paper) -> Set[str]:
    aliases = {"id:" + paper.openalex_id}
    doi = normalize_doi(paper.doi)
    if doi:
        aliases.add("doi:" + doi.casefold())
    if paper.pmid:
        aliases.add("pmid:" + paper.pmid)
    if paper.openalex_id.startswith("pmid:"):
        aliases.add(paper.openalex_id)
    return aliases


def _merge_metadata(papers: List[Paper]) -> Tuple[Paper, Dict[str, Any]]:
    # Prefer an actual OpenAlex identity; sort all remaining ties rather than
    # letting the first HTTP response choose the publication's metadata.
    ordered = sorted(papers, key=lambda paper: (
        not bool(re.fullmatch(r"W\d+", paper.openalex_id)),
        paper.openalex_id,
        json.dumps(paper.to_api_dict(), sort_keys=True),
    ))
    updates: Dict[str, Any] = {}
    conflicts: Dict[str, Any] = {}
    for field in ("doi", "pmid", "title", "year", "venue", "topic", "field", "type", "oa_url"):
        values = [getattr(paper, field) for paper in ordered if getattr(paper, field) not in (None, "")]
        if field == "doi":
            values = [normalize_doi(value) for value in values]
        if values:
            updates[field] = values[0]
            distinct = {json.dumps(value, sort_keys=True): value for value in values}
            if len(distinct) > 1:
                conflicts[field] = [distinct[key] for key in sorted(distinct)]
    updates["abstract"] = max((paper.abstract for paper in ordered), key=lambda value: (len(value), value))
    updates["authors"] = max((paper.authors for paper in ordered), key=lambda value: (len(value), tuple(value)))
    citation_counts = [paper.cited_by_count for paper in ordered if paper.cited_by_count is not None]
    updates["cited_by_count"] = max(citation_counts, default=None)
    oa_status = [paper.is_oa for paper in ordered if paper.is_oa is not None]
    updates["is_oa"] = any(oa_status) if oa_status else None
    # A sparse or older provider record must never erase a retraction warning.
    updates["is_retracted"] = any(paper.is_retracted for paper in ordered)
    updates["links"] = {
        key: next((paper.links[key] for paper in ordered if paper.links.get(key)), None)
        for key in sorted({key for paper in ordered for key in paper.links})
    }
    for field in ("referenced_works", "related_works"):
        updates[field] = sorted({wid for paper in ordered for wid in getattr(paper, field)})
    return ordered[0].model_copy(update=updates), conflicts


def _fuse_candidates(
    entries: List[Tuple[Paper, Dict[str, Any]]], excluded_aliases: Set[str],
) -> List[Dict[str, Any]]:
    """Unify transitive identifiers before exclusions, metadata merging or caps."""
    parents = list(range(len(entries)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    by_alias: Dict[str, int] = {}
    for index, (paper, _) in enumerate(entries):
        for alias in _paper_aliases(paper):
            if alias in by_alias:
                parents[find(index)] = find(by_alias[alias])
            else:
                by_alias[alias] = index
    groups: Dict[int, List[Tuple[Paper, Dict[str, Any]]]] = {}
    for index, entry in enumerate(entries):
        groups.setdefault(find(index), []).append(entry)

    fused = []
    for group in groups.values():
        papers = [paper for paper, _ in group]
        aliases = {alias for paper in papers for alias in _paper_aliases(paper)}
        if aliases & excluded_aliases:
            continue
        paper, conflicts = _merge_metadata(papers)
        signals = {json.dumps(signal, sort_keys=True): signal for _, signal in group}
        fused.append({
            "paper": paper,
            "signals": [signals[key] for key in sorted(signals)],
            "metadata_sources": sorted({"pubmed" if item.openalex_id.startswith("pmid:") else "openalex" for item in papers}),
            "metadata_conflicts": conflicts,
        })
    return sorted(fused, key=_admission_order)[:200]


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
    entries: List[Tuple[Paper, Dict[str, Any]]] = []
    excluded_aliases = {alias for paper in seeds for alias in _paper_aliases(paper)}
    for value in disliked:
        kind, parsed = parse_seed(value)
        if kind == "openalex":
            excluded_aliases.add("id:" + parsed)
        elif kind == "doi":
            excluded_aliases.add("doi:" + parsed.casefold())
        elif kind == "pmid":
            excluded_aliases.update(("pmid:" + parsed, "id:pmid:" + parsed))

    def add_paper(paper: Optional[Paper], signal: Dict[str, Any]) -> None:
        if paper:
            entries.append((paper, signal))

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
        available = [(index, paper) for index, paper in enumerate(seeds)
                     if re.fullmatch(r"W\d+", paper.openalex_id)]
        if len(available) < len(seeds):
            errors.append("Citation graph unavailable for PubMed-only seeds without an OpenAlex identity; other retrieval signals remain available.")
        batches = await asyncio.gather(*(client.citing(p.openalex_id) for _, p in available))
        for (index, _), works in zip(available, batches):
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
        except LLMError as exc:
            if exc.systemic:
                raise
            errors.append(f"Discovery query generation: {exc}")
            queries = []
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

    async def _tavily(session: httpx.AsyncClient, key: str, query: str, **extra: Any) -> List[Dict[str, Any]]:
        response = await session.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {key}"},
            json={"query": query, "max_results": 8, "search_depth": "basic", **extra},
        )
        response.raise_for_status()
        return response.json().get("results", [])

    async def web_search() -> None:
        """Tavily agentic search: general results become web context; scholarly hits are resolved
        to DOIs/PMIDs (OpenAlex, then PubMed) and ranked like any other candidate."""
        nonlocal web_context
        if not _enabled(cfg, "web"):
            return
        key = cfg.get("tavily_api_key") or os.getenv("TAVILY_API_KEY", "")
        if not key:
            errors.append("Tavily: not configured; add TAVILY_API_KEY to enable web discovery")
            return
        topic = question.strip() or " ".join(paper.title for paper in seeds)
        try:
            async with httpx.AsyncClient(timeout=30.0) as session:
                general, scholarly = await asyncio.gather(
                    _tavily(session, key, topic + " research"),
                    _tavily(session, key, topic + " study", include_domains=_SCHOLARLY_DOMAINS),
                    return_exceptions=True,
                )
            if isinstance(general, Exception):
                errors.append(f"Tavily general search: {general}")
                general = []
            if isinstance(scholarly, Exception):
                errors.append(f"Tavily scholarly search: {scholarly}")
                scholarly = []
            web_context = [
                {
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "snippet": str(item.get("content") or "")[:300],
                }
                for item in general
            ]
            dois: List[str] = []
            pmids: List[str] = []
            found_urls: Dict[str, str] = {}
            for item in [*scholarly, *general]:
                url = str(item.get("url") or "")
                text = f"{url} {item.get('content') or ''}"
                pmid_match = _PUBMED_URL_RE.search(url)
                if pmid_match:
                    pmids.append(pmid_match.group(1))
                    found_urls.setdefault("pmid:" + pmid_match.group(1), url)
                for match in _DOI_RE.findall(text):
                    doi = match.rstrip(".,;:!?)]}>\"'")
                    if doi:
                        dois.append(doi)
                        found_urls.setdefault(doi.lower(), url)
            dois = list(dict.fromkeys(dois))[:20]
            pmids = list(dict.fromkeys(pmids))[:20]
            resolved_dois: Set[str] = set()
            if dois:
                for work in await client.by_dois(dois):
                    doi = normalize_doi(work.get("doi"))
                    resolved_dois.add((doi or "").lower())
                    add_work(work, {"type": "web", "url": found_urls.get((doi or "").lower(), "")})
            if client.rate_limited or len(resolved_dois) < len(dois) or pmids:
                for kind, value in [*(("doi", d) for d in dois if d.lower() not in resolved_dois), *(("pmid", p) for p in pmids)]:
                    try:
                        paper = await lookup_pubmed(kind, value)
                    except Exception:
                        paper = None
                    if paper:
                        add_paper(paper, {"type": "web", "url": found_urls.get(value.lower() if kind == "doi" else "pmid:" + value, ""), "source": "pubmed"})
        except Exception as exc:
            message = f"Tavily search: {exc}"
            log.warning("%s", message)
            errors.append(message)

    tasks = [asyncio.create_task(route()) for route in (
        cited_by_seeds, cites_seeds, related, semantic_scholar, search_candidates, web_search,
    )]
    try:
        await asyncio.gather(*tasks)
    except (Exception, asyncio.CancelledError):
        # A provider-wide failure stops this run; do not leave unrelated API
        # requests running after the caller has already received the error.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    errors.extend(client.errors)
    return _fuse_candidates(entries, excluded_aliases), web_context, errors
