"""Web context enrichment via Tavily.

* ``question_context`` — general web background for the question (fed to the answer model).
* ``add_supporting_sources`` — for each recommended paper, find web pages that discuss it
  (news, trial registries, guidelines, commentaries) and attach them as ``supporting_sources``.

Failures are logged and never break the pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Dict, List

from openoe.sources.tavily import clean_title, domain, extract_ids, tavily_search
from openoe.utils.dedup import normalize_doi
from openoe.utils.score import to_percent
from openoe.utils.text import clean, truncate

log = logging.getLogger(__name__)

SearchFn = Callable[..., Awaitable[Dict[str, Any]]]


def _to_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in results:
        url = r.get("url", "")
        title = clean_title(r.get("title", ""))
        if not url or not title:
            continue
        out.append({
            "title": title,
            "url": url,
            "domain": domain(url),
            "snippet": truncate(clean(r.get("content", "")), 300),
            # Tavily relevance is already absolute in [0, 1].
            "score_pct": to_percent(r.get("score"), 1.0),
        })
    return out


async def question_context(query: str, cfg: Dict[str, Any], *, search: SearchFn = tavily_search) -> Dict[str, Any]:
    """Returns ``{"answer": str | None, "sources": [...]}`` (empty dict if disabled/failed)."""
    if not cfg.get("enabled"):
        return {}
    try:
        data = await search(
            query,
            api_key=cfg.get("api_key", ""),
            max_results=int(cfg.get("max_results", 5)),
            search_depth=cfg.get("search_depth", "basic"),
            include_answer=bool(cfg.get("include_answer", True)),
            exclude_domains=cfg.get("exclude_domains"),
        )
    except Exception as e:
        log.warning("Web context search failed: %s", e)
        return {}
    return {"answer": data.get("answer"), "sources": _to_sources(data.get("results", []))}


def _same_paper(paper: Dict[str, Any], src: Dict[str, Any]) -> bool:
    if paper.get("url") and paper["url"] == src["url"]:
        return True
    pdoi = normalize_doi(paper.get("doi"))
    if pdoi and pdoi == normalize_doi(extract_ids(src["url"]).get("doi")):
        return True
    a, b = (paper.get("title") or "").lower(), src["title"].lower()
    return bool(a) and SequenceMatcher(None, a, b).ratio() >= 0.9


async def add_supporting_sources(
    papers: List[Dict[str, Any]], cfg: Dict[str, Any], *, search: SearchFn = tavily_search,
) -> int:
    """Attach ``supporting_sources`` to the first ``top_k`` paper dicts. Returns how many got any."""
    sup = cfg.get("supporting", {}) or {}
    top_k = int(sup.get("top_k", 5))
    if not cfg.get("enabled") or top_k <= 0:
        return 0
    per_paper = int(sup.get("max_results", 3))
    targets = [p for p in papers if p.get("doc_type") == "paper" and p.get("title")][:top_k]

    async def _one(p: Dict[str, Any]) -> None:
        try:
            data = await search(
                p["title"],
                api_key=cfg.get("api_key", ""),
                max_results=per_paper + 2,
                search_depth=cfg.get("search_depth", "basic"),
                exclude_domains=cfg.get("exclude_domains"),
            )
        except Exception as e:
            log.warning("Supporting-source search failed for %r: %s", p["title"][:60], e)
            p["supporting_sources"] = []
            return
        srcs = [s for s in _to_sources(data.get("results", [])) if not _same_paper(p, s)]
        p["supporting_sources"] = srcs[:per_paper]

    await asyncio.gather(*[_one(p) for p in targets])
    return sum(1 for p in targets if p.get("supporting_sources"))
