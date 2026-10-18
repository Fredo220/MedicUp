"""Tavily web search.

``TavilySource`` is a retrieval route that searches scholarly web domains
(PubMed, PMC, publishers, preprint servers) and turns hits into paper
documents, so web discovery is fused with OpenAlex via RRF.
``tavily_search`` is also used by the web-context enrichment stage.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from openoe.core.models import Document
from openoe.utils.text import clean

log = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"

_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>\[\]()|,;]+)", re.I)
_DOI_SUFFIX = re.compile(r"(/(full|abstract|pdf|epdf|html|meta))+$", re.I)
_PMID_URL = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
_PMID_TXT = re.compile(r"\bPMID:?\s*(\d{5,9})\b")
_PMC = re.compile(r"\b(PMC\d{4,})\b")
_SITE_SUFFIX = re.compile(r"\s+[-|–]\s+(PubMed|PMC|Europe PMC|ScienceDirect|Nature|bioRxiv|medRxiv)$", re.I)


def clean_title(title: str) -> str:
    """Drop site branding (``Frontiers | X``, ``X - PubMed``) so titles fuzzy-match OpenAlex."""
    t = clean(title)
    if " | " in t:
        t = max(t.split(" | "), key=len).strip()
    return _SITE_SUFFIX.sub("", t)


async def tavily_search(
    query: str,
    *,
    api_key: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    include_answer: bool = False,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set")
    body: Dict[str, Any] = {
        "query": query[:400],
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": include_answer,
    }
    if include_domains:
        body["include_domains"] = include_domains
    if exclude_domains:
        body["exclude_domains"] = exclude_domains
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(TAVILY_URL, headers={"Authorization": f"Bearer {api_key}"}, json=body)
        r.raise_for_status()
        return r.json()


def extract_ids(url: str, text: str = "") -> Dict[str, str]:
    """Pull DOI / PMID / PMCID out of a result URL or snippet."""
    out: Dict[str, str] = {}
    m = _DOI.search(url or "") or _DOI.search(text or "")
    if m:
        doi = _DOI_SUFFIX.sub("", m.group(1).rstrip("."))
        out["doi"] = f"https://doi.org/{doi}"
    m = _PMID_URL.search(url or "") or _PMID_TXT.search(text or "")
    if m:
        out["pmid"] = m.group(1)
    m = _PMC.search(url or "") or _PMC.search(text or "")
    if m:
        out["pmcid"] = m.group(1)
    return out


def domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


class TavilySource:
    """Scholarly web discovery via Tavily."""

    async def search(
        self,
        *,
        query: str,
        keywords: List[str],
        embedding: Optional[List[float]] = None,
        mode: str = "scholarly",
        limit: int = 10,
        route_name: str = "tavily_web",
        config: Dict[str, Any] = {},
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        # keywords[0] is the translated natural-language question when translation ran; it suits
        # web search better than the OpenAlex expression in ``query``. Single words are not.
        q = keywords[0] if keywords and " " in keywords[0].strip() else query
        data = await tavily_search(
            q,
            api_key=config.get("api_key", ""),
            max_results=limit,
            search_depth=config.get("search_depth", "basic"),
            include_domains=config.get("include_domains") if mode == "scholarly" else None,
            exclude_domains=config.get("exclude_domains"),
        )
        results = data.get("results", [])
        log.info("Tavily [%s] %r → %d results", route_name, q[:50], len(results))

        docs: List[Document] = []
        for rank, item in enumerate(results, 1):
            title = clean_title(item.get("title", ""))
            url = item.get("url", "")
            if not title or not url:
                continue
            content = clean(item.get("content", ""))
            docs.append(Document(
                id=url,
                source="tavily",
                route=route_name,
                doc_type="paper",
                title=title,
                content=content,
                metadata={"url": url, "domain": domain(url), **extract_ids(url, content)},
                score=float(item.get("score") or 0.0),
                score_type="tavily",
            ))
        return docs
