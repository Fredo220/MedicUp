"""OpenAlex API search source (Lite mode).

Three parallel search strategies for maximum recall:
  1. Keyword search: LLM-crafted search expression (high precision)
  2. Keyword search: Translated natural query (broader recall)  
  3. Semantic search: HyDE hypothetical document (conceptual similarity)

Results are merged, deduplicated by OpenAlex ID, and capped at limit.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import date
from typing import Any, Dict, List, Optional

import httpx

from openoe.core.models import Document
from openoe.utils.text import clean

log = logging.getLogger(__name__)


def _safe_error(exc: Exception) -> str:
    """Error text without the request URL, which can contain the API key."""
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


class OpenAlexSource:
    """Multi-strategy OpenAlex search: keyword + keyword + semantic."""

    async def search(
        self,
        *,
        query: str,
        keywords: List[str],
        embedding: Optional[List[float]] = None,
        mode: str = "keyword",
        limit: int = 20,
        route_name: str = "openalex_kw",
        config: Dict[str, Any] = {},
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        base = config.get("base_url", "https://api.openalex.org").rstrip("/")
        email = config.get("email")
        api_key = config.get("api_key") or os.getenv("OPENALEX_API_KEY", "")
        errors: List[str] = []
        hyde_text = config.get("hyde_text")  # injected by orchestrator

        fv = ["type:article"]
        years = (filters or {}).get("date_range_years")
        if years:
            fv.append(f"from_publication_date:{date(date.today().year - int(years), 1, 1).isoformat()}")
        filter_str = ",".join(fv)

        primary_q = query.strip()                                         # LLM search expression
        secondary_q = keywords[0] if keywords and keywords[0] != primary_q else None  # translated

        async def _kw_search(search_q: str, n: int, label: str) -> List[Dict]:
            params: Dict[str, Any] = {"search": search_q, "per-page": n, "filter": filter_str}
            if email:
                params["mailto"] = email
            if api_key:
                params["api_key"] = api_key
            try:
                async with httpx.AsyncClient(timeout=30.0) as c:
                    r = await c.get(f"{base}/works", params=params)
                    r.raise_for_status()
                    data = r.json()
                    total = data.get("meta", {}).get("count", 0)
                    results = data.get("results", [])
                    log.info("OpenAlex [%s] %r → %d total, %d returned", label, search_q[:50], total, len(results))
                    return results
            except Exception as e:
                log.warning("OpenAlex [%s] failed: %s", label, _safe_error(e))
                errors.append(_safe_error(e))
                return []

        async def _semantic_search(text: str, label: str) -> List[Dict]:
            # Semantic search via AI embeddings. GET = query in URL.
            # Keep VERY short (~30 words) — strip numbers, parens, special chars to reduce URL bloat.
            import re as _re
            ascii_text = text.encode("ascii", errors="ignore").decode("ascii")
            cleaned = _re.sub(r"[^a-zA-Z\s,.\-]", " ", ascii_text)  # only letters
            words = [w for w in cleaned.split() if len(w) > 2]
            if len(words) < 5:
                log.warning("OpenAlex [%s] semantic skipped: too few English words", label)
                return []
            truncated = " ".join(words[:40])
            params: Dict[str, Any] = {"search.semantic": truncated, "per-page": 50, "filter": filter_str}
            if email:
                params["mailto"] = email
            if api_key:
                params["api_key"] = api_key
            try:
                async with httpx.AsyncClient(timeout=30.0) as c:
                    r = await c.get(f"{base}/works", params=params)
                    r.raise_for_status()
                    data = r.json()
                    results = data.get("results", [])
                    log.info("OpenAlex [%s] semantic → %d returned", label, len(results))
                    return results
            except Exception as e:
                log.warning("OpenAlex [%s] semantic failed: %s", label, _safe_error(e))
                errors.append(_safe_error(e))
                return []

        # Run searches sequentially with 1s delay to respect rate limits
        all_batches: List[List[Dict]] = []
        all_batches.append(await _kw_search(primary_q, limit, "keyword-primary"))
        if secondary_q:
            await asyncio.sleep(1)
            all_batches.append(await _kw_search(secondary_q, limit, "keyword-secondary"))
        if hyde_text:
            await asyncio.sleep(1)
            all_batches.append(await _semantic_search(hyde_text, "semantic-hyde"))

        if errors and not any(all_batches):
            raise RuntimeError(f"OpenAlex unavailable: {errors[-1]}")

        # Merge and deduplicate by OpenAlex ID, preserving priority order
        seen: set[str] = set()
        merged: List[Dict] = []
        for batch in all_batches:
            for item in batch:
                oa_id = item.get("id", "")
                if oa_id and oa_id not in seen:
                    seen.add(oa_id)
                    merged.append(item)

        log.info("OpenAlex total: %d unique from %d search routes", len(merged), len(all_batches))

        docs: List[Document] = []
        for rank, item in enumerate(merged[:limit], 1):
            title = clean(item.get("display_name", ""))
            if not title:
                continue
            abstract = _rebuild_abstract(item.get("abstract_inverted_index") or {})
            score = item.get("relevance_score") or float(len(merged) - rank + 1)
            docs.append(Document(
                id=item.get("id", f"openalex:{rank}"),
                source="openalex",
                route=route_name,
                doc_type="paper",
                title=title,
                content=abstract,
                metadata={
                    "doi": item.get("doi"),
                    "publication_year": item.get("publication_year"),
                    "openalex_id": item.get("id"),
                },
                score=float(score),
                score_type="api_rank",
            ))
        return docs


def _rebuild_abstract(idx: Dict[str, List[int]]) -> str:
    if not idx:
        return ""
    words: Dict[int, str] = {}
    for token, positions in idx.items():
        for pos in positions:
            words[pos] = token
    return " ".join(words[i] for i in sorted(words))