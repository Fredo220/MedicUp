"""Async OpenAlex client with request-local in-memory GET caching."""
from __future__ import annotations

import asyncio
import copy
import logging
import os
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote

import httpx

from openoe.discovery.models import normalize_doi, short_openalex_id

log = logging.getLogger(__name__)

SELECT_FIELDS = (
    "id,doi,title,publication_year,cited_by_count,authorships,primary_location,"
    "abstract_inverted_index,referenced_works,related_works,ids,type,open_access,"
    "is_retracted,primary_topic"
)


def _describe_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return type(exc).__name__


class OpenAlexClient:
    def __init__(self, base_url: str, email: str, api_key: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY", "")
        self.cache: Dict[tuple, Any] = {}
        self.errors: List[str] = []
        self.search_calls = 0
        self.rate_limited = False

    async def _get(self, url: str, params: Dict[str, Any]) -> Any:
        request_params = dict(params)
        request_params["select"] = SELECT_FIELDS
        if self.email:
            request_params["mailto"] = self.email
        if self.api_key:
            request_params["api_key"] = self.api_key
        cache_key = (url, tuple(sorted((key, str(value)) for key, value in request_params.items())))
        if cache_key in self.cache:
            return copy.deepcopy(self.cache[cache_key])
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=request_params)
                if response.status_code == 429:
                    await asyncio.sleep(1.5)
                    response = await client.get(url, params=request_params)
                if response.status_code == 429:
                    self.rate_limited = True
                    raise httpx.HTTPStatusError(
                        "OpenAlex rate limited (429)", request=response.request, response=response
                    )
                response.raise_for_status()
                data = response.json()
            self.cache[cache_key] = data
            return copy.deepcopy(data)
        except Exception as exc:
            message = f"OpenAlex GET {url} failed: {_describe_error(exc)}"
            log.warning("%s", message)
            self.errors.append(message)
            return {}

    async def get_work(self, key: str) -> Dict[str, Any]:
        key = str(key).strip()
        if key.startswith("W"):
            endpoint_key = key
        elif key.lower().startswith("doi:"):
            endpoint_key = key
        elif key.lower().startswith("pmid:"):
            endpoint_key = key
        else:
            endpoint_key = key
        data = await self._get(f"{self.base_url}/works/{quote(endpoint_key, safe=':')}", {})
        return data if isinstance(data, dict) else {}

    async def get_works(self, ids: Iterable[str]) -> List[Dict[str, Any]]:
        work_ids = list(dict.fromkeys(short_openalex_id(item) for item in ids if item))
        results: List[Dict[str, Any]] = []
        for start in range(0, len(work_ids), 50):
            batch = work_ids[start:start + 50]
            if not batch:
                continue
            data = await self._get(
                f"{self.base_url}/works",
                {"filter": f"openalex_id:{'|'.join(batch)}", "per-page": 50},
            )
            results.extend(data.get("results", []) if isinstance(data, dict) else [])
        return results

    async def by_dois(self, dois: Iterable[str]) -> List[Dict[str, Any]]:
        normalized = list(dict.fromkeys(
            doi.lower() for item in dois if (doi := normalize_doi(item))
        ))
        results: List[Dict[str, Any]] = []
        for start in range(0, len(normalized), 50):
            batch = normalized[start:start + 50]
            data = await self._get(
                f"{self.base_url}/works",
                {"filter": f"doi:{'|'.join(batch)}", "per-page": 50},
            )
            results.extend(data.get("results", []) if isinstance(data, dict) else [])
        return results

    async def citing(self, wid: str, n: int = 25) -> List[Dict[str, Any]]:
        data = await self._get(
            f"{self.base_url}/works",
            {"filter": f"cites:{short_openalex_id(wid)}", "sort": "cited_by_count:desc", "per-page": n},
        )
        return data.get("results", []) if isinstance(data, dict) else []

    async def search(self, q: str, n: int = 15) -> List[Dict[str, Any]]:
        if self.search_calls >= 3:
            log.warning("OpenAlex search budget exhausted for this discovery run")
            return []
        self.search_calls += 1
        data = await self._get(
            f"{self.base_url}/works",
            {"search": q, "filter": "type:article", "per-page": n},
        )
        return data.get("results", []) if isinstance(data, dict) else []
