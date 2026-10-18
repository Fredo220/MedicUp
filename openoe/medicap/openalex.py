"""Small OpenAlex client for Medicap: lookups by DOI/PMID/ID, search, batch fetch, citing works."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

log = logging.getLogger(__name__)

SELECT = ",".join([
    "id", "doi", "ids", "display_name", "publication_year", "primary_location", "authorships",
    "cited_by_count", "type", "open_access", "is_retracted", "referenced_works",
    "abstract_inverted_index", "primary_topic",
])


def short_id(oa_id: str) -> str:
    return (oa_id or "").rstrip("/").rsplit("/", 1)[-1]


def abstract(inv: Optional[Dict[str, List[int]]]) -> str:
    if not inv:
        return ""
    pos = sorted((i, w) for w, idxs in inv.items() for i in idxs)
    return " ".join(w for _, w in pos)


_TYPE = {"review": "Review", "preprint": "Preprint", "article": "Article", "letter": "Letter",
         "editorial": "Editorial", "book-chapter": "Chapter", "dataset": "Dataset"}


def to_paper(w: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAlex work → the flat paper dict the Medicap UI uses."""
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    names = []
    for a in (w.get("authorships") or [])[:3]:
        n = ((a.get("author") or {}).get("display_name") or "").split()
        if n:
            names.append(f"{n[-1]} {''.join(p[0] for p in n[:-1])}".strip())
    if len(w.get("authorships") or []) > 3:
        names.append("et al")
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", w.get("doi") or "", flags=re.I)
    pmid = short_id((w.get("ids") or {}).get("pmid") or "")
    wtype = w.get("type") or "article"
    topic = w.get("primary_topic") or {}
    return {
        "id": short_id(w.get("id", "")),
        "t": w.get("display_name") or "Untitled",
        "au": ", ".join(names) or "Unknown authors",
        "y": w.get("publication_year"),
        "v": src.get("display_name") or "",
        "type": _TYPE.get(wtype, wtype.replace("-", " ").title()),
        "peer": int(wtype != "preprint" and src.get("type") != "repository"),
        "oa": int(bool((w.get("open_access") or {}).get("is_oa"))),
        "cit": w.get("cited_by_count") or 0,
        "doi": doi,
        "pmid": pmid,
        "url": f"https://doi.org/{doi}" if doi else (loc.get("landing_page_url") or w.get("id", "")),
        "abstract": abstract(w.get("abstract_inverted_index")),
        "topic": topic.get("display_name") or (topic.get("subfield") or {}).get("display_name") or "",
        "flag": "retracted" if w.get("is_retracted") else None,
        "refs": [short_id(r) for r in (w.get("referenced_works") or [])],
    }


_RESP_CACHE: Dict[str, Tuple[float, Any]] = {}
_RESP_TTL = 6 * 3600


class OpenAlex:
    def __init__(self, client: httpx.AsyncClient, base: str = "https://api.openalex.org",
                 email: Optional[str] = None, concurrency: int = 6, api_key: Optional[str] = None) -> None:
        self.c, self.base, self.email, self.api_key = client, base.rstrip("/"), email, api_key
        self.sem = asyncio.Semaphore(concurrency)
        self.limited = False      # set when OpenAlex answers 429 with a long Retry-After

    async def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        p = dict(params or {})
        if self.email:
            p["mailto"] = self.email
        ck = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(p.items()))
        hit = _RESP_CACHE.get(ck)
        if hit and time.time() - hit[0] < _RESP_TTL:
            return hit[1]
        if self.api_key:
            p["api_key"] = self.api_key
        async with self.sem:
            for attempt in range(3):
                try:
                    r = await self.c.get(f"{self.base}{path}", params=p)
                    if r.status_code == 404:
                        return None
                    if r.status_code == 429:
                        wait = float(r.headers.get("retry-after") or 1)
                        if wait > 10 or attempt == 2:
                            self.limited = True
                            log.warning("OpenAlex rate limit hit on %s (retry-after %ss)", path, wait)
                            return None
                        await asyncio.sleep(wait)
                        continue
                    r.raise_for_status()
                    data = r.json()
                    if len(_RESP_CACHE) > 5000:
                        _RESP_CACHE.clear()
                    _RESP_CACHE[ck] = (time.time(), data)
                    return data
                except httpx.HTTPError as e:
                    if attempt == 2:
                        log.warning("OpenAlex %s failed: %s", path, e)
                        return None
                    await asyncio.sleep(0.5)
        return None

    async def work(self, ref: str) -> Optional[Dict[str, Any]]:
        """``ref`` = ``W123``, ``doi:10.x/y`` or ``pmid:123``."""
        return await self._get(f"/works/{ref}", {"select": SELECT})

    async def search(self, q: str, n: int = 25, filt: str = "") -> List[Dict[str, Any]]:
        q = re.sub(r"[^\w\s-]", " ", q)          # OpenAlex rejects some punctuation (e.g. "?") in search
        params: Dict[str, Any] = {"search": " ".join(q.split()), "per-page": n, "select": SELECT}
        if filt:
            params["filter"] = filt
        data = await self._get("/works", params)
        return (data or {}).get("results", [])

    async def citing(self, wid: str, n: int = 25) -> List[Dict[str, Any]]:
        data = await self._get("/works", {"filter": f"cites:{wid}", "per-page": n,
                                          "sort": "cited_by_count:desc", "select": SELECT})
        return (data or {}).get("results", [])

    async def batch(self, ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list(dict.fromkeys(i for i in ids if i))
        chunks = [ids[i:i + 50] for i in range(0, len(ids), 50)]

        async def _one(chunk: List[str]) -> List[Dict[str, Any]]:
            data = await self._get("/works", {"filter": "openalex_id:" + "|".join(chunk),
                                              "per-page": 50, "select": SELECT})
            return (data or {}).get("results", [])

        out: List[Dict[str, Any]] = []
        for res in await asyncio.gather(*[_one(c) for c in chunks]):
            out.extend(res)
        return out
