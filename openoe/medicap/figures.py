"""One figure per paper for the Medicap graph.

Tavily image search on the paper title, keeping only images whose URL provably belongs to that paper
(contains its DOI, DOI suffix, a long DOI number, or its PMC id). Unmatched papers get ``None`` and the
UI draws an initials badge instead — better no picture than somebody else's figure.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from openoe.sources.tavily import tavily_search

log = logging.getLogger(__name__)

_CACHE: Dict[str, Optional[Dict[str, str]]] = {}
_CACHE_MAX = 2000
ALLOWED: set = set()          # image URLs this server picked; the /img proxy serves only these
_IMG_MAX = 6 * 1024 * 1024


async def fetch_image(url: str) -> Optional[tuple]:
    """Fetch an image we selected earlier (publisher CDNs often block cross-origin embedding)."""
    if url not in ALLOWED:
        return None
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 (Medicap figure preview)"}) as c:
        r = await c.get(url)
    ctype = r.headers.get("content-type", "").split(";")[0].strip()
    if r.status_code != 200 or not ctype.startswith("image/") or len(r.content) > _IMG_MAX:
        return None
    return r.content, ctype


def match_keys(doi: str, pmcid: str = "") -> List[str]:
    doi = (doi or "").lower().strip()
    keys: List[str] = []
    if doi:
        suffix = doi.split("/", 1)[-1]
        keys += [doi, doi.replace("/", "%2f"), suffix]
        keys += re.findall(r"\d{6,}", suffix)
    num = re.sub(r"\D", "", pmcid or "")
    if num:
        keys += [f"/{num}/", f"pmc{num}"]
    return [k for k in dict.fromkeys(keys) if len(k) >= 6]


def pick_image(images: List[Any], keys: List[str]) -> Optional[Dict[str, str]]:
    for img in images:
        url = img if isinstance(img, str) else (img or {}).get("url", "")
        desc = "" if isinstance(img, str) else (img or {}).get("description") or ""
        low = url.lower()
        if low.startswith("https://") and any(k in low for k in keys):
            return {"url": url, "desc": desc[:200], "source": "tavily"}
    return None


async def _pmcid(c: httpx.AsyncClient, doi: str, pmid: str) -> str:
    q = f'DOI:"{doi}"' if doi else (f"EXT_ID:{pmid} AND SRC:MED" if pmid else "")
    if not q:
        return ""
    try:
        r = await c.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                        params={"query": q, "format": "json", "resultType": "lite", "pageSize": 1})
        res = r.json().get("resultList", {}).get("result", [])
        return (res[0].get("pmcid") or "") if res else ""
    except Exception as e:  # noqa: BLE001
        log.debug("Europe PMC lookup failed for %s: %s", doi or pmid, e)
        return ""


async def figures(papers: List[Dict[str, Any]], cfg: Dict[str, Any], *, search=tavily_search,
                  concurrency: int = 10) -> Dict[str, Optional[Dict[str, str]]]:
    web = cfg.get("stages", {}).get("web_context", {}) or {}
    api_key = web.get("api_key", "")
    out: Dict[str, Optional[Dict[str, str]]] = {}
    todo = []
    for p in papers:
        pid = str(p.get("id") or "")
        if not pid:
            continue
        if pid in _CACHE:
            out[pid] = _CACHE[pid]
        else:
            todo.append(p)
    if not todo or not api_key:
        return {**out, **{str(p["id"]): None for p in todo}}
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=20.0) as c:
        async def _one(p: Dict[str, Any]) -> None:
            async with sem:
                pmcid = await _pmcid(c, p.get("doi") or "", p.get("pmid") or "")
                keys = match_keys(p.get("doi") or "", pmcid)
                fig = None
                if keys and p.get("t"):
                    try:
                        data = await search(p["t"], api_key=api_key, max_results=5, include_images=True)
                        fig = pick_image(data.get("images", []), keys)
                    except Exception as e:  # noqa: BLE001
                        log.warning("Figure search failed for %s: %s", p["id"], e)
                        out[str(p["id"])] = None
                        return  # don't cache transient failures
                out[str(p["id"])] = fig
                if fig:
                    ALLOWED.add(fig["url"])
                if len(_CACHE) < _CACHE_MAX:
                    _CACHE[str(p["id"])] = fig
        await asyncio.gather(*[_one(p) for p in todo])
    return out
