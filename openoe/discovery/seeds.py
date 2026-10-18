"""Seed parsing and resolution."""
from __future__ import annotations

import re
from typing import List, Tuple
from urllib.parse import unquote, urlparse

from openoe.discovery.models import Paper, paper_from_work
from openoe.discovery.openalex import OpenAlexClient

_DOI_RE = re.compile(r"10\.\d{4,9}/\S+", re.I)
_TRAILING_PUNCTUATION = ".,;:!?)]}>\"'"


def _clean_doi(value: str) -> str:
    return value.rstrip(_TRAILING_PUNCTUATION)


def parse_seed(seed: str) -> Tuple[str, str]:
    value = seed.strip()
    parsed = urlparse(value if "://" in value else "")
    host = parsed.netloc.lower()
    if "doi.org" in host:
        match = _DOI_RE.search(unquote(parsed.path))
        if match:
            return "doi", _clean_doi(match.group(0))
    if "pubmed.ncbi.nlm.nih.gov" in host:
        match = re.search(r"/(\d+)(?:/|$)", parsed.path)
        if match:
            return "pmid", match.group(1)
    if "openalex.org" in host:
        match = re.search(r"/(W\d+)(?:/|$)", parsed.path, flags=re.I)
        if match:
            return "openalex", match.group(1).upper()
    if re.fullmatch(r"\d+", value):
        return "pmid", value
    if re.fullmatch(r"W\d+", value, flags=re.I):
        return "openalex", value.upper()
    doi_match = _DOI_RE.search(value)
    if doi_match:
        return "doi", _clean_doi(doi_match.group(0))
    return "title", value


async def resolve_seeds(
    strings: List[str], client: OpenAlexClient,
) -> Tuple[List[Paper], List[str]]:
    papers: List[Paper] = []
    unresolved: List[str] = []
    seen_keys: set[Tuple[str, str]] = set()
    seen_ids: set[str] = set()
    unique = []
    for raw in strings[:10]:
        kind, value = parse_seed(raw)
        dedupe_key = (kind, value.casefold())
        if value and dedupe_key not in seen_keys:
            seen_keys.add(dedupe_key)
            unique.append((raw, kind, value))

    for raw, kind, value in unique:
        if kind == "title":
            matches = await client.search(value, 1)
            work = matches[0] if matches else {}
        elif kind == "doi":
            work = await client.get_work(f"doi:{value}")
        elif kind == "pmid":
            work = await client.get_work(f"pmid:{value}")
        else:
            work = await client.get_work(value)
        paper = paper_from_work(work)
        if paper is None or paper.openalex_id in seen_ids:
            unresolved.append(raw)
            continue
        papers.append(paper)
        seen_ids.add(paper.openalex_id)
    return papers, unresolved
