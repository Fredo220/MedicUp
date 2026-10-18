"""Seed parsing and resolution."""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple
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


_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "of", "in", "on", "to", "a", "an", "by", "or", "at",
    "trial", "study", "paper", "nejm", "lancet", "jama",
}


def _tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.casefold()) if t not in _STOPWORDS and (len(t) > 2 or t.isdigit())]


def pick_title_match(query: str, papers: List[Paper]) -> Optional[Paper]:
    """Pick the paper a clinician most likely meant by a title or trial name.

    Rejects matches that share too few words with the query, and prefers the
    exact title, then the most-cited paper (the primary report over editorials).
    """
    wanted = set(_tokens(query))
    if not wanted:
        return None
    best: Optional[Paper] = None
    best_score = float("-inf")
    normalized_query = " ".join(_tokens(query))
    for paper in papers:
        title_tokens = set(_tokens(paper.title))
        text_tokens = title_tokens | set(_tokens(paper.abstract))
        coverage = len(wanted & text_tokens) / len(wanted)
        if coverage < 0.6:
            continue
        title_coverage = len(wanted & title_tokens) / len(wanted)
        exact = 3.0 if " ".join(_tokens(paper.title)) == normalized_query else 0.0
        commentary = -10.0 if paper.type in {"editorial", "letter", "erratum"} else 0.0
        score = exact + coverage + title_coverage + commentary + 0.3 * math.log10(paper.cited_by_count + 1)
        if score > best_score:
            best, best_score = paper, score
    return best


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
            candidates = [p for p in (paper_from_work(w) for w in await client.lookup(value)) if p]
            paper = pick_title_match(value, candidates)
            if paper is None or paper.openalex_id in seen_ids:
                unresolved.append(raw)
                continue
            papers.append(paper)
            seen_ids.add(paper.openalex_id)
            continue
        if kind == "doi":
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
