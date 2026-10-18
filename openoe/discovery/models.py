"""Models and OpenAlex conversion helpers for literature discovery."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from pydantic import BaseModel, Field

from openoe.sources.openalex import _rebuild_abstract


class Paper(BaseModel):
    openalex_id: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    title: str
    abstract: str = ""
    year: Optional[int] = None
    venue: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    cited_by_count: int = 0
    is_oa: bool = False
    oa_url: Optional[str] = None
    type: Optional[str] = None
    links: Dict[str, Optional[str]] = Field(default_factory=dict)
    referenced_works: List[str] = Field(default_factory=list, exclude=True)
    related_works: List[str] = Field(default_factory=list, exclude=True)

    def to_api_dict(self) -> Dict[str, Any]:
        data = self.model_dump(exclude={"referenced_works", "related_works"})
        data["study_type"] = study_type(self.title, self.abstract, self.type)
        return data


class DiscoverRequest(BaseModel):
    question: str = ""
    seeds: List[str] = Field(min_length=1)
    disliked: List[str] = Field(default_factory=list)
    limit: int = 20


class DiscoverResponse(BaseModel):
    seeds: List[Dict[str, Any]]
    unresolved: List[str]
    results: List[Dict[str, Any]]
    web_context: List[Dict[str, str]]
    stats: Dict[str, Any]
    timings: Dict[str, float]
    errors: List[str]


class FollowupRequest(BaseModel):
    question: str
    history: List[Dict[str, str]] = Field(default_factory=list)
    papers: List[Dict[str, Any]]


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def study_type(title: str, abstract: str = "", work_type: Optional[str] = None) -> Optional[str]:
    """Plain-language evidence type, only when the title or abstract states it."""
    t = (title or "").casefold()
    a = (abstract or "").casefold()
    if work_type in {"editorial", "letter", "erratum"}:
        return "Editorial"
    if _has(r"meta-?analys|systematic review", t):
        return "Meta-analysis"
    if _has(r"guideline|consensus|position statement|scientific statement|focused update|recommendations", t):
        return "Guideline"
    if _has(r"rationale (?:for )?and design|design and rationale|study design|baseline characteristics|protocol", t):
        return "Trial design"
    if _has(r"pooled|patient-level|participant-level|individual patient data", t):
        return "Pooled analysis"
    randomized = _has(r"randomi[sz]ed|placebo-controlled", t + " " + a)
    if randomized and _has(r"post[- ]hoc|prespecified|pre-specified|secondary analysis|subgroup|according to", t + " " + a):
        return "Trial sub-analysis"
    if randomized and work_type != "review":
        return "RCT"
    if work_type == "review" or _has(r"\breview\b", t):
        return "Review"
    if _has(r"cohort|registry|retrospective|observational|real-world|case-control|cross-sectional", t + " " + a):
        return "Observational"
    return None


def short_openalex_id(value: Any) -> str:
    if not value:
        return ""
    text = str(value).rstrip("/")
    match = re.search(r"(W\d+)$", text)
    return match.group(1) if match else text


def normalize_doi(value: Any) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    text = text.rstrip(".,;:)]}>\"'")
    return text or None


def _pmid_from_ids(ids: Any) -> Optional[str]:
    value = (ids or {}).get("pmid")
    if not value:
        return None
    match = re.search(r"(?:pubmed\.ncbi\.nlm\.nih\.gov/)?(\d+)", str(value), flags=re.I)
    return match.group(1) if match else None


def paper_from_work(work: Dict[str, Any]) -> Optional[Paper]:
    openalex_id = short_openalex_id(work.get("id"))
    title = str(work.get("title") or work.get("display_name") or "").strip()
    if not openalex_id or not title:
        return None

    raw_authors = [
        str((entry.get("author") or {}).get("display_name") or "").strip()
        for entry in (work.get("authorships") or [])
    ]
    authors = [name for name in raw_authors if name][:3]
    if len([name for name in raw_authors if name]) > 3:
        authors.append("et al.")

    doi = normalize_doi(work.get("doi"))
    pmid = _pmid_from_ids(work.get("ids"))
    location = work.get("primary_location") or {}
    source = location.get("source") or {}
    oa = work.get("open_access") or {}
    doi_url = f"https://doi.org/{quote(doi, safe='/')}" if doi else None
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{quote(pmid)}/" if pmid else None
    return Paper(
        openalex_id=openalex_id,
        doi=doi,
        pmid=pmid,
        title=title,
        abstract=_rebuild_abstract(work.get("abstract_inverted_index") or {}),
        year=work.get("publication_year"),
        venue=source.get("display_name"),
        authors=authors,
        cited_by_count=int(work.get("cited_by_count") or 0),
        is_oa=bool(oa.get("is_oa")),
        oa_url=oa.get("oa_url"),
        type=work.get("type"),
        links={
            "openalex": work.get("id"),
            "doi": doi_url,
            "pubmed": pubmed_url,
        },
        referenced_works=[short_openalex_id(item) for item in (work.get("referenced_works") or [])],
        related_works=[short_openalex_id(item) for item in (work.get("related_works") or [])],
    )
