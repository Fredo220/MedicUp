"""PubMed abstract retrieval and XML parsing."""
from __future__ import annotations

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import httpx

from openoe.discovery.models import Paper

log = logging.getLogger(__name__)

_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def parse_pubmed_abstracts(xml: str) -> Dict[str, str]:
    root = ET.fromstring(xml)
    abstracts: Dict[str, str] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        if not pmid:
            continue
        parts = []
        for element in article.findall(".//Abstract/AbstractText"):
            text = "".join(element.itertext()).strip()
            if text:
                label = (element.get("Label") or "").strip()
                parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(parts).strip()
        if abstract:
            abstracts[pmid] = abstract
    return abstracts


class PubMedFetchError(RuntimeError):
    def __init__(self, message: str, abstracts: Dict[str, str]) -> None:
        super().__init__(message)
        self.abstracts = abstracts


async def fetch_abstracts(pmids: List[str]) -> Dict[str, str]:
    unique_pmids = list(dict.fromkeys(
        str(pmid).strip() for pmid in pmids if str(pmid).strip()
    ))
    abstracts: Dict[str, str] = {}
    async with httpx.AsyncClient(timeout=10.0) as client:
        for start in range(0, len(unique_pmids), 100):
            batch = unique_pmids[start:start + 100]
            try:
                response = await client.get(
                    _EFETCH_URL,
                    params={"db": "pubmed", "id": ",".join(batch), "retmode": "xml"},
                )
                response.raise_for_status()
                abstracts.update(parse_pubmed_abstracts(response.text))
            except Exception as exc:
                if isinstance(exc, httpx.HTTPStatusError):
                    detail = f"HTTP {exc.response.status_code}"
                else:
                    detail = type(exc).__name__
                message = f"PubMed abstract fetch failed: {detail}"
                log.warning("%s", message)
                raise PubMedFetchError(message, abstracts) from exc
    return abstracts


def parse_pubmed_articles(xml: str) -> List[Paper]:
    """Build Paper objects straight from PubMed EFetch XML (fallback when OpenAlex is unavailable)."""
    root = ET.fromstring(xml)
    abstracts = parse_pubmed_abstracts(xml)
    papers: List[Paper] = []
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        title_el = article.find(".//ArticleTitle")
        title = " ".join("".join(title_el.itertext()).split()) if title_el is not None else ""
        if not pmid or not title:
            continue
        doi = None
        for el in [*article.findall(".//ArticleIdList/ArticleId"), *article.findall(".//ELocationID")]:
            if (el.get("IdType") or el.get("EIdType") or "").lower() == "doi" and (el.text or "").strip():
                doi = el.text.strip()
                break
        year = None
        for path in (".//ArticleDate/Year", ".//JournalIssue/PubDate/Year", ".//JournalIssue/PubDate/MedlineDate"):
            match = re.search(r"\d{4}", article.findtext(path) or "")
            if match:
                year = int(match.group(0))
                break
        authors = []
        for author in article.findall(".//AuthorList/Author"):
            name = f"{(author.findtext('LastName') or '').strip()} {(author.findtext('Initials') or '').strip()}".strip()
            name = name or (author.findtext("CollectiveName") or "").strip()
            if name:
                authors.append(name)
        ptypes = [(pt.text or "").strip().lower() for pt in article.findall(".//PublicationTypeList/PublicationType")]
        papers.append(Paper(
            openalex_id=f"pmid:{pmid}",
            doi=doi,
            pmid=pmid,
            title=title,
            abstract=abstracts.get(pmid, ""),
            year=year,
            venue=(article.findtext(".//Journal/Title") or "").strip() or None,
            authors=authors[:10],
            is_retracted="retracted publication" in ptypes,
            type="article",
            links={
                "doi": f"https://doi.org/{doi}" if doi else None,
                "pubmed": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "openalex": None,
            },
        ))
    return papers


def _eutils_params(**params: object) -> Dict[str, object]:
    params["tool"] = "medicup"
    if os.getenv("OPENALEX_EMAIL"):
        params["email"] = os.getenv("OPENALEX_EMAIL")
    if os.getenv("NCBI_API_KEY"):
        params["api_key"] = os.getenv("NCBI_API_KEY")
    return params


async def _eutils_get(client: httpx.AsyncClient, url: str, params: Dict[str, object]) -> httpx.Response:
    """E-utilities allow 3 requests/s without a key; back off once on 429 instead of failing."""
    response = await client.get(url, params=params)
    for delay in (0.7, 1.5):
        if response.status_code != 429:
            break
        await asyncio.sleep(delay)
        response = await client.get(url, params=params)
    response.raise_for_status()
    return response


async def search_pubmed(query: str, n: int = 25) -> List[Paper]:
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await _eutils_get(client, _ESEARCH_URL, _eutils_params(
            db="pubmed", term=query, retmax=n, retmode="json", sort="relevance",
        ))
        ids = (response.json().get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return []
        response = await _eutils_get(client, _EFETCH_URL, _eutils_params(db="pubmed", id=",".join(ids), retmode="xml"))
    return parse_pubmed_articles(response.text)


async def lookup_pubmed(kind: str, value: str) -> Optional[Paper]:
    terms = {"doi": f'"{value}"[AID]', "pmid": f"{value}[PMID]", "title": f"{value}[Title]"}
    term = terms.get(kind)
    if not term:
        return None
    papers = await search_pubmed(term, 1)
    if not papers and kind == "title":
        papers = await search_pubmed(value, 1)
    return papers[0] if papers else None
