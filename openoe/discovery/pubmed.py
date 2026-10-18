"""PubMed abstract retrieval and XML parsing."""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List

import httpx

log = logging.getLogger(__name__)

_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


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
