"""Document deduplication by DOI / file_hash / fuzzy title."""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Dict, List, Sequence
from openoe.core.models import Document


_DOI_PREFIX = re.compile(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", re.I)


def normalize_doi(doi) -> str:
    """``https://doi.org/10.1/X`` / ``doi:10.1/X`` / ``10.1/X`` → ``10.1/x``."""
    return _DOI_PREFIX.sub("", str(doi or "").strip()).lower()


def _key_value(doc: Document, field: str) -> str:
    val = doc.metadata.get(field, "")
    return normalize_doi(val) if field == "doi" else str(val or "").strip().lower()


def canonical_key(doc: Document, field: str = "doi") -> str:
    val = _key_value(doc, field)
    return f"{field}:{val}" if val else f"id:{doc.id.lower()}"


def dedup(
    docs: Sequence[Document],
    field: str = "doi",
    fuzzy_title: bool = True,
    threshold: float = 0.92,
) -> List[Document]:
    by_key: Dict[str, Document] = {}
    overflow: List[Document] = []
    for d in docs:
        k = _key_value(d, field)
        if k:
            existing = by_key.get(k)
            if existing is None or (d.score or 0) > (existing.score or 0):
                by_key[k] = d
        else:
            overflow.append(d)
    merged = list(by_key.values()) + overflow
    if not fuzzy_title:
        return sorted(merged, key=lambda d: d.score or 0, reverse=True)
    unique: List[Document] = []
    for d in merged:
        t = " ".join(d.title.lower().split())
        dup = False
        for u in unique:
            if SequenceMatcher(None, t, " ".join(u.title.lower().split())).ratio() >= threshold:
                keep, other = (d, u) if (d.score or 0) > (u.score or 0) else (u, d)
                _absorb(keep, other)
                if keep is d:
                    unique.remove(u)
                    unique.append(d)
                dup = True
                break
        if not dup:
            unique.append(d)
    return sorted(unique, key=lambda d: d.score or 0, reverse=True)


def _absorb(keep: Document, other: Document) -> None:
    """Fill ``keep``'s metadata gaps from a fuzzy duplicate and union their ``found_via`` routes."""
    meta = dict(keep.metadata)
    for k, v in other.metadata.items():
        if v and not meta.get(k):
            meta[k] = v
    via = list(keep.metadata.get("found_via") or [keep.route]) + list(other.metadata.get("found_via") or [other.route])
    meta["found_via"] = list(dict.fromkeys(r for r in via if r))
    keep.metadata = meta

