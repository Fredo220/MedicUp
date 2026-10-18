"""Document deduplication by DOI / file_hash / fuzzy title."""
from __future__ import annotations
from copy import copy
from difflib import SequenceMatcher
from typing import Dict, List, Sequence
from openoe.core.models import Document


def canonical_key(doc: Document, field: str = "doi") -> str:
    val = str(doc.metadata.get(field, "")).strip().lower()
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
        k = str(d.metadata.get(field, "")).strip().lower()
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
                if (d.score or 0) > (u.score or 0):
                    unique.remove(u)
                    unique.append(d)
                dup = True
                break
        if not dup:
            unique.append(d)
    return sorted(unique, key=lambda d: d.score or 0, reverse=True)
