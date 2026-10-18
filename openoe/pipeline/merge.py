"""Stage 3 — Merge & Deduplicate.

Fuses results from multiple retrieval routes via RRF and removes duplicates.
"""
from __future__ import annotations

from copy import copy
from typing import Any, Dict, List, Sequence

from openoe.core.models import Document
from openoe.utils.dedup import canonical_key, dedup


def merge_documents(
    docs: Sequence[Document],
    *,
    cfg: Dict[str, Any],
) -> List[Document]:
    """Apply RRF scoring, dedup, and limit."""
    strategy = cfg.get("strategy", "rrf")
    rrf_k = int(cfg.get("rrf_k", 60))
    dedup_field = cfg.get("dedup", "doi")
    limit = int(cfg.get("total_limit", 80))

    if strategy == "rrf":
        merged = _rrf(docs, rrf_k, dedup_field)
    else:
        merged = sorted(docs, key=lambda d: d.score or 0, reverse=True)

    deduped = dedup(merged, field=dedup_field)
    return deduped[:limit] if limit > 0 else deduped


# ---------------------------------------------------------------------------

def _rrf(docs: Sequence[Document], k: int, dedup_field: str) -> List[Document]:
    """Reciprocal Rank Fusion across routes."""
    # Group by route first for per-route ranking
    by_route: Dict[str, List[Document]] = {}
    for d in docs:
        by_route.setdefault(d.route, []).append(d)

    scores: Dict[str, float] = {}
    chosen: Dict[str, Document] = {}

    for route_docs in by_route.values():
        for rank, d in enumerate(route_docs, 1):
            key = canonical_key(d, dedup_field)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chosen or (d.score or 0) > (chosen[key].score or 0):
                chosen[key] = d

    result: List[Document] = []
    for key, score in scores.items():
        d = copy(chosen[key])
        d.score = score
        d.score_type = "rrf"
        result.append(d)

    return sorted(result, key=lambda d: d.score or 0, reverse=True)
