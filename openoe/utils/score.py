"""Score normalisation to a 1–100 % scale."""
from __future__ import annotations

from typing import Optional, Sequence

from openoe.core.models import Document


def to_percent(score: Optional[float], top: Optional[float]) -> int:
    """Map ``score`` onto 1–100 relative to ``top`` (the best possible / best observed score)."""
    if not score or not top or top <= 0 or score <= 0:
        return 1
    return max(1, min(100, round(100 * score / top)))


def assign_percent(docs: Sequence[Document]) -> None:
    """Set ``score_pct`` on each doc relative to the best score in the set (best = 100 %)."""
    top = max((d.score or 0.0 for d in docs), default=0.0)
    for d in docs:
        d.score_pct = to_percent(d.score, top)
