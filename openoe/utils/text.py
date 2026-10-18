"""Text cleaning and chunking utilities."""
from __future__ import annotations
import re
from typing import List

_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return _WS.sub(" ", (text or "")).strip()


def truncate(text: str, limit: int = 900) -> str:
    """Truncate to *limit* chars on a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …"


def extract_keywords_heuristic(text: str, max_kw: int = 10) -> List[str]:
    """Simple regex-based keyword extraction (fallback)."""
    tokens = re.findall(r"[A-Za-z\u4e00-\u9fff]{2,}", text)
    seen: set[str] = set()
    out: List[str] = []
    for t in tokens:
        low = t.lower()
        if low not in seen and len(low) > 2:
            seen.add(low)
            out.append(t)
        if len(out) >= max_kw:
            break
    return out
