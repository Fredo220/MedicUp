"""Core data models used across the entire pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from openoe.core.trace import ExecutionTrace

# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------

DocType = Literal["paper", "guideline"]


@dataclass
class Document:
    """统一文档模型 — 论文和指南共用。"""

    id: str
    source: str                     # "openalex" | "local_papers" | "guidelines"
    route: str                      # "openalex_kw" | "local_papers_vec" | ...
    doc_type: DocType
    title: str
    content: str                    # 论文=abstract, 指南=全文
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = None
    score_type: Optional[str] = None
    relevance: Optional[bool] = None
    relevant_chunks: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Pipeline context (mutable, passed through stages)
# ---------------------------------------------------------------------------

@dataclass
class QueryContext:
    """Flows through Stage 0→5, each stage enriches it."""

    question: str
    history: List[Dict[str, str]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

    # Stage 0 — Safety
    safety: Optional[Dict[str, str]] = None       # {"flag": ..., "reason": ...}

    # Stage 1 — Enhance
    hyde_text: Optional[str] = None
    translated: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    search_query: Optional[str] = None
    embedding: Optional[List[float]] = None

    # Stage 2 — Retrieval (raw, per-route)
    raw_papers: List[Document] = field(default_factory=list)
    raw_guidelines: List[Document] = field(default_factory=list)

    # Stage 3 — Merge
    merged_papers: List[Document] = field(default_factory=list)
    merged_guidelines: List[Document] = field(default_factory=list)

    # Stage 4 — Filter
    filtered_papers: List[Document] = field(default_factory=list)
    filtered_guidelines: List[Document] = field(default_factory=list)

    # Stage 5 — Answer
    answer: Optional[str] = None
    markdown: Optional[str] = None
    cited_sources: List[Dict[str, Any]] = field(default_factory=list)

    # Observability
    trace: ExecutionTrace = field(default_factory=ExecutionTrace)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    sources: Optional[List[str]] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, str]] = Field(default_factory=list)
    config_overrides: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    markdown: str
    citations: List[Dict[str, Any]]
    stats: Dict[str, Any]
    trace_id: str
    timings: Dict[str, float]
