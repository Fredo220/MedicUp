"""Execution trace for observability."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4


@dataclass
class RetrievalTrace:
    route: str
    query_used: str
    result_count: int
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class LLMTrace:
    step: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class ExecutionTrace:
    request_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    timings: Dict[str, float] = field(default_factory=dict)
    retrieval: List[RetrievalTrace] = field(default_factory=list)
    llm: List[LLMTrace] = field(default_factory=list)

    def start(self) -> None:
        self.started_at = datetime.now(timezone.utc)

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc)

    def time(self, name: str, ms: float) -> None:
        self.timings[name] = round(ms, 2)
