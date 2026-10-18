"""Stage 2 — Multi-source Retrieval.

Fans out to all enabled retrieval routes concurrently and collects results.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from openoe.core.models import Document
from openoe.core.trace import RetrievalTrace

log = logging.getLogger(__name__)

# Source type → callable factory
SourceFn = Callable[..., List[Document]]


async def retrieve(
    routes_cfg: Dict[str, Dict[str, Any]],
    *,
    question: str,
    keywords: List[str],
    embedding: Optional[List[float]],
    source_registry: Dict[str, Any],
    requested: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run all enabled routes concurrently.

    Returns ``{"papers": [...], "guidelines": [...], "traces": [...]}``.
    """
    active = {
        name: rcfg for name, rcfg in routes_cfg.items()
        if rcfg.get("enabled") and (not requested or name in requested)
    }

    async def _run(name: str, rcfg: Dict[str, Any]) -> tuple[str, List[Document], RetrievalTrace]:
        src_name = rcfg.get("source", name)
        source = source_registry.get(src_name)
        if source is None:
            return name, [], RetrievalTrace(route=name, query_used="", result_count=0, error=f"source '{src_name}' not registered")

        mode = rcfg.get("mode", "keyword")
        limit = int(rcfg.get("limit", 10))
        t0 = time.perf_counter()
        try:
            # Merge route config with extra context (e.g. hyde_text for semantic search)
            route_config = dict(rcfg.get("config", {}))
            if extra_context:
                route_config.update(extra_context)
            docs = await source.search(
                query=question, keywords=keywords, embedding=embedding,
                mode=mode, limit=limit, route_name=name,
                config=route_config, filters=filters,
            )
            trace = RetrievalTrace(
                route=name,
                query_used=" | ".join(keywords) if keywords else question,
                result_count=len(docs),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
            return name, docs, trace
        except Exception as e:
            log.warning("Route %s failed: %s", name, e)
            trace = RetrievalTrace(
                route=name, query_used=question, result_count=0,
                latency_ms=(time.perf_counter() - t0) * 1000, error=str(e),
            )
            return name, [], trace

    tasks = [_run(n, c) for n, c in active.items()]
    results = await asyncio.gather(*tasks)

    papers: List[Document] = []
    guidelines: List[Document] = []
    traces: List[RetrievalTrace] = []

    for _name, docs, trace in results:
        traces.append(trace)
        for d in docs:
            (papers if d.doc_type == "paper" else guidelines).append(d)

    return {"papers": papers, "guidelines": guidelines, "traces": traces}
