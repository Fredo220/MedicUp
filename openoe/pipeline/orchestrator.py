"""Pipeline orchestrator — wires Stage 0→5 with tracing, progress, and error handling."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional

from openoe.core.config import apply_overrides
from openoe.core.models import Document, QueryContext, QueryRequest, QueryResponse
from openoe.llm.client import LLMClient
from openoe.llm.prompts import PromptManager
from openoe.pipeline import safety, enhance, retrieval, merge, filter, answer

log = logging.getLogger(__name__)

# Progress callback type: async fn(stage, message, data)
ProgressFn = Callable[[str, str, Dict[str, Any]], Coroutine]


async def _noop_progress(stage: str, msg: str, data: Dict[str, Any]) -> None:
    pass


class Orchestrator:
    def __init__(
        self,
        *,
        config: Dict[str, Any],
        llm: LLMClient,
        prompts: PromptManager,
        source_registry: Dict[str, Any],
        embed_fn=None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.prompts = prompts
        self.sources = source_registry
        self.embed_fn = embed_fn
        self.trace_store: Dict[str, Dict[str, Any]] = {}

    def update_config(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.llm = LLMClient(config.get("llm", {}))

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        return self.trace_store.get(trace_id)

    async def run(self, req: QueryRequest, progress: Optional[ProgressFn] = None) -> QueryResponse:
        emit = progress or _noop_progress
        cfg = apply_overrides(self.config, req.config_overrides)
        stages = cfg.get("stages", {})

        ctx = QueryContext(question=req.question.strip(), history=req.history, config=cfg)
        ctx.trace.start()
        t_total = time.perf_counter()
        log.info("┌─ Query [%s]: %s", ctx.trace.request_id[:8], ctx.question[:80])

        # ── Stage 0: Safety ──────────────────────────────────
        await emit("safety", "正在进行安全检查…", {})
        t0 = time.perf_counter()
        ctx.safety = await safety.check_safety(ctx.question, stages.get("safety", {}), self.llm, self.prompts)
        ctx.trace.time("stage0_safety_ms", (time.perf_counter() - t0) * 1000)
        flag = ctx.safety.get("flag", "medical_related") if ctx.safety else "skipped"
        log.info("├─ Stage 0 Safety: %s (%.0fms)", flag, ctx.trace.timings.get("stage0_safety_ms", 0))
        await emit("safety", f"安全检查完成: {flag}", {"flag": flag})

        if ctx.safety and flag in ("unsafe", "non_medical"):
            log.warning("└─ Pipeline stopped at safety: %s — %s", flag, ctx.safety.get("reason", ""))
            await emit("done", "安全检查未通过", {"flag": flag})
            return self._safety_response(ctx, flag, ctx.safety.get("reason", ""))

        # ── Stage 1: Enhance ─────────────────────────────────
        await emit("enhance", "正在增强查询 (HyDE + 翻译 + 关键词)…", {})
        t1 = time.perf_counter()
        enhanced = await enhance.enhance_query(
            ctx.question, stages.get("enhance", {}), self.llm, self.prompts,
            history=ctx.history, embed_fn=self.embed_fn,
        )
        ctx.hyde_text = enhanced["hyde"]
        ctx.translated = enhanced["translated"]
        ctx.keywords = enhanced["keywords"]
        ctx.search_query = enhanced["search_query"]
        ctx.embedding = enhanced["embedding"]
        ctx.trace.time("stage1_enhance_ms", (time.perf_counter() - t1) * 1000)
        log.info("├─ Stage 1 Enhance (%.0fms): search_query=%s, hyde=%s, translated=%s",
                 ctx.trace.timings["stage1_enhance_ms"],
                 ctx.search_query[:60] if ctx.search_query else "none",
                 "yes" if ctx.hyde_text else "no",
                 "yes" if ctx.translated else "no")
        await emit("enhance", "查询增强完成", {
            "search_query": ctx.search_query[:80] if ctx.search_query else None,
            "translated": ctx.translated[:80] if ctx.translated else None,
        })

        # ── Stage 2: Retrieval ───────────────────────────────
        await emit("retrieval", "正在检索文献…", {})
        t2 = time.perf_counter()
        effective_query = ctx.search_query or ctx.translated or ctx.hyde_text or ctx.question
        retrieval_keywords = list(ctx.keywords)
        if ctx.translated and (not retrieval_keywords or retrieval_keywords[0] != ctx.translated):
            retrieval_keywords.insert(0, ctx.translated)
        log.info("├─ Stage 2 Retrieval: query=%s", effective_query[:80])
        ret = await retrieval.retrieve(
            cfg.get("retrieval", {}).get("routes", {}),
            question=effective_query,
            keywords=retrieval_keywords,
            embedding=ctx.embedding,
            source_registry=self.sources,
            requested=req.sources,
            filters=req.filters or None,
            extra_context={"hyde_text": ctx.hyde_text} if ctx.hyde_text else None,
        )
        ctx.raw_papers = ret["papers"]
        ctx.raw_guidelines = ret["guidelines"]
        ctx.trace.retrieval.extend(ret["traces"])
        ctx.trace.time("stage2_retrieval_ms", (time.perf_counter() - t2) * 1000)
        for t in ret["traces"]:
            status = f"{t.result_count} results" if not t.error else f"ERROR: {t.error}"
            log.info("│  └─ route %-20s %s (%.0fms)", t.route, status, t.latency_ms)
        log.info("├─ Stage 2 total: %d papers, %d guidelines (%.0fms)",
                 len(ctx.raw_papers), len(ctx.raw_guidelines), ctx.trace.timings["stage2_retrieval_ms"])
        await emit("retrieval", f"检索到 {len(ctx.raw_papers)} 篇论文, {len(ctx.raw_guidelines)} 条指南", {
            "papers": len(ctx.raw_papers), "guidelines": len(ctx.raw_guidelines),
        })

        if ctx.trace.retrieval and all(t.error for t in ctx.trace.retrieval):
            log.error("└─ All retrieval routes failed!")
            raise RuntimeError("All retrieval routes failed: " + "; ".join(f"{t.route}: {t.error}" for t in ctx.trace.retrieval))

        # ── Stage 3: Merge ───────────────────────────────────
        await emit("merge", "正在合并去重…", {})
        t3 = time.perf_counter()
        merge_cfg = cfg.get("merge", {})
        ctx.merged_papers = merge.merge_documents(ctx.raw_papers, cfg=merge_cfg.get("papers", {}))
        ctx.merged_guidelines = merge.merge_documents(ctx.raw_guidelines, cfg=merge_cfg.get("guidelines", {}))
        ctx.trace.time("stage3_merge_ms", (time.perf_counter() - t3) * 1000)
        log.info("├─ Stage 3 Merge (%.0fms): %d→%d papers, %d→%d guidelines",
                 ctx.trace.timings["stage3_merge_ms"],
                 len(ctx.raw_papers), len(ctx.merged_papers),
                 len(ctx.raw_guidelines), len(ctx.merged_guidelines))
        await emit("merge", f"去重后 {len(ctx.merged_papers)} 篇论文, {len(ctx.merged_guidelines)} 条指南", {
            "papers": len(ctx.merged_papers), "guidelines": len(ctx.merged_guidelines),
        })

        # ── Stage 4: Filter & Extract ────────────────────────
        await emit("filter", f"正在过滤 {len(ctx.merged_papers)} 篇论文…", {"total": len(ctx.merged_papers)})
        t4 = time.perf_counter()
        filter_cfg = stages.get("filter", {})
        ctx.filtered_papers = await filter.filter_papers(
            ctx.question, ctx.merged_papers, filter_cfg.get("paper_relevance", {}), self.llm, self.prompts,
        )
        ctx.filtered_guidelines = await filter.extract_guideline_passages(
            ctx.question, ctx.merged_guidelines, filter_cfg.get("guideline_extract", {}), self.llm, self.prompts,
        )
        ctx.trace.time("stage4_filter_ms", (time.perf_counter() - t4) * 1000)
        log.info("├─ Stage 4 Filter (%.0fms): %d→%d papers, %d guidelines with chunks",
                 ctx.trace.timings["stage4_filter_ms"],
                 len(ctx.merged_papers), len(ctx.filtered_papers), len(ctx.filtered_guidelines))
        await emit("filter", f"保留 {len(ctx.filtered_papers)} 篇相关论文, {len(ctx.filtered_guidelines)} 条指南", {
            "papers_kept": len(ctx.filtered_papers), "guidelines_kept": len(ctx.filtered_guidelines),
        })

        # ── Stage 5: Answer ──────────────────────────────────
        await emit("answer", "正在生成循证回答…", {})
        t5 = time.perf_counter()
        log.info("├─ Stage 5 Answer: generating with model=%s ...", stages.get("answer", {}).get("model", "?"))
        ans = await answer.generate_answer(
            ctx.question, ctx.filtered_papers, ctx.filtered_guidelines,
            stages.get("answer", {}), self.llm, self.prompts, history=ctx.history,
        )
        ctx.answer = ans["answer"]
        ctx.markdown = ans["markdown"]
        ctx.cited_sources = ans["citations"]
        ctx.trace.llm.append(ans["trace"])
        ctx.trace.time("stage5_answer_ms", (time.perf_counter() - t5) * 1000)
        log.info("├─ Stage 5 done (%.0fms): %d citations, answer=%d chars",
                 ctx.trace.timings["stage5_answer_ms"], len(ctx.cited_sources), len(ctx.answer or ""))

        # ── Finalize ─────────────────────────────────────────
        ctx.trace.finish()
        ctx.trace.time("total_ms", (time.perf_counter() - t_total) * 1000)
        self.trace_store[ctx.trace.request_id] = self._serialize(ctx)
        log.info("└─ Done [%s] total=%.0fms | papers=%d guidelines=%d citations=%d",
                 ctx.trace.request_id[:8], ctx.trace.timings["total_ms"],
                 len(ctx.filtered_papers), len(ctx.filtered_guidelines), len(ctx.cited_sources))
        await emit("done", "回答生成完成", {"citations": len(ctx.cited_sources)})

        return QueryResponse(
            answer=ctx.answer or "",
            markdown=ctx.markdown or "",
            citations=ctx.cited_sources,
            stats=self._stats(ctx),
            trace_id=ctx.trace.request_id,
            timings=ctx.trace.timings,
        )

    # ── Helpers ───────────────────────────────────────────────

    def _safety_response(self, ctx: QueryContext, flag: str, reason: str) -> QueryResponse:
        if flag == "unsafe":
            md = f"## ⚠️ 安全警告\n\n您的问题未通过安全检查。\n\n**原因：** {reason}"
        else:
            md = f"## ℹ️ 温馨提示\n\n您的问题似乎不属于医疗健康领域。\n\n**系统判断：** {reason}"
        ctx.trace.finish()
        return QueryResponse(
            answer=md, markdown=md, citations=[],
            stats={"stopped_at": "safety", "flag": flag},
            trace_id=ctx.trace.request_id, timings=ctx.trace.timings,
        )

    @staticmethod
    def _stats(ctx: QueryContext) -> Dict[str, Any]:
        return {
            "routes_queried": [t.route for t in ctx.trace.retrieval],
            "route_errors": {t.route: t.error for t in ctx.trace.retrieval if t.error},
            "papers_retrieved": len(ctx.raw_papers),
            "papers_after_merge": len(ctx.merged_papers),
            "papers_after_filter": len(ctx.filtered_papers),
            "guidelines_retrieved": len(ctx.raw_guidelines),
            "guidelines_after_merge": len(ctx.merged_guidelines),
            "guidelines_after_filter": len(ctx.filtered_guidelines),
            "total_citations": len(ctx.cited_sources),
        }

    @staticmethod
    def _serialize(ctx: QueryContext) -> Dict[str, Any]:
        return {
            "request_id": ctx.trace.request_id,
            "started_at": ctx.trace.started_at.isoformat() if ctx.trace.started_at else None,
            "finished_at": ctx.trace.finished_at.isoformat() if ctx.trace.finished_at else None,
            "question": ctx.question,
            "hyde": ctx.hyde_text,
            "translated": ctx.translated,
            "search_query": ctx.search_query,
            "keywords": ctx.keywords,
            "safety": ctx.safety,
            "timings": ctx.trace.timings,
            "retrieval": [
                {"route": t.route, "query": t.query_used, "count": t.result_count,
                 "latency_ms": t.latency_ms, "error": t.error}
                for t in ctx.trace.retrieval
            ],
            "llm_calls": [
                {"step": t.step, "model": t.model, "provider": t.provider,
                 "tokens": t.prompt_tokens + t.completion_tokens, "latency_ms": t.latency_ms}
                for t in ctx.trace.llm
            ],
            "citations": len(ctx.cited_sources),
        }
