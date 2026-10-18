"""FastAPI application — routes, WebSocket progress, static frontend."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from openoe.core.config import load_config, validate_config
from openoe.core.models import QueryRequest, QueryResponse
from openoe.discovery.followup import answer_followup
from openoe.discovery.models import DiscoverRequest, DiscoverResponse, FollowupRequest
from openoe.discovery.service import discover
from openoe.llm.client import LLMClient, LLMError
from openoe.llm.prompts import PromptManager
from openoe.medicap import discover as medicap
from openoe.medicap import figures as medicap_figures_mod
from openoe.pipeline.orchestrator import Orchestrator
from openoe.sources.openalex import OpenAlexSource
from openoe.sources.tavily import TavilySource

log = logging.getLogger(__name__)


class MedicapResolveRequest(BaseModel):
    lines: List[str] = Field(max_length=50)


class MedicapFiguresRequest(BaseModel):
    papers: List[Dict[str, Any]]


class MedicapDiscoverRequest(BaseModel):
    question: str = ""
    seeds: List[str] = Field(max_length=50)
    limit: int = Field(40, ge=5, le=100)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(config_path: Optional[str] = None) -> FastAPI:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(config_path)
    validate_config(cfg)

    app = FastAPI(title="MedicUp", version="0.2.0", description="Evidence-based medical literature search engine")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    prompts = PromptManager()
    llm = LLMClient(cfg.get("llm", {}))

    source_registry = _build_sources(cfg)

    orch = Orchestrator(config=cfg, llm=llm, prompts=prompts, source_registry=source_registry)

    app.state.config = cfg
    app.state.orchestrator = orch
    app.state.llm = llm
    app.state.prompts = prompts

    @app.exception_handler(LLMError)
    async def provider_error(request: Request, error: LLMError):
        return JSONResponse(status_code=503, content={"detail": error.to_dict()})

    # ── Routes ────────────────────────────────────────────────

    @app.get("/health", tags=["system"])
    async def health():
        routes = cfg.get("retrieval", {}).get("routes", {})
        try:
            llm.ensure_configured()
            provider = {"configured": True, "connectivity_checked": False}
        except LLMError as error:
            provider = {"configured": False, "connectivity_checked": False, **error.to_dict()}
        return {
            "status": "ok" if provider["configured"] else "degraded",
            "llm": provider,
            "version": "0.2.0",
            "routes_enabled": [k for k, v in routes.items() if v.get("enabled")],
        }

    @app.post("/api/query", response_model=QueryResponse, tags=["rag"])
    async def query(req: QueryRequest):
        return await orch.run(req)

    @app.post("/api/search", tags=["rag"])
    async def search(req: QueryRequest):
        resp = await orch.run(req)
        return {"citations": resp.citations, "stats": resp.stats, "trace_id": resp.trace_id, "timings": resp.timings}

    @app.post("/api/discover", response_model=DiscoverResponse, tags=["discovery"])
    async def discover_papers(req: DiscoverRequest):
        return await discover(
            req,
            llm=llm,
            prompts=prompts,
            cfg=cfg.get("discovery", {}),
        )

    @app.post("/api/followup", tags=["discovery"])
    async def followup(req: FollowupRequest):
        if not medicap.llm_available(app.state.config):
            raise HTTPException(status_code=503, detail="No LLM configured: set NEBIUS_API_KEY in .env and restart.")
        return await answer_followup(
            llm=llm,
            prompts=prompts,
            cfg=cfg.get("discovery", {}),
            question=req.question,
            history=req.history,
            papers=req.papers,
        )

    @app.get("/api/sources", tags=["rag"])
    async def sources():
        routes = cfg.get("retrieval", {}).get("routes", {})
        return [
            {"name": k, "enabled": v.get("enabled", False), "source": v.get("source", k), "mode": v.get("mode", "keyword")}
            for k, v in routes.items()
        ]

    @app.get("/api/config", tags=["admin"])
    async def get_config():
        return _redact(cfg)

    @app.get("/api/traces/{trace_id}", tags=["admin"])
    async def get_trace(trace_id: str):
        t = orch.get_trace(trace_id)
        if t is None:
            return {"error": "trace not found"}
        return t

    # ── Medicap: seed-based discovery ─────────────────────────
    @app.post("/api/medicap/resolve", tags=["medicap"])
    async def medicap_resolve(body: MedicapResolveRequest):
        return {"results": await medicap.resolve(body.lines, app.state.config)}

    @app.post("/api/medicap/discover", tags=["medicap"])
    async def medicap_discover(body: MedicapDiscoverRequest):
        try:
            return await medicap.discover(body.question, body.seeds, app.state.config, llm=llm, limit=body.limit)
        except medicap.OpenAlexLimited as e:
            return JSONResponse(status_code=503, content={"detail": str(e)})

    @app.post("/api/medicap/figures", tags=["medicap"])
    async def medicap_figures(body: MedicapFiguresRequest):
        return {"figures": await medicap_figures_mod.figures(body.papers[:60], app.state.config)}

    @app.get("/api/medicap/img", tags=["medicap"], include_in_schema=False)
    async def medicap_img(u: str):
        got = await medicap_figures_mod.fetch_image(u)
        if not got:
            return Response(status_code=404)
        return Response(content=got[0], media_type=got[1], headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/medicap/status", tags=["medicap"])
    async def medicap_status():
        c = app.state.config
        return {"judge": medicap.llm_available(c),
                "llm": medicap.llm_available(c),
                "tavily": bool((c.get("stages", {}).get("web_context", {}) or {}).get("api_key"))}

    # ── WebSocket: real-time progress ─────────────────────────
    @app.websocket("/ws/query")
    async def ws_query(ws: WebSocket):
        await ws.accept()
        try:
            raw = await ws.receive_text()
            data = json.loads(raw)
            req = QueryRequest(**data)

            async def send_progress(stage: str, message: str, detail: Dict[str, Any]) -> None:
                try:
                    await ws.send_json({"type": "progress", "stage": stage, "message": message, "data": detail})
                except Exception:
                    pass

            response = await orch.run(req, progress=send_progress)
            await ws.send_json({"type": "result", "data": response.model_dump()})
        except LLMError as error:
            await ws.send_json({"type": "error", **error.to_dict()})
        except WebSocketDisconnect:
            log.info("WebSocket client disconnected")
        except Exception as e:
            log.error("WebSocket error: %s", e)
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    @app.websocket("/ws/discover")
    async def ws_discover(ws: WebSocket):
        await ws.accept()
        try:
            raw = await ws.receive_text()
            req = DiscoverRequest(**json.loads(raw))

            async def send_progress(stage: str, message: str, detail: Dict[str, Any]) -> None:
                try:
                    await ws.send_json({"type": "progress", "stage": stage, "message": message, "data": detail})
                except Exception:
                    pass

            response = await discover(
                req,
                progress=send_progress,
                llm=llm,
                prompts=prompts,
                cfg=cfg.get("discovery", {}),
            )
            await ws.send_json({"type": "result", "data": response})
        except LLMError as error:
            await ws.send_json({"type": "error", **error.to_dict()})
        except WebSocketDisconnect:
            log.info("Discovery WebSocket client disconnected")
        except Exception as e:
            log.error("Discovery WebSocket error: %s", e)
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    # ── Frontend ──────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def landing():
        return FileResponse(_STATIC_DIR / "landing.html", media_type="text/html")

    @app.get("/app", include_in_schema=False)
    @app.get("/medicap", include_in_schema=False)
    async def medicap_ui():
        return FileResponse(_STATIC_DIR / "medicap.html", media_type="text/html")

    @app.get("/chat", include_in_schema=False)
    @app.get("/qa", include_in_schema=False)
    async def index():
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    return app


# ── Helpers ───────────────────────────────────────────────────

def _build_sources(cfg: dict) -> dict:
    """Create source instances based on configured routes."""
    registry: dict = {}
    routes = cfg.get("retrieval", {}).get("routes", {})

    if any(r.get("source") == "openalex" and r.get("enabled") for r in routes.values()):
        registry["openalex"] = OpenAlexSource()
    if any(r.get("source") == "tavily" and r.get("enabled") for r in routes.values()):
        registry["tavily"] = TavilySource()

    return registry


def _redact(cfg: dict) -> dict:
    """Redact sensitive values in config."""
    import copy
    out = copy.deepcopy(cfg)
    for key in list(out.keys()):
        if isinstance(out[key], dict):
            out[key] = _redact(out[key])
        elif isinstance(out[key], str) and any(s in key.lower() for s in ("password", "api_key", "secret", "token")):
            out[key] = "***"
    return out
