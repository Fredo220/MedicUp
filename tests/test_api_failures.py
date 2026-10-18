import asyncio
from copy import deepcopy

import httpx

from openoe.api.app import create_app
from openoe.discovery.models import DiscoverRequest


def app_without_credentials(monkeypatch):
    cfg = {
        "retrieval": {"routes": {"stub": {"enabled": False}}},
        "stages": {"answer": {"model": "stub"}},
        "discovery": {"small_model": "small", "strong_model": "strong", "embedding_model": "embed"},
        "llm": {"default_provider": "nebius", "providers": {"nebius": {
            "base_url": "https://provider.invalid/v1", "api_key": "",
        }}},
    }
    monkeypatch.setattr("openoe.api.app.load_config", lambda *args: deepcopy(cfg))
    return create_app()


def test_discovery_rejects_missing_provider_before_retrieval(monkeypatch):
    app = app_without_credentials(monkeypatch)
    async def unexpected(*args, **kwargs):
        raise AssertionError("Should stop before contacting scholarly databases")
    monkeypatch.setattr("openoe.discovery.service.resolve_seeds", unexpected)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            response = await client.post("/api/discover", json={"question": "melanoma"})
            assert response.status_code == 503
            assert response.json()["detail"]["code"] == "missing_credentials"
            assert "NEBIUS_API_KEY" in response.json()["detail"]["message"]
    asyncio.run(run())


def test_public_configuration_cannot_change_provider_destination(monkeypatch):
    app = app_without_credentials(monkeypatch)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            response = await client.patch("/api/config", json={"overrides": {
                "llm.providers.nebius.base_url": "https://attacker.invalid/v1",
            }})
            assert response.status_code == 405
            assert app.state.orchestrator.llm.cfg["providers"]["nebius"]["base_url"] == "https://provider.invalid/v1"
    asyncio.run(run())


def test_seed_budget_is_explicit_instead_of_silent_truncation():
    import pytest
    with pytest.raises(ValueError):
        DiscoverRequest(seeds=[f"W{i}" for i in range(51)])


def test_per_request_configuration_cannot_redirect_source_credentials(monkeypatch):
    app = app_without_credentials(monkeypatch)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            response = await client.post("/api/query", json={"question": "melanoma", "config_overrides": {
                "retrieval.routes.openalex_kw.config.base_url": "https://attacker.invalid",
            }})
            assert response.status_code == 422
    asyncio.run(run())


def test_websocket_reports_one_actionable_configuration_failure(monkeypatch):
    from fastapi.testclient import TestClient
    app = app_without_credentials(monkeypatch)
    with TestClient(app) as client:
        with client.websocket_connect("/ws/discover") as ws:
            ws.send_json({"question": "melanoma"})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "missing_credentials"
            assert "NEBIUS_API_KEY" in error["message"]
            assert "Judge W" not in error["message"]


def test_configured_discovery_and_cited_answer_complete_through_real_api(monkeypatch):
    import json
    cfg = {
        "retrieval": {"routes": {"stub": {"enabled": False}}},
        "stages": {"answer": {"model": "strong"}},
        "discovery": {
            "small_model": "small", "strong_model": "strong", "embedding_model": "embed",
            "signals": {name: {"enabled": name == "search"} for name in (
                "cited_by_seeds", "cites_seeds", "related", "s2_recommendation", "search", "web",
            )},
        },
        "llm": {"default_provider": "nebius", "providers": {"nebius": {
            "base_url": "https://provider.invalid/v1", "api_key": "dummy-review-key",
        }}},
    }
    class Database:
        rate_limited = False
        def __init__(self, **kwargs):
            self.errors = []
        async def search(self, query, limit):
            return [{"id": "https://openalex.org/W1", "title": "Relevant study",
                     "abstract_inverted_index": {"Observed": [0], "response.": [1]}},
                    {"id": "https://openalex.org/W2", "title": "Retracted study", "is_retracted": True}]
    monkeypatch.setattr("openoe.api.app.load_config", lambda *args: deepcopy(cfg))
    monkeypatch.setattr("openoe.discovery.service.OpenAlexClient", Database)
    requests = []
    def upstream(request):
        requests.append(request)
        body = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(body["input"])
            ]})
        system = body["messages"][0]["content"]
        if "search queries" in system:
            content = '["melanoma"]'
        elif "Candidate paper:" in body["messages"][-1]["content"]:
            content = json.dumps({"relevance": 8, "why": "The study addresses the question.",
                                  "quote": "Observed response.", "related_seeds": [], "study_design": "unclear"})
        elif "JSON list of 3" in system:
            content = "[]"
        else:
            content = "The study observed a response [^1]."
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    actual_client = httpx.AsyncClient
    def clients(**kwargs):
        if "transport" not in kwargs:
            kwargs["transport"] = httpx.MockTransport(upstream)
        return actual_client(**kwargs)
    monkeypatch.setattr("openoe.llm.client.httpx.AsyncClient", clients)
    app = create_app()
    async def run():
        async with actual_client(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
            response = await client.post("/api/discover", json={"question": "melanoma"})
            assert response.status_code == 200
            data = response.json()
            assert data["errors"] == []
            assert [p["openalex_id"] for p in data["results"]] == ["W1"]
            assert data["results"][0]["assessment_status"] == "assessed"
            assert data["stats"]["n_assessed"] == 1
            answer = await client.post("/api/followup", json={"question": "What was observed?", "papers": data["results"]})
            assert answer.status_code == 200
            assert answer.json()["assessment_status"] == "answered"
            assert answer.json()["citations"][0]["openalex_id"] == "W1"
    asyncio.run(run())
    assert all(r.url.host == "provider.invalid" for r in requests)
