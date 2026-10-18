"""Provider failures must be actionable, bounded, and safe to expose to the UI."""
import asyncio

import httpx
import pytest

from openoe.llm.client import LLMClient


def client_config(key="dummy-review-key"):
    return {
        "default_provider": "nebius",
        "providers": {"nebius": {"base_url": "https://provider.invalid/v1", "api_key": key}},
        "retry": {"max_retries": 3, "backoff": [0]},
    }


def install_transport(monkeypatch, handler):
    real_client = httpx.AsyncClient
    monkeypatch.setattr("openoe.llm.client.httpx.AsyncClient", lambda **kw: real_client(
        transport=httpx.MockTransport(handler), **kw,
    ))


def test_missing_credentials_fail_before_any_http_request(monkeypatch):
    calls = []
    install_transport(monkeypatch, lambda req: calls.append(req))
    client = LLMClient(client_config(""))
    with pytest.raises(RuntimeError) as exc:
        asyncio.run(client.chat(step="queries", prompt="q", model="chat"))
    assert getattr(exc.value, "code", None) == "missing_credentials"
    assert "NEBIUS_API_KEY" in str(exc.value)
    assert calls == []


@pytest.mark.parametrize("status,body,code", [
    (401, {"detail": "bad dummy-review-key"}, "authentication"),
    (402, {"detail": "budget exhausted"}, "quota_exceeded"),
    (403, {"error": {"message": "access denied"}}, "access_denied"),
    (404, {"detail": "unknown model"}, "model_unavailable"),
    (422, {"detail": "invalid parameter"}, "invalid_request"),
    (429, {"error": {"code": "insufficient_quota"}}, "quota_exceeded"),
])
def test_permanent_errors_do_not_retry_and_shared_failures_are_suppressed(monkeypatch, status, body, code):
    calls = []
    def handle(req):
        calls.append(req)
        return httpx.Response(status, json=body)
    install_transport(monkeypatch, handle)
    client = LLMClient(client_config())
    for _ in range(2):
        with pytest.raises(RuntimeError) as exc:
            asyncio.run(client.chat(step="judge", prompt="q", model="chat"))
        assert getattr(exc.value, "code", None) == code
        assert "dummy-review-key" not in str(exc.value)
    assert len(calls) == (2 if status == 422 else 1)


def test_transient_throttle_retries_then_succeeds(monkeypatch):
    calls = []
    def handle(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(429, json={"detail": "rate limited"}, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})
    install_transport(monkeypatch, handle)
    result = asyncio.run(LLMClient(client_config()).chat(step="judge", prompt="q", model="chat"))
    assert result.content == "ok"
    assert len(calls) == 2


@pytest.mark.parametrize("vectors", [
    [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],
    [{"index": 0, "embedding": [1.0]}, {"index": 2, "embedding": [2.0]}],
    [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [2.0, 3.0]}],
    [{"index": 0, "embedding": ["oops"]}, {"index": 1, "embedding": [2.0]}],
])
def test_invalid_embedding_mapping_is_not_used_for_ranking(monkeypatch, vectors):
    install_transport(monkeypatch, lambda req: httpx.Response(200, json={"data": vectors}))
    with pytest.raises(RuntimeError):
        asyncio.run(LLMClient(client_config()).embed(["first", "second"], "embeddings"))


def test_a_known_auth_failure_stops_embedding_requests_too(monkeypatch):
    calls = []
    def handle(req):
        calls.append(req)
        return httpx.Response(401, json={"detail": "unauthorized"})
    install_transport(monkeypatch, handle)
    client = LLMClient(client_config())
    with pytest.raises(RuntimeError):
        asyncio.run(client.chat(step="queries", prompt="q", model="chat"))
    with pytest.raises(RuntimeError):
        asyncio.run(client.embed(["paper"], "embeddings"))
    assert len(calls) == 1


def test_budget_failure_is_rechecked_after_a_short_recovery_window(monkeypatch):
    now = [0.0]
    monkeypatch.setattr("openoe.llm.client.time.monotonic", lambda: now[0])
    calls = []
    def handle(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(402, json={"detail": "budget exhausted"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "recovered"}}]})
    install_transport(monkeypatch, handle)
    client = LLMClient(client_config())
    with pytest.raises(RuntimeError):
        asyncio.run(client.chat(step="judge", prompt="q", model="chat"))
    now[0] = 61.0
    result = asyncio.run(client.chat(step="judge", prompt="q", model="chat"))
    assert result.content == "recovered"
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 422])
def test_one_invalid_request_does_not_poison_later_valid_requests(monkeypatch, status):
    calls = []
    def handle(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(status, json={"detail": "input exceeds context limit"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "valid short answer"}}]})
    install_transport(monkeypatch, handle)
    client = LLMClient(client_config())
    with pytest.raises(RuntimeError):
        asyncio.run(client.chat(step="answer", prompt="long input", model="chat"))
    result = asyncio.run(client.chat(step="answer", prompt="short input", model="chat"))
    assert result.content == "valid short answer"
    assert len(calls) == 2
