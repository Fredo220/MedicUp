"""Async LLM client with retry, backoff, reasoning control, and local debug logging."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from openoe.core.trace import LLMTrace

log = logging.getLogger(__name__)

_LOG_DIR = Path(os.getenv("OPENOE_LLM_LOG_DIR", "./logs/llm"))
_DEBUG = os.getenv("OPENOE_DEBUG", "").lower() in ("1", "true", "yes")


def _dump(step: str, request_body: Dict, response_data: Dict, elapsed_ms: float) -> None:
    """Write full prompt/response JSON for debugging. Only active when OPENOE_DEBUG=true."""
    if not _DEBUG:
        return
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        fname = _LOG_DIR / f"{ts}_{step}_{id(response_data) % 10000:04d}.json"
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "elapsed_ms": round(elapsed_ms, 1),
            "request": {
                "model": request_body.get("model"),
                "provider": request_body.get("provider"),
                "max_completion_tokens": request_body.get("max_completion_tokens"),
                "reasoning": request_body.get("reasoning"),
                "messages": request_body.get("messages"),
            },
            "response": {
                "choices": response_data.get("choices"),
                "usage": response_data.get("usage"),
                "model": response_data.get("model"),
                "id": response_data.get("id"),
            },
        }
        fname.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        log.debug("Failed to write LLM debug log: %s", e)


@dataclass
class LLMResult:
    content: str
    trace: LLMTrace


class LLMError(RuntimeError):
    """Safe, actionable provider failure shared by HTTP and WebSocket routes."""

    def __init__(self, code: str, message: str, *, status_code: Optional[int] = None,
                 retryable: bool = False, systemic: bool = True) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        self.systemic = systemic

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": str(self), "provider_status": self.status_code,
                "retryable": self.retryable}


class LLMClient:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg or {}
        retry = self.cfg.get("retry", {})
        self._retries = max(1, min(5, int(retry.get("max_retries", 3))))
        self._backoff = retry.get("backoff", [2, 4, 6])
        self._permanent_errors: Dict[str, LLMError] = {}
        self._failure_times: Dict[str, float] = {}
        self._requests = asyncio.Semaphore(4)

    def ensure_configured(self) -> None:
        """Local configuration check, without inference or a network request."""
        self._provider_settings()

    def _provider_settings(self) -> tuple[str, str, str]:
        provider = self.cfg.get("default_provider", "nebius")
        config = self.cfg.get("providers", {}).get(provider, {})
        base = str(config.get("base_url") or "").strip().rstrip("/")
        key = str(config.get("api_key") or "").strip()
        if len(key) > 1 and key[0] == key[-1] and key[0] in "\"'":
            key = key[1:-1].strip()
        if not key or key in {"your-token-factory-key", "your-nebius-api-key"}:
            raise LLMError("missing_credentials",
                           "Nebius Token Factory is not configured. Set NEBIUS_API_KEY or "
                           "NEBIUS_API_KEY_FILE in the backend environment, then restart the app.")
        if not base:
            raise LLMError("missing_credentials", f"The {provider} provider URL is missing. Check the backend configuration and restart.")
        return provider, base, key

    def _http_error(self, response: httpx.Response, model: str) -> LLMError:
        status = response.status_code
        # Inspect provider error codes for classification, never expose raw response
        # bodies: upstream errors may echo authorization or user data.
        try:
            detail = json.dumps(response.json()).lower()
        except (ValueError, TypeError):
            detail = ""
        if status == 401:
            return LLMError("authentication", "Nebius rejected the API key (HTTP 401). Check NEBIUS_API_KEY in the backend and restart.", status_code=status)
        if status == 402 or (status == 429 and any(s in detail for s in ("insufficient_quota", "budget exhausted", "credit balance", "quota_exceeded"))):
            return LLMError("quota_exceeded", "Nebius inference budget or credits are exhausted. Check the Token Factory project's billing and spending limits.", status_code=status)
        if status == 403:
            return LLMError("access_denied", "Nebius denied access (HTTP 403). Check the API key's project and model permissions.", status_code=status)
        if status == 404:
            return LLMError("model_unavailable", f"Nebius could not find configured model {model} (HTTP 404). Check the model ID available to this project.", status_code=status)
        if status == 429:
            return LLMError("rate_limited", "Nebius is rate limiting requests. Wait briefly and retry; check the project's request and token limits.", status_code=status, retryable=True)
        if status in {408, 409, 425} or status >= 500:
            return LLMError("unavailable", f"Nebius is temporarily unavailable (HTTP {status}). Retry shortly.", status_code=status, retryable=True)
        return LLMError("invalid_request", f"Nebius rejected the model request (HTTP {status}). Check the configured model and request settings.", status_code=status)

    async def _post(self, operation: str, body: Dict[str, Any], *, model: str) -> Dict[str, Any]:
        _, base, key = self._provider_settings()
        # Suppress the per-paper cascade, but allow recovery after billing or
        # model access is repaired without poisoning this process forever.
        for lookup, failed_at in list(self._failure_times.items()):
            if time.monotonic() - failed_at >= 60:
                self._permanent_errors.pop(lookup, None)
                self._failure_times.pop(lookup, None)
        for lookup in ("provider", f"model:{model}"):
            if lookup in self._permanent_errors:
                raise self._permanent_errors[lookup]
        last_error = LLMError("unavailable", "Nebius did not return a response.", retryable=True)
        for attempt in range(self._retries):
            delay = self._delay(attempt)
            async with self._requests:
                # Another in-flight call may have established an auth/billing error.
                for lookup in ("provider", f"model:{model}"):
                    if lookup in self._permanent_errors:
                        raise self._permanent_errors[lookup]
                try:
                    async with httpx.AsyncClient(timeout=120.0) as client:
                        response = await client.post(f"{base}/{operation}",
                                                     headers={"Authorization": f"Bearer {key}"}, json=body)
                    if response.is_error:
                        last_error = self._http_error(response, model)
                        if not last_error.retryable:
                            scope = ("provider" if last_error.code in {"authentication", "access_denied", "quota_exceeded"}
                                     else f"model:{model}" if last_error.code == "model_unavailable" else None)
                            if scope:
                                self._permanent_errors[scope] = last_error
                                self._failure_times[scope] = time.monotonic()
                            raise last_error
                        try:
                            delay = max(delay, min(30.0, float(response.headers.get("Retry-After", "0"))))
                        except ValueError:
                            pass
                    else:
                        try:
                            data = response.json()
                        except ValueError:
                            raise LLMError("invalid_response", "Nebius returned invalid JSON. Retry this request.", systemic=False)
                        if not isinstance(data, dict):
                            raise LLMError("invalid_response", "Nebius returned an unexpected response format.", systemic=False)
                        return data
                except httpx.RequestError:
                    last_error = LLMError("unavailable", "Cannot reach Nebius Token Factory. Check backend network access and NEBIUS_BASE_URL.", retryable=True)
            log.warning("[LLM %s] attempt %d: %s", operation, attempt + 1, last_error.code)
            if attempt + 1 < self._retries:
                await asyncio.sleep(delay)
        raise last_error

    async def chat(
        self,
        *,
        step: str,
        prompt: str,
        system: str = "You are a helpful assistant.",
        model: str,
        provider: Optional[str] = None,
        max_tokens: int = 4096,
        reasoning_effort: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> LLMResult:
        """Call an OpenAI-compatible chat endpoint (Nebius Token Factory by default).

        Args:
            max_tokens: max_completion_tokens (includes reasoning tokens for reasoning models).
            reasoning_effort: "low" | "medium" | "high" | None.
                For lightweight tasks (Y/N, keywords), use "low" to minimize reasoning overhead.
            provider: OpenRouter-only provider routing (e.g. "google-vertex").
        """
        default_prov, _, _ = self._provider_settings()

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if default_prov == "openrouter":
            if provider:
                body["provider"] = {"order": [provider], "allow_fallbacks": False}
            if reasoning_effort:
                body["reasoning"] = {"effort": reasoning_effort}

        t0 = time.perf_counter()
        data = await self._post("chat/completions", body, model=model)
        elapsed = (time.perf_counter() - t0) * 1000
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMError("invalid_response", "Nebius returned no chat choices. Retry this request.", systemic=False)
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise LLMError("truncated_response", "The model response reached its token limit before completion. Reduce the input or increase the configured output limit.", systemic=False)
        message = choice.get("message") or {}
        if not isinstance(message, dict) or choice.get("error"):
            raise LLMError("invalid_response", "Nebius returned an incomplete chat response.", systemic=False)
        if message.get("refusal") or choice.get("finish_reason") == "content_filter":
            raise LLMError("refused", "The model declined this request; no assessment was produced.", systemic=False)
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("invalid_response", "Nebius returned an empty chat response; no assessment was produced.", systemic=False)
        usage = data.get("usage") or {}
        trace = LLMTrace(step=step, model=model, provider=provider or default_prov,
                         prompt_tokens=usage.get("prompt_tokens", 0),
                         completion_tokens=usage.get("completion_tokens", 0), latency_ms=elapsed)
        log.info("[LLM %s] OK %.0fms | %s | %d+%d tok", step, elapsed, model,
                 trace.prompt_tokens, trace.completion_tokens)
        _dump(step, body, data, elapsed)
        return LLMResult(content=content.strip(), trace=trace)

    async def embed(self, texts: List[str], model: str) -> List[List[float]]:
        """Create OpenAI-compatible embeddings in bounded batches."""
        self.ensure_configured()

        async def embed_batch(start: int, batch: List[str]) -> List[List[float]]:
            data = await self._post("embeddings", {"model": model, "input": batch,
                                                  "encoding_format": "float"}, model=model)
            items = data.get("data")
            if (not isinstance(items, list) or len(items) != len(batch)
                    or any(not isinstance(item, dict) or type(item.get("index")) is not int for item in items)
                    or sorted(item["index"] for item in items) != list(range(len(batch)))):
                raise LLMError("invalid_response", "Nebius embedding indices do not match the input batch.", systemic=False)
            batch_vectors = [item.get("embedding") for item in sorted(items, key=lambda item: item["index"])]
            if (any(not isinstance(vector, list) or not vector for vector in batch_vectors)
                    or len({len(vector) for vector in batch_vectors}) != 1
                    or any(type(value) not in (int, float) or not math.isfinite(value)
                           for vector in batch_vectors for value in vector)):
                raise LLMError("invalid_response", "Nebius returned invalid or inconsistent embedding vectors.", systemic=False)
            return batch_vectors

        batches = [
            texts[start:start + 32]
            for start in range(0, len(texts), 32)
        ]
        results = await asyncio.gather(*(
            embed_batch(start, batch)
            for start, batch in zip(range(0, len(texts), 32), batches)
        ))
        vectors = [vector for batch_vectors in results for vector in batch_vectors]
        if len({len(vector) for vector in vectors}) > 1:
            raise LLMError("invalid_response", "Nebius embedding dimensions differ between batches.", systemic=False)
        return vectors

    def _delay(self, attempt: int) -> float:
        return float(self._backoff[min(attempt, len(self._backoff) - 1)]) if self._backoff else 2.0
