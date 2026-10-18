"""Async LLM client with retry, backoff, reasoning control, and local debug logging."""
from __future__ import annotations

import asyncio
import json
import logging
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


class LLMClient:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg or {}
        retry = self.cfg.get("retry", {})
        self._retries = int(retry.get("max_retries", 3))
        self._backoff = retry.get("backoff", [2, 4, 6])

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
        """Call LLM via OpenRouter.

        Args:
            max_tokens: max_completion_tokens (includes reasoning tokens for reasoning models).
            reasoning_effort: "low" | "medium" | "high" | None.
                For lightweight tasks (Y/N, keywords), use "low" to minimize reasoning overhead.
            provider: OpenRouter provider routing (e.g. "google-vertex").
        """
        default_prov = self.cfg.get("default_provider", "openrouter")
        prov_cfg = self.cfg.get("providers", {}).get(default_prov, {})
        base = str(prov_cfg.get("base_url", "")).rstrip("/")
        key = prov_cfg.get("api_key", "")
        if not base or not key:
            raise RuntimeError(f"LLM provider '{default_prov}' missing base_url or api_key")

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
        }
        if provider:
            body["provider"] = {"order": [provider], "allow_fallbacks": False}
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}

        t0 = time.perf_counter()
        last_err = ""

        for attempt in range(self._retries):
            try:
                async with httpx.AsyncClient(timeout=120.0) as c:
                    r = await c.post(
                        f"{base}/chat/completions",
                        headers={"Authorization": f"Bearer {key}"},
                        json=body,
                    )
                    r.raise_for_status()
                    data = r.json()

                elapsed = (time.perf_counter() - t0) * 1000

                choice = data.get("choices", [{}])[0]
                if choice.get("error"):
                    last_err = str(choice["error"])
                    log.warning("[LLM %s] attempt %d error: %s", step, attempt + 1, last_err)
                    _dump(step, body, data, elapsed)
                    if attempt < self._retries - 1:
                        await asyncio.sleep(self._delay(attempt))
                        continue
                    raise RuntimeError(last_err)

                content = (choice.get("message") or {}).get("content") or ""
                content = content.strip()
                if not content:
                    last_err = "empty response"
                    log.warning("[LLM %s] attempt %d empty | finish_reason=%s | model=%s",
                                step, attempt + 1, choice.get("finish_reason"), data.get("model", model))
                    _dump(step, body, data, elapsed)
                    if attempt < self._retries - 1:
                        await asyncio.sleep(self._delay(attempt))
                        continue
                    raise RuntimeError(last_err)

                usage = data.get("usage", {})
                ptok = usage.get("prompt_tokens", 0)
                ctok = usage.get("completion_tokens", 0)
                log.info("[LLM %s] OK %.0fms | %s via %s | %d+%d tok | %s",
                         step, elapsed, model, provider or "auto", ptok, ctok,
                         content[:60].replace("\n", " "))
                _dump(step, body, data, elapsed)

                trace = LLMTrace(
                    step=step, model=model, provider=provider or default_prov,
                    prompt_tokens=ptok, completion_tokens=ctok, latency_ms=elapsed,
                )
                return LLMResult(content=content, trace=trace)

            except httpx.HTTPStatusError as e:
                last_err = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                log.warning("[LLM %s] attempt %d: %s", step, attempt + 1, last_err)
            except Exception as e:
                last_err = str(e)
                log.warning("[LLM %s] attempt %d: %s", step, attempt + 1, last_err)

            if attempt < self._retries - 1:
                await asyncio.sleep(self._delay(attempt))

        raise RuntimeError(f"LLM '{step}' failed after {self._retries} retries: {last_err}")

    def _delay(self, attempt: int) -> float:
        return float(self._backoff[min(attempt, len(self._backoff) - 1)]) if self._backoff else 2.0