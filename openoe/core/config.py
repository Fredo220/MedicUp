"""YAML config loading with env-var expansion and validation."""
from __future__ import annotations

import os
import re
import shlex
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_ENV = re.compile(r"\$\{([A-Z0-9_]+)(?::([^}]*))?\}")


def _load_dotenv() -> None:
    """Load .env file from project root into os.environ (if it exists)."""
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if candidate.is_file():
            for line in candidate.read_text("utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].lstrip()
                key, _, val = line.partition("=")
                if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
                    continue
                parts = shlex.split(val, comments=True)
                os.environ.setdefault(key.strip(), " ".join(parts))
            break
    for name in ("NEBIUS_API_KEY", "TAVILY_API_KEY", "OPENALEX_API_KEY"):
        path = os.getenv(f"{name}_FILE")
        if path and not os.getenv(name):
            try:
                value = Path(path).expanduser().read_text("utf-8").strip()
            except OSError:
                raise ValueError(f"Cannot read {name}_FILE. Check the backend secret mount and permissions.") from None
            if "\n" in value or "\r" in value:
                raise ValueError(f"{name}_FILE must contain a single API key.")
            os.environ[name] = value


def _expand_env(text: str) -> str:
    return _ENV.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), text)


def load_config(path: Optional[str] = None, *, default: str = "pipeline.yaml") -> Dict[str, Any]:
    """Load a YAML config file with ``${ENV:default}`` expansion."""
    _load_dotenv()
    if path:
        p = Path(path)
        raw = p.read_text("utf-8") if p.exists() else resources.files("openoe.config").joinpath(path).read_text("utf-8")
    else:
        raw = resources.files("openoe.config").joinpath(default).read_text("utf-8")
    def expand(value: Any) -> Any:
        if isinstance(value, str):
            return _expand_env(value)
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value
    cfg = expand(yaml.safe_load(raw) or {})
    cfg.setdefault("retrieval", {"routes": {}})
    cfg.setdefault("stages", {})
    cfg.setdefault("merge", {})
    cfg.setdefault("llm", {"default_provider": "nebius", "providers": {}})
    return cfg


def deep_merge(base: Dict, override: Dict) -> Dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: Dict, overrides: Dict[str, Any]) -> Dict:
    """Apply dot-notation overrides like ``{"stages.safety.enabled": false}``."""
    out = deepcopy(cfg)
    for dotted, val in overrides.items():
        parts = dotted.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = val
    return out


def validate_config(cfg: Dict) -> None:
    if not cfg.get("retrieval", {}).get("routes"):
        raise ValueError("Config must define retrieval.routes")
    if not cfg.get("stages", {}).get("answer"):
        raise ValueError("Config must define stages.answer")
