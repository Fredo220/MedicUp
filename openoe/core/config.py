"""YAML config loading with env-var expansion and validation."""
from __future__ import annotations

import os
import re
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
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())
            break


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
    cfg = yaml.safe_load(_expand_env(raw)) or {}
    cfg.setdefault("retrieval", {"routes": {}})
    cfg.setdefault("stages", {})
    cfg.setdefault("merge", {})
    cfg.setdefault("llm", {"default_provider": "openrouter", "providers": {}})
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
