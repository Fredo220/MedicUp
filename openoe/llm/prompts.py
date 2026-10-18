"""Prompt template loading and rendering."""
from __future__ import annotations

from importlib import resources
from typing import Any, Dict

import yaml


class PromptManager:
    def __init__(self) -> None:
        raw = resources.files("openoe.config").joinpath("prompts.yaml").read_text("utf-8")
        self._store: Dict[str, str] = yaml.safe_load(raw) or {}

    def get(self, name: str, default: str = "") -> str:
        v = self._store.get(name, default)
        if not isinstance(v, str):
            raise TypeError(f"Prompt '{name}' is not a string")
        return v

    def render(self, name: str, **kw: Any) -> str:
        return self.get(name).format(**kw)
