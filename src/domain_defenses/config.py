from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
_config: dict[str, Any] | None = None


def _load() -> dict[str, Any]:
    global _config
    if _config is None:
        with open(_CONFIG_PATH) as f:
            _config = yaml.safe_load(f) or {}
    return _config


def get_thinking_params(mode: str) -> dict[str, Any]:
    """Return max_tokens and system_suffix for the given thinking mode."""
    thinking = _load().get("thinking", {})
    params = thinking.get(mode)
    if params is None:
        valid = sorted(k for k in thinking if k != "default")
        raise ValueError(f"Unknown thinking mode '{mode}'. Valid: {', '.join(valid)}")
    return params


def get_provider_models(provider: str | None = None) -> dict[str, str]:
    """Return the model-name → inspect-ai model string map for a provider."""
    cfg = _load().get("provider", {})
    if provider is None:
        provider = cfg.get("default", "ollama")
    prov = cfg.get(provider)
    if prov is None:
        valid = sorted(k for k in cfg if k != "default")
        raise ValueError(f"Unknown provider '{provider}'. Valid: {', '.join(valid)}")
    return prov.get("models", {})
