from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from inspect_ai.model import GenerateConfig, Model, get_model

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


def get_runtime_config(runtime: str | None = None) -> dict[str, Any]:
    """Return a configured runtime profile."""
    cfg = _load().get("runtime", {})
    if runtime is None:
        runtime = cfg.get("default", "t4_hf")
    profile = cfg.get(runtime)
    if profile is None:
        valid = sorted(k for k in cfg if k != "default")
        raise ValueError(f"Unknown runtime '{runtime}'. Valid: {', '.join(valid)}")
    return profile


def get_runtime_model_name(
    kind: str,
    *,
    runtime: str | None = None,
    model_key: str | None = None,
    model_name: str | None = None,
) -> str:
    """Resolve a runtime model alias to an Inspect AI model string."""
    if model_name is not None:
        return model_name

    profile = get_runtime_config(runtime)
    provider_name = profile.get("provider")
    if not provider_name:
        raise ValueError(f"Runtime '{runtime or profile}' does not define provider")

    key = model_key or profile.get(f"{kind}_model")
    if not key:
        raise ValueError(f"Runtime profile does not define {kind}_model")

    models = get_provider_models(provider_name)
    try:
        return models[key]
    except KeyError as exc:
        valid = ", ".join(sorted(models))
        raise ValueError(
            f"Unknown {kind} model key '{key}' for provider '{provider_name}'. "
            f"Valid: {valid}"
        ) from exc


def get_runtime_model_args(runtime: str | None = None) -> dict[str, Any]:
    """Return model constructor args for the runtime profile."""
    return dict(get_runtime_config(runtime).get("model_args", {}))


def get_runtime_generate_config(
    runtime: str | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> GenerateConfig:
    """Return GenerateConfig merged from runtime defaults and overrides."""
    params = dict(get_runtime_config(runtime).get("generate", {}))
    if overrides:
        params.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
    return GenerateConfig(**params)


def get_runtime_guard_config(runtime: str | None = None) -> dict[str, Any]:
    """Return guard generation parameters for the runtime profile."""
    return dict(get_runtime_config(runtime).get("guard", {}))


def build_runtime_model(
    kind: str,
    *,
    runtime: str | None = None,
    model_key: str | None = None,
    model_name: str | None = None,
) -> Model:
    """Build an Inspect AI model from the runtime profile."""
    resolved_model_name = get_runtime_model_name(
        kind,
        runtime=runtime,
        model_key=model_key,
        model_name=model_name,
    )
    model_args = (
        {}
        if model_name is not None and runtime is None
        else get_runtime_model_args(runtime)
    )
    return get_model(resolved_model_name, **model_args)
