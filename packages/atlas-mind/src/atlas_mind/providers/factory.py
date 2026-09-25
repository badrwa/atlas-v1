"""Build providers from configuration — the only place that knows about `kind`.

Adding a new backend = one class + one branch here.  Nothing else changes.
"""

from __future__ import annotations

import logging

import httpx

from atlas_core.config import AppConfig, ProviderConfig
from atlas_core.contracts import LlmProvider
from atlas_core.errors import ConfigError
from atlas_mind.providers.gemini import GeminiProvider
from atlas_mind.providers.openai_compatible import OpenAiCompatibleProvider

log = logging.getLogger(__name__)

_REGISTRY: dict[str, type] = {
    "openai_compatible": OpenAiCompatibleProvider,
    "gemini": GeminiProvider,
}


def register_provider_kind(kind: str, cls: type) -> None:
    """Extension point (used by tests and by Atlas's own plugin packs)."""
    _REGISTRY[kind] = cls


def build_provider(
    config: ProviderConfig,
    *,
    env: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> LlmProvider:
    """Instantiate one provider from its config entry."""
    try:
        cls = _REGISTRY[config.kind]
    except KeyError as exc:
        raise ConfigError(
            f"unknown provider kind {config.kind!r} (known: {', '.join(sorted(_REGISTRY))})"
        ) from exc
    return cls(config, api_key=config.api_key(env), client=client)


def build_providers(
    config: AppConfig,
    *,
    env: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> list[LlmProvider]:
    """Every usable provider, in the configured order (missing keys are skipped)."""
    providers = [
        build_provider(entry, env=env, client=client) for entry in config.ordered_providers(env)
    ]
    if not providers:
        log.warning("no providers configured — add a key to .env (see .env.example)")
    return providers


__all__ = ["build_provider", "build_providers", "register_provider_kind"]
