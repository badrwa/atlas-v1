"""Provider implementations — one class per wire protocol, never per vendor."""

from atlas_mind.providers.factory import build_provider, build_providers, register_provider_kind
from atlas_mind.providers.gemini import GeminiProvider
from atlas_mind.providers.openai_compatible import OpenAiCompatibleProvider

__all__ = [
    "GeminiProvider",
    "OpenAiCompatibleProvider",
    "build_provider",
    "build_providers",
    "register_provider_kind",
]
