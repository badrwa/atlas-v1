"""Atlas exception hierarchy — one base class so callers can catch Atlas, not Python."""

from __future__ import annotations


class AtlasError(Exception):
    """Base class for every Atlas-specific error."""


class ConfigError(AtlasError):
    """Configuration is missing, malformed or contradictory."""


class ProviderError(AtlasError):
    """An LLM/ASR/TTS provider failed in a way the caller should know about."""


class ProviderUnavailable(ProviderError):
    """Provider is down, unreachable, or not configured (key missing)."""


class RateLimited(ProviderError):
    """Provider asked us to slow down (HTTP 429)."""

    def __init__(self, message: str, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class QuotaExhausted(ProviderError):
    """Our own daily budget for this provider is used up."""


class AuthFailed(ProviderError):
    """The provider rejected our credentials — tell the human, don't retry."""


class AudioError(AtlasError):
    """Microphone/speaker problem (device busy, unplugged, wrong format)."""


class VaultError(AtlasError):
    """Obsidian vault missing, not writable, or a write violated the contract."""


class SkillError(AtlasError):
    """A skill refused or failed to perform its action."""


class PermissionDenied(SkillError):
    """The speaker/permission gate refused this action. Never retried automatically."""


class Unsupported(AtlasError):
    """Feature not available in this build/level yet."""


__all__ = [
    "AtlasError",
    "AudioError",
    "AuthFailed",
    "ConfigError",
    "PermissionDenied",
    "ProviderError",
    "ProviderUnavailable",
    "QuotaExhausted",
    "RateLimited",
    "SkillError",
    "Unsupported",
    "VaultError",
]
