"""Builders shared by the atlas-mind tests (not fixtures — plain functions).

`FakeProvider` scripts are nested lists of dicts, which is exactly the kind of
shape that makes a test unreadable at the call site.  These helpers name the
three things a test actually wants to say: it answers, it fails, or it says
nothing at all.
"""

from __future__ import annotations

from typing import Any

from atlas_core.config import AppConfig
from atlas_core.fakes import FakeProvider
from atlas_mind.chat import ChatSession
from atlas_mind.router import ProviderRouter

Turn = list[dict[str, Any]]
Script = list[Turn]


def say(text: str) -> Turn:
    """One turn where the provider says exactly this."""
    return [{"text": text}]


def say_in_two(first: str, second: str) -> Turn:
    """One turn streamed in two deltas — the shape a real provider produces."""
    return [{"text": first}, {"text": second}]


def fail(kind: str = "unavailable") -> Turn:
    """One turn where the provider fails: `unavailable` or `rate_limited`."""
    return [{"error": kind}]


def silence() -> Turn:
    """One turn where the provider streams nothing and raises nothing."""
    return []


def provider(*turns: Turn, name: str = "fake", **kwargs: Any) -> FakeProvider:
    """A scripted fake provider (one turn per call, last turn repeats)."""
    return FakeProvider(name=name, script=list(turns), **kwargs)


def session_for(
    config: AppConfig, *turns: Turn, structured: bool = False, **kwargs: Any
) -> ChatSession:
    """A ChatSession over a scripted fake provider."""
    return ChatSession(
        config,
        ProviderRouter([provider(*turns)]),
        structured=structured,
        **kwargs,
    )
