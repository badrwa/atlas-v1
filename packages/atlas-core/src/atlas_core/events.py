"""Typed events + an async event bus.

Rule of the house (architecture doc D3): anything user-visible travels as an
event.  The core publishes; UI, logger, tray and vault-log subscribe.  The core
never imports a subscriber.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Event:
    """Base event. `topic` is stable API — the UI protocol depends on it."""

    topic: str = field(init=False, default="event")
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        return {"topic": self.topic, "ts": self.ts.isoformat(), **_payload(self)}


def _payload(ev: Event) -> dict[str, Any]:
    """Serialize dataclass fields minus the bookkeeping ones."""
    out: dict[str, Any] = {}
    for slot in getattr(type(ev), "__dataclass_fields__", {}):
        if slot in {"topic", "ts"}:
            continue
        value = getattr(ev, slot)
        out[slot] = value.isoformat() if isinstance(value, datetime) else value
    return out


# ── interaction / UI ─────────────────────────────────────────────────
@dataclass(slots=True)
class StateChanged(Event):
    topic: str = field(init=False, default="state.changed")
    state: str = ""
    previous: str = ""


@dataclass(slots=True)
class WakeDetected(Event):
    topic: str = field(init=False, default="wake.detected")
    keyword: str = ""
    score: float = 0.0


@dataclass(slots=True)
class SpeakerMatched(Event):
    topic: str = field(init=False, default="speaker.matched")
    name: str = ""
    score: float = 0.0
    owner: bool = False


@dataclass(slots=True)
class TokenDelta(Event):
    topic: str = field(init=False, default="llm.token")
    text: str = ""


@dataclass(slots=True)
class ReplyFinished(Event):
    topic: str = field(init=False, default="llm.done")
    text: str = ""
    language: str = ""
    provider: str = ""


@dataclass(slots=True)
class MoodChanged(Event):
    topic: str = field(init=False, default="mood.changed")
    mood: str = ""
    energy: float = 0.0
    warmth: float = 0.0
    humor_allowed: bool = True


@dataclass(slots=True)
class AudioLevel(Event):
    topic: str = field(init=False, default="audio.level")
    level: float = 0.0


@dataclass(slots=True)
class ToolStarted(Event):
    topic: str = field(init=False, default="tool.started")
    skill: str = ""
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolFinished(Event):
    topic: str = field(init=False, default="tool.finished")
    skill: str = ""
    ok: bool = True
    spoken: str = ""


@dataclass(slots=True)
class ConfirmationRequested(Event):
    topic: str = field(init=False, default="confirm.requested")
    question: str = ""
    timeout_s: float = 10.0


@dataclass(slots=True)
class DegradedModeChanged(Event):
    topic: str = field(init=False, default="system.degraded")
    degraded: bool = False
    reason: str = ""
    mode: str = "lean"


@dataclass(slots=True)
class TimingRecorded(Event):
    topic: str = field(init=False, default="timing.recorded")
    stage: str = ""
    ms: float = 0.0
    provider: str = ""


@dataclass(slots=True)
class ErrorRaised(Event):
    topic: str = field(init=False, default="system.error")
    where: str = ""
    message: str = ""
    fatal: bool = False


Handler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """Small async pub/sub.

    * Subscribers are isolated: one raising subscriber cannot break a publisher.
    * ``slow_subscriber`` warnings surface UI/logger backpressure instead of
      letting it stall a conversation (architecture doc, L5 requirements).
    """

    def __init__(self, *, queue_size: int = 256) -> None:
        self._queues: dict[str, list[asyncio.Queue[Event]]] = {}
        self._handlers: dict[str, list[Handler]] = {}
        self._queue_size = queue_size
        self._published = 0
        self._dropped = 0

    # ── subscribing ──────────────────────────────────────────────────
    def subscribe(self, *topics: str, queue_size: int | None = None) -> asyncio.Queue[Event]:
        """Subscribe by pulling from the returned queue (used by the UI bridge)."""
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_size or self._queue_size)
        for topic in topics:
            self._queues.setdefault(topic, []).append(queue)
        return queue

    def on(self, topic: str, handler: Handler) -> None:
        """Subscribe with a callback (used by loggers, audit, timings)."""
        self._handlers.setdefault(topic, []).append(handler)

    def matches(self, event: Event) -> list[asyncio.Queue[Event]]:
        return [*self._queues.get(event.topic, []), *self._queues.get("*", [])]

    # ── publishing ───────────────────────────────────────────────────
    async def publish(self, event: Event) -> None:
        self._published += 1
        for queue in self.matches(event):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._dropped += 1
                log.warning("slow_subscriber topic=%s dropped=%d", event.topic, self._dropped)
                with contextlib.suppress(asyncio.QueueEmpty):  # drop oldest, keep newest
                    queue.get_nowait()
                    queue.put_nowait(event)
        for handler in self._handlers.get(event.topic, []) + self._handlers.get("*", []):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("event_handler_failed topic=%s", event.topic)

    def emit(self, event: Event) -> None:
        """Fire-and-forget publish for non-async call sites."""
        try:
            asyncio.get_running_loop().create_task(self.publish(event))
        except RuntimeError:  # no loop (CLI/tests) → run synchronously
            asyncio.run(self.publish(event))

    # ── introspection ────────────────────────────────────────────────
    @property
    def stats(self) -> dict[str, int]:
        return {"published": self._published, "dropped": self._dropped}


__all__ = [
    "AudioLevel",
    "ConfirmationRequested",
    "DegradedModeChanged",
    "ErrorRaised",
    "Event",
    "EventBus",
    "Handler",
    "MoodChanged",
    "ReplyFinished",
    "SpeakerMatched",
    "StateChanged",
    "TimingRecorded",
    "TokenDelta",
    "ToolFinished",
    "ToolStarted",
    "WakeDetected",
]
