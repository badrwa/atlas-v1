"""The event → orb pipeline, with the two throttles that keep the core fast.

`UiHub` is pure Python: it takes `Event`s from the `EventBus` and produces
`UiMessage`s, and it owns the three policies LEVEL-05 asks for by name.

* **level is throttled to 30 Hz.**  The playback meter fires ~30 times a second;
  a slow socket must not turn that into backpressure on the audio thread.
* **tokens are coalesced every 50 ms.**  One caption per token would be a message
  per token; one per 50 ms is a caption that looks live.
* **a stalled socket drops, never blocks.**  Each session has a bounded queue;
  when it is full the *oldest* message goes.  A UI that cannot keep up loses
  pixels, never a conversation.

A hub with no sessions still does the work (cheap, and it keeps `snapshot()`
current), so `atlas ui --no-window` and the tests exercise the same path a real
orb does.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from atlas_core.events import Event
from atlas_ui.protocol import (
    HelloMessage,
    UiMessage,
)
from atlas_ui.theme import CAPTION_FADE_S, ThemeEngine

#: LEVEL-05 step 2: "throttles LevelMsg to 30 Hz and coalesces token deltas
#: every ~50 ms".  Named constants, because a test asserts exactly these numbers.
LEVEL_HZ = 30.0
TOKEN_BATCH_S = 0.05
#: A session that is behind keeps at most this many messages.
QUEUE_SIZE = 64
#: Clocks are floats: a source that fires exactly at the interval must not be
#: throttled by a rounding error (`0.0333 - 0.0 < 0.0333` is true in binary).
TIME_EPSILON = 1e-6


log = logging.getLogger(__name__)


@dataclass(slots=True)
class HubStats:
    received: int = 0
    emitted: int = 0
    throttled: int = 0
    coalesced: int = 0
    dropped: int = 0
    by_topic: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "emitted": self.emitted,
            "throttled": self.throttled,
            "coalesced": self.coalesced,
            "dropped": self.dropped,
            "by_topic": dict(self.by_topic),
        }


class UiSession:
    """One websocket's outbox: bounded, lossy when full, never blocking."""

    def __init__(self, name: str = "", *, queue_size: int = QUEUE_SIZE, hub: UiHub | None = None) -> None:
        self.name = name
        self.queue: deque[UiMessage] = deque(maxlen=queue_size)
        self.connected = True
        self.dropped = 0
        self.hub = hub

    def send(self, message: UiMessage) -> bool:
        """Queue a message; False means the queue was full and the oldest went."""
        if not self.connected:
            return False
        full = len(self.queue) == self.queue.maxlen
        self.queue.append(message)
        if full:
            self.dropped += 1
            if self.hub is not None:
                self.hub.stats.dropped += 1
        return not full

    def drain(self) -> list[UiMessage]:
        out = list(self.queue)
        self.queue.clear()
        return out

    def close(self) -> None:
        self.connected = False
        if self.hub is not None:
            self.hub.sessions = [session for session in self.hub.sessions if session is not self]


class UiHub:
    """`Event` in, `UiMessage` out — the UI's whole brain, without a server."""

    def __init__(
        self,
        *,
        theme: ThemeEngine | None = None,
        level_hz: float = LEVEL_HZ,
        token_batch_s: float = TOKEN_BATCH_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.theme = theme or ThemeEngine()
        self.level_hz = float(level_hz)
        self.token_batch_s = float(token_batch_s)
        self._clock = clock
        self.sessions: list[UiSession] = []
        self.stats = HubStats()
        # presentation state, so a late client can be caught up in one message
        self.state = "dormant"
        self.previous_state = ""
        self.mood = self.theme.mood("calm")
        self.level = 0.0
        self.caption = ""
        self.caption_role = "atlas"
        self.caption_at = 0.0
        self.degraded = False
        # throttle bookkeeping
        self._last_level_at = -1e9
        self._pending_level: float | None = None
        self._pending_levels = 0
        #: The whole reply so far.  A caption is the *growing* sentence, not the
        #: batch: sending only the new deltas would make the card flicker through
        #: fragments ("lam, kifash") instead of reading like a sentence.
        self._reply_text = ""
        self._token_since = -1e9
        self._pending_caption = False

    # ── sessions ─────────────────────────────────────────────────────
    def connect(self, name: str = "") -> UiSession:
        session = UiSession(name, hub=self)
        self.sessions.append(session)
        self.stats.by_topic.setdefault("session", 0)
        return session

    def session_count(self) -> int:
        return sum(1 for session in self.sessions if session.connected)

    def _broadcast(self, messages: list[UiMessage]) -> None:
        if not messages:
            return
        self.stats.emitted += len(messages)
        for session in list(self.sessions):
            for message in messages:
                session.send(message)

    # ── the pipeline ─────────────────────────────────────────────────
    def handle(self, event: Event) -> list[UiMessage]:
        """One event → zero or more messages (throttling happens here)."""
        topic = event.topic
        self.stats.received += 1
        self.stats.by_topic[topic] = self.stats.by_topic.get(topic, 0) + 1

        messages = self._convert(topic, event)
        if messages:
            self._broadcast(messages)
        return messages

    def _policies(self) -> dict[str, Any]:
        """One place where a topic becomes a message policy.

        The keys must match `protocol.TOPIC_MESSAGES` exactly (tested): protocol
        says *what* travels, this says *how* — and a topic in one table but not
        the other is a message the generated TypeScript cannot describe, or an
        event the orb silently drops.
        """
        return {
            "state.changed": self._on_state,
            "mood.changed": self._on_mood,
            "audio.level": self._on_level,
            "llm.token": self._on_token,
            "llm.done": self._on_reply,
            "tool.started": self._on_tool,
            "tool.finished": self._on_tool,
            "confirm.requested": self._on_confirm,
            "system.degraded": self._on_degraded,
            "system.error": self._on_error,
            "speaker.matched": self._on_speaker,
        }

    def policy_topics(self) -> frozenset[str]:
        return frozenset(self._policies())

    def _convert(self, topic: str, event: Event) -> list[UiMessage]:
        handler = self._policies().get(topic)
        if handler is None:
            return []
        return handler(event)

    # ── per-topic policies ───────────────────────────────────────────
    def _on_state(self, event: Event) -> list[UiMessage]:
        state = str(getattr(event, "state", "") or "")
        if not state or state == self.state:
            return []  # the orb already draws this; do not wake it for nothing
        self.previous_state, self.state = self.state, state
        return [self.theme.state(state, previous=self.previous_state)]

    def _on_mood(self, event: Event) -> list[UiMessage]:
        message = self.theme.mood(
            str(getattr(event, "mood", "") or "calm"),
            energy=float(getattr(event, "energy", 0.5) or 0.5),
            warmth=float(getattr(event, "warmth", 0.6) or 0.6),
            humor_allowed=bool(getattr(event, "humor_allowed", True)),
        )
        if message == self.mood:
            return []
        self.mood = message
        return [message]

    def _on_level(self, event: Event) -> list[UiMessage]:
        # The newest value always replaces the pending one: a meter is not a log,
        # and a late frame describing the past is worse than no frame at all.
        self._pending_level = float(getattr(event, "level", 0.0) or 0.0)
        self._pending_levels += 1
        interval = 1.0 / self.level_hz if self.level_hz > 0 else 0.0
        if self._clock() - self._last_level_at < interval - TIME_EPSILON:
            self.stats.throttled += 1
            return []
        return self._flush_level()

    def _flush_level(self) -> list[UiMessage]:
        if self._pending_level is None:
            return []
        value = self._pending_level
        if self._pending_levels > 1:
            self.stats.coalesced += 1
        self._pending_level = None
        self._pending_levels = 0
        self._last_level_at = self._clock()
        self.level = value
        return [self.theme.level(value)]

    def _on_token(self, event: Event) -> list[UiMessage]:
        text = str(getattr(event, "text", "") or "")
        if not text:
            return []
        now = self._clock()
        first = not self._reply_text
        self._reply_text += text
        if first:
            self._token_since = now
            # The first delta of a reply is shown at once: the orb looks alive
            # from the first word, and the batching below keeps the rest cheap.
            return self._flush_tokens(final=False)
        if now - self._token_since < self.token_batch_s - TIME_EPSILON:
            self._pending_caption = True
            self.stats.throttled += 1
            return []
        return self._flush_tokens(final=False)

    def _on_reply(self, event: Event) -> list[UiMessage]:
        text = str(getattr(event, "text", "") or "")
        language = str(getattr(event, "language", "") or "")
        if text:
            self._reply_text = text
        return self._flush_tokens(final=True, language=language)

    def _flush_tokens(self, *, final: bool, language: str = "") -> list[UiMessage]:
        if not self._reply_text:
            return []
        text = self._reply_text
        if final:
            self._reply_text = ""
        self._pending_caption = False
        self._token_since = self._clock()
        caption = self.theme.caption(
            text, role="atlas", final=final, language=language or self.theme.language
        )
        self.caption = caption.text
        self.caption_role = caption.role
        self.caption_at = self._clock()
        return [caption]

    def _on_tool(self, event: Event) -> list[UiMessage]:
        phase = "finished" if event.topic == "tool.finished" else "started"
        return [
            self.theme.tool(
                str(getattr(event, "skill", "") or ""),
                phase=phase,
                ok=bool(getattr(event, "ok", True)),
            )
        ]

    def _on_confirm(self, event: Event) -> list[UiMessage]:
        return [
            self.theme.confirm(
                str(getattr(event, "question", "") or ""),
                timeout_s=float(getattr(event, "timeout_s", 10.0) or 10.0),
            )
        ]

    def _on_error(self, event: Event) -> list[UiMessage]:
        """An error the user must not have to read a log file to notice.

        Transient errors flash the orb red and the next state change puts it back;
        a *fatal* one leaves the degraded chip on, because "Atlas is half broken"
        has to stay true for as long as it is true.  The message text stays out of
        the caption: it is for the console and the log, not for a 420 px card.
        """
        where = str(getattr(event, "where", "") or "")
        message = str(getattr(event, "message", "") or "")
        fatal = bool(getattr(event, "fatal", False))
        log.warning("ui_error where=%s fatal=%s message=%s", where, fatal, message)
        if fatal:
            self.degraded = True
            return [self.theme.degraded(degraded=True, reason=where or "error", mode="error")]
        return [self.theme.state("error", previous=self.state)]

    def _on_degraded(self, event: Event) -> list[UiMessage]:
        degraded = bool(getattr(event, "degraded", False))
        self.degraded = degraded
        return [
            self.theme.degraded(
                degraded=degraded,
                reason=str(getattr(event, "reason", "") or ""),
                mode=str(getattr(event, "mode", "") or "lean"),
            )
        ]

    def _on_speaker(self, event: Event) -> list[UiMessage]:
        """A matched voice is a caption, not a new widget: the orb has no room.

        The orb shows *that* it knows who is talking (a small name chip with the
        confidence), because a private answer to the wrong person is the failure
        this level exists to make visible.
        """
        name = str(getattr(event, "name", "") or "")
        if not name:
            return []
        score = float(getattr(event, "score", 0.0) or 0.0)
        owner = bool(getattr(event, "owner", False))
        label = name if owner else f"{name} · {score:.2f}"
        caption = self.theme.caption(label, role="atlas", final=False)
        return [caption]

    # ── catching a client up ─────────────────────────────────────────
    def hello(self, *, version: str = "", powers: dict[str, bool] | None = None) -> HelloMessage:
        """The first message every session gets: what this orb is looking at."""
        return HelloMessage(
            version=version,
            states=list(self._state_names()),
            powers=powers or self._powers(),
        )

    def _state_names(self) -> tuple[str, ...]:
        from atlas_ui.theme import ORB_STATES

        return ORB_STATES

    def _powers(self) -> dict[str, bool]:
        """The three glyphs the orb draws: mic on/off, cloud in use, vault open."""
        return {
            "mic": not self.theme.microphone_muted and self.state in {"waking", "listening"},
            "cloud": bool(self.stats.by_topic.get("llm.token", 0)),
            "vault": False,
        }

    def snapshot(self) -> list[UiMessage]:
        """Everything a client that just connected needs to draw the current truth."""
        messages: list[UiMessage] = [self.hello()]
        messages.append(self.theme.state(self.state, previous=self.previous_state))
        messages.append(self.mood)
        if self.caption:
            messages.append(self.theme.caption(self.caption, role=self.caption_role, final=True))
        if self.degraded:
            messages.append(self.theme.degraded(degraded=True, reason="already degraded"))
        return messages

    def caption_is_stale(self) -> bool:
        """True once the caption has been on screen longer than the fade timer."""
        return bool(self.caption) and self._clock() - self.caption_at > CAPTION_FADE_S

    def tick(self) -> list[UiMessage]:
        """Called a few times a second: flushes batching and clears stale captions.

        Without this, a reply whose last token arrived inside a batch would wait
        for the *next* event to be shown — the orb would look asleep at the end
        of a sentence, which is exactly when the user is looking at it.
        """
        messages: list[UiMessage] = []
        if (
            self._pending_caption
            and self._clock() - self._token_since >= self.token_batch_s - TIME_EPSILON
        ):
            messages.extend(self._flush_tokens(final=False))
        if self.caption_is_stale():
            self.caption = ""
            messages.append(self.theme.caption("", role="atlas", final=True))
        if self._pending_level is not None and self._clock() - self._last_level_at >= (
            1.0 / self.level_hz if self.level_hz > 0 else 0.0
        ) - TIME_EPSILON:
            messages.extend(self._flush_level())
        if messages:
            self._broadcast(messages)
        return messages


__all__ = ["LEVEL_HZ", "QUEUE_SIZE", "TOKEN_BATCH_S", "HubStats", "UiHub", "UiSession"]
