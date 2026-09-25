"""The interaction FSM — one state machine drives the mic, the UI and the latency budget.

Diagram: architecture doc D6.  This module is intentionally pure: states in,
states out, no I/O, no sleeping, no audio.  Timeouts are *fed in* as events by
the caller so the machine stays testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from atlas_core.errors import AtlasError


class State(StrEnum):
    DORMANT = "dormant"
    WAKING = "waking"
    LISTENING = "listening"
    THINKING = "thinking"
    CONFIRMING = "confirming"
    SPEAKING = "speaking"
    ERROR = "error"


class Signal(StrEnum):
    WAKE = "wake"
    WAKE_TIMEOUT = "wake_timeout"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"
    SESSION_TIMEOUT = "session_timeout"
    FIRST_SENTENCE_READY = "first_sentence_ready"
    NEEDS_CONFIRMATION = "needs_confirmation"
    CONFIRMED = "confirmed"
    DENIED = "denied"
    CONFIRM_TIMEOUT = "confirm_timeout"
    DONE_SPEAKING = "done_speaking"
    STOP_REQUESTED = "stop_requested"
    PROVIDER_FAILED = "provider_failed"
    RECOVERED = "recovered"


# (state, signal) -> next state.  Missing pairs raise IllegalTransition.
TRANSITIONS: dict[tuple[State, Signal], State] = {
    (State.DORMANT, Signal.WAKE): State.WAKING,
    (State.WAKING, Signal.SPEECH_STARTED): State.LISTENING,
    (State.WAKING, Signal.WAKE_TIMEOUT): State.DORMANT,
    (State.WAKING, Signal.WAKE): State.WAKING,  # double "atlas" → stay awake
    (State.LISTENING, Signal.SPEECH_ENDED): State.THINKING,
    (State.LISTENING, Signal.SESSION_TIMEOUT): State.DORMANT,
    (State.LISTENING, Signal.STOP_REQUESTED): State.DORMANT,
    (State.THINKING, Signal.FIRST_SENTENCE_READY): State.SPEAKING,
    (State.THINKING, Signal.NEEDS_CONFIRMATION): State.CONFIRMING,
    (State.THINKING, Signal.PROVIDER_FAILED): State.ERROR,
    (State.THINKING, Signal.STOP_REQUESTED): State.DORMANT,
    (State.CONFIRMING, Signal.CONFIRMED): State.THINKING,
    (State.CONFIRMING, Signal.DENIED): State.SPEAKING,
    (State.CONFIRMING, Signal.CONFIRM_TIMEOUT): State.SPEAKING,  # timeout == no
    (State.SPEAKING, Signal.DONE_SPEAKING): State.DORMANT,
    (State.SPEAKING, Signal.STOP_REQUESTED): State.DORMANT,
    (State.SPEAKING, Signal.SPEECH_STARTED): State.LISTENING,  # follow-up window
    (State.ERROR, Signal.RECOVERED): State.SPEAKING,
    (State.ERROR, Signal.STOP_REQUESTED): State.DORMANT,
}

# States in which the microphone must be muted for output (half-duplex).
MUTED_STATES = frozenset({State.SPEAKING})

# States that prove Atlas is actively listening (used by the tray/privacy UI).
HOT_STATES = frozenset({State.WAKING, State.LISTENING, State.THINKING, State.CONFIRMING})


class IllegalTransition(AtlasError):
    """A signal arrived that has no meaning in the current state."""


@dataclass
class InteractionFSM:
    """Table-driven, side-effect free, and fully testable.

    Hooks (`on_enter`/`on_exit`) are how the outside world reacts: publish a
    `StateChanged` event, mute the mic, start a timeout timer…
    """

    state: State = State.DORMANT
    history: list[tuple[State, Signal, State]] = field(default_factory=list)
    _on_enter: dict[State, list[Callable[[State, Signal, State], None]]] = field(default_factory=dict)
    _on_exit: dict[State, list[Callable[[State, Signal, State], None]]] = field(default_factory=dict)

    # ── hooks ────────────────────────────────────────────────────────
    def on_enter(self, state: State, callback: Callable[[State, Signal, State], None]) -> None:
        self._on_enter.setdefault(state, []).append(callback)

    def on_exit(self, state: State, callback: Callable[[State, Signal, State], None]) -> None:
        self._on_exit.setdefault(state, []).append(callback)

    # ── core ─────────────────────────────────────────────────────────
    def can(self, signal: Signal) -> bool:
        return (self.state, signal) in TRANSITIONS

    def send(self, signal: Signal) -> State:
        key = (self.state, signal)
        if key not in TRANSITIONS:
            raise IllegalTransition(f"{signal} is illegal in {self.state}")
        previous, self.state = self.state, TRANSITIONS[key]
        for callback in self._on_exit.get(previous, []):
            callback(previous, signal, self.state)
        for callback in self._on_enter.get(self.state, []):
            callback(previous, signal, self.state)
        self.history.append((previous, signal, self.state))
        return self.state

    def try_send(self, signal: Signal) -> State | None:
        """Send if legal, otherwise ignore — for noisy real-world inputs."""
        if not self.can(signal):
            return None
        return self.send(signal)

    # ── derived ──────────────────────────────────────────────────────
    @property
    def mic_should_be_muted(self) -> bool:
        return self.state in MUTED_STATES

    @property
    def is_hot(self) -> bool:
        return self.state in HOT_STATES

    def reset(self) -> None:
        self.state = State.DORMANT
        self.history.clear()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "mic_muted": self.mic_should_be_muted,
            "hot": self.is_hot,
            "transitions": len(self.history),
        }


__all__ = [
    "HOT_STATES",
    "MUTED_STATES",
    "TRANSITIONS",
    "IllegalTransition",
    "InteractionFSM",
    "Signal",
    "State",
]
