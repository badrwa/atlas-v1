"""Mood — a small, honest state that decays.

The orb (L5) needs a colour, and the persona needs to know whether a joke is
welcome.  L8 grows this into a real mood model; what exists here is deliberately
minimal and *predictable*: a mood is a label with an intensity, it moves towards
calm over turns, and a strong signal moves it faster.

Where the mood comes from, in priority order:

1. The model's own read (`emotion` in a structured turn) — best signal.
2. A local heuristic over the reply text — used on the fast streaming path,
   where waiting for JSON would cost the time-to-first-token that makes Atlas
   feel alive.

The heuristic is intentionally crude and documented as such: it is not
"sentiment analysis", it is a handful of markers that move a dial a little.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas_mind.envelope import ReplyEnvelope

# Emotion → (mood label, how far it pushes the dial, warmth, humour allowed)
EMOTION_EFFECT: dict[str, tuple[str, float, float, bool]] = {
    "calm": ("calm", 0.1, 0.60, True),
    "happy": ("happy", 0.9, 0.80, True),
    "amused": ("happy", 0.8, 0.75, True),
    "playful": ("happy", 0.7, 0.70, True),
    "focused": ("focused", 0.5, 0.55, True),
    "serious": ("serious", 0.7, 0.45, False),
    "tired": ("tired", 0.6, 0.55, False),
    "frustrated": ("frustrated", 0.8, 0.40, False),
}

# Local markers for the fast path.  Keyed by mood, matched on normalised text.
MARKERS: dict[str, tuple[str, ...]] = {
    "happy": ("haha", "mabrouk", "zwin", "mzyan bezzaf", "great", "nice one", "brilliant"),
    "amused": ("as usual", "typical", "of course", "zwina", "hhh"),
    "serious": ("careful", "important", "attention", "warning", "hsab", "khtar"),
    "tired": ("3yit", "tired", "late", "sleep", "n3as", "fatigué"),
    "frustrated": ("ma9aditch", "still failing", "again and again", "machi khdam"),
}

# The owner's own words matter more than Atlas's tone.
OWNER_SIGNALS: dict[str, tuple[str, ...]] = {
    "frustrated": ("machi khdam", "ma9aditch", "not working", "broken", "z3af", "error again"),
    "tired": ("3yit", "tired", "sleepy", "n3ass", "long day"),
    "serious": ("urgent", "mohim", "important", "problem", "mochkil"),
}


@dataclass(slots=True)
class MoodView:
    """What the persona and the orb need to know (L8 deepens this)."""

    label: str = "calm"
    energy: float = 0.5
    warmth: float = 0.6
    humor_allowed: bool = True
    intensity: float = 0.5

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "energy": round(self.energy, 2),
            "warmth": round(self.warmth, 2),
            "humor_allowed": self.humor_allowed,
            "intensity": round(self.intensity, 2),
        }


class MoodEngine:
    """Tracks one mood across turns; decays towards calm."""

    def __init__(self, *, decay: float = 0.25, label: str = "calm") -> None:
        self.decay = decay
        self.view = MoodView(label=label)
        self.history: list[str] = []

    # ── inputs ───────────────────────────────────────────────────────
    def observe_owner(self, text: str) -> None:
        """The owner's own mood wins: if they are frustrated, Atlas is not playful."""
        lowered = text.lower()
        for mood, markers in OWNER_SIGNALS.items():
            if any(marker in lowered for marker in markers):
                label, push, warmth, humour = EMOTION_EFFECT[mood]
                self._apply(label, push * 1.2, warmth, humour)
                return

    def observe_reply(self, text: str) -> MoodView:
        """Fast path: guess from the words Atlas just used."""
        lowered = text.lower()
        best: tuple[str, int] | None = None
        for mood, markers in MARKERS.items():
            hits = sum(1 for marker in markers if marker in lowered)
            if hits and (best is None or hits > best[1]):
                best = (mood, hits)
        if best is None:
            self._decay()
            return self.view
        label, hits = best
        _, push, warmth, humour = EMOTION_EFFECT[label]
        self._apply(label, push * min(1.0, 0.5 + 0.25 * hits), warmth, humour)
        return self.view

    def observe_envelope(self, envelope: ReplyEnvelope) -> MoodView:
        """Structured path: the model said how it sounds; that beats a keyword."""
        label, push, warmth, humour = EMOTION_EFFECT.get(
            envelope.emotion, EMOTION_EFFECT["calm"]
        )
        if envelope.mood_delta:
            boost = {"warm": 0.2, "playful": 0.3, "focused": 0.2}.get(envelope.mood_delta, 0.1)
            push = min(1.0, push + boost)
        self._apply(label, push, warmth, humour)
        return self.view

    # ── mechanics ────────────────────────────────────────────────────
    def _apply(self, label: str, push: float, warmth: float, humour: bool) -> None:
        self.view.label = label
        self.view.intensity = min(1.0, push)
        self.view.energy = round(min(1.0, 0.4 + push * 0.5), 2)
        self.view.warmth = round(warmth, 2)
        self.view.humor_allowed = humour
        self.history.append(label)

    def _decay(self) -> None:
        """Nothing notable happened: drift back towards calm, never snap."""
        self.view.intensity = max(0.0, self.view.intensity - self.decay)
        if self.view.intensity <= 0.35:
            self.view.label = "calm"
            self.view.energy = 0.5
            self.view.warmth = 0.6
            self.view.humor_allowed = True

    def reset(self) -> None:
        self.view = MoodView()
        self.history.clear()


__all__ = ["EMOTION_EFFECT", "MARKERS", "MoodEngine", "MoodView"]
