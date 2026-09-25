"""Prosody — *how* a sentence is said, decided before it is synthesised.

The plan's table (LEVEL-03 step 7) asks for rate, pitch and energy per mood.  On
this hardware one of those three is a lie:

* **rate** is real.  Piper takes `length_scale` (the inverse), the Darija sidecar
  takes a rate, and both change how the sentence actually sounds.
* **energy** is real, and it is not the model's job: it is a gain multiplier
  applied by the player (`AudioPlayer`), where it also feeds the orb's amplitude.
* **pitch** is *not* real.  Piper's ONNX voices and OuteTTS-style vocoders have
  no pitch control — nudging `noise_scale` changes *variation*, not pitch.  So
  this module carries `expressiveness` instead and documents what it really does:
  a small amount of vocal variability, which is what "sounds alive" means in
  practice.  Faking pitch by resampling would change speed and formants together
  and make Atlas sound like a bad tape — the plan's own pitfall list says use
  rate + energy + wording, and that is what this does.

Mood arrives as a `MoodView` from L1, but this module never imports `atlas-mind`:
it reads `label` and `intensity` off whatever object it is handed (duck-typed),
so the audio package stays independent of the brain — and every test here runs
without a provider key.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, time
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Prosody:
    """Three numbers a voice can honour, plus the label they came from."""

    rate: float = 1.0
    energy: float = 1.0
    expressiveness: float = 1.0
    label: str = "calm"

    def for_piper(self) -> dict[str, float]:
        """Piper: `length_scale` is seconds-per-phoneme, so it is the inverse."""
        return {
            "length_scale": round(1.0 / self.rate, 4),
            "noise_scale": round(0.667 * self.expressiveness, 4),
            "noise_w": round(0.8 * self.expressiveness, 4),
        }

    def for_sidecar(self) -> dict[str, float]:
        """The Darija sidecar gets a rate only — accept it and say so."""
        return {"rate": self.rate, "gain": self.energy}

    def blend(self, other: Prosody, weight: float) -> Prosody:
        """Move this plan `weight` (0..1) of the way towards `other`.

        A mild mood should be a mild change: `intensity` from the mood engine
        scales how far from calm a sentence actually drifts.
        """
        weight = max(0.0, min(1.0, weight))
        if weight == 0.0:
            return self
        lerp = lambda a, b: round(a + (b - a) * weight, 4)  # noqa: E731 - one use
        return replace(
            self,
            rate=lerp(self.rate, other.rate),
            energy=lerp(self.energy, other.energy),
            expressiveness=lerp(self.expressiveness, other.expressiveness),
            label=other.label if weight >= 0.5 else self.label,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "rate": self.rate,
            "energy": self.energy,
            "expressiveness": self.expressiveness,
            "label": self.label,
        }


CALM = Prosody()

#: The plan's table, as data: mood (or sentence role) → prosody.
ROLE_TABLE: dict[str, Prosody] = {
    "calm": Prosody(1.00, 1.00, 1.00, "calm"),  # calm answer
    "amused": Prosody(1.05, 1.10, 1.20, "amused"),  # joke: a touch faster, louder
    "serious": Prosody(0.92, 0.90, 0.80, "serious"),  # bad news: slower, flatter
    "quiet": Prosody(0.95, 0.75, 0.90, "quiet"),  # late night
    "confirm": Prosody(0.95, 1.00, 1.00, "confirm"),  # "bghiti nkemmel?"
}

#: Mood label → role.  The one place that translation lives.
MOOD_ROLE: dict[str, str] = {
    "happy": "amused",
    "amused": "amused",
    "playful": "amused",
    "serious": "serious",
    "frustrated": "serious",
    "tired": "quiet",
    "focused": "calm",
    "calm": "calm",
}

#: Questions get the confirmation contour unless the mood says otherwise.
QUESTION_MARKS = tuple("?؟")


def in_quiet_hours(now: time, start: str = "22:30", end: str = "07:00") -> bool:
    """Is `now` inside the quiet window?  Handles the midnight wrap."""
    try:
        low, high = time.fromisoformat(start), time.fromisoformat(end)
    except ValueError:
        log.warning("quiet_hours_invalid start=%r end=%r", start, end)
        return False
    if low <= high:
        return low <= now < high
    return now >= low or now < high  # window crosses midnight


class ProsodyDirector:
    """Turns (mood, sentence, clock) into one `Prosody`.

    Deliberately synchronous and dependency-free: it runs once per sentence, in
    the middle of the speech pipeline, and must never be the reason audio is
    late.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        quiet_hours: tuple[str, str] = ("22:30", "07:00"),
        clock: Any = None,
    ) -> None:
        self.enabled = enabled
        self.quiet_hours = quiet_hours
        self.clock = clock or datetime.now

    def role_for(self, text: str = "", *, mood: Any = None) -> str:
        """Sentence role, mood label, quiet hours — in that order of authority.

        A `?` wins over a happy mood (a question is asked like a question), but
        a serious mood wins over a `?`: "are you sure?" about a deleted file is
        not a chirp.
        """
        label = str(getattr(mood, "label", "") or "").lower()
        role = MOOD_ROLE.get(label, "calm")
        if role == "serious":
            return role
        if text.rstrip().endswith(QUESTION_MARKS):
            return "confirm"
        return role

    def plan(self, text: str = "", *, mood: Any = None, now: datetime | None = None) -> Prosody:
        """The prosody for one sentence."""
        if not self.enabled:
            return CALM

        role = self.role_for(text, mood=mood)
        base = ROLE_TABLE.get(role, CALM)

        # A quiet-hours sentence is softer even when the mood is cheerful — the
        # person is probably in bed, not at a party.
        moment = now or self.clock()
        quiet = in_quiet_hours(moment.time(), *self.quiet_hours)
        if quiet and role not in {"serious"}:
            base = base.blend(ROLE_TABLE["quiet"], 1.0 if role == "quiet" else 0.7)

        intensity = float(getattr(mood, "intensity", 0.0) or 0.0)
        if intensity <= 0 or role == "calm":
            return base
        # `intensity` is how sure the mood engine is; 0.5 is its neutral value.
        return CALM.blend(base, min(1.0, 0.4 + intensity))

    def for_question(self, text: str, *, mood: Any = None) -> Prosody:
        """The continuation prompt ("bghiti nkemmel?") — always a question."""
        if not self.enabled:
            return CALM
        return self.plan("?", mood=mood)


__all__ = [
    "CALM",
    "MOOD_ROLE",
    "ROLE_TABLE",
    "Prosody",
    "ProsodyDirector",
    "in_quiet_hours",
]
