"""What the orb is told, and how it is told: state → motion, mood → colour.

`atlas-ui`'s browser side draws; this module decides *what* it draws, so the
meanings live in Python where they can be tested.  Two rules from the L5 design
language are enforced here rather than in JavaScript:

* **state is motion, mood is hue** — a message carries one and not the other, so
  the orb can never confuse "thinking" with "sad".
* **captions are text the browser only has to lay out** — clamped to two lines,
  direction-tagged for Arabic, and never truncated mid-word.

Nothing here imports a web framework: this is the layer that must keep working
when WebView2 is missing and the orb runs headless.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from atlas_core.fsm import MUTED_STATES, State
from atlas_ui.protocol import (
    MOOD_HUES,
    CaptionMessage,
    ConfirmMessage,
    DegradedMessage,
    LevelMessage,
    MoodMessage,
    StateMessage,
    ToolMessage,
)

#: The order the orb's own state machine walks (LEVEL-05 §design language).
#: `muted` and `degraded` are not FSM states — they are overlays the orb draws on
#: top of whatever motion is current, which is exactly how the plan describes them.
#: The loop's own vocabulary (atlas_audio.LoopState) is the source of truth; this
#: table is where those names become motion, plus the three states that only exist
#: with a face ("dormant", "confirming", "error").  `idle` and `stopped` both read
#: to a human as "Atlas is off" — they breathe rather than inventing a second and
#: third kind of doing nothing — and `followup` is the short "still here" pulse.
STATE_MOTIONS: dict[str, str] = {
    "idle": "breathing",
    "dormant": "breathing",
    "stopped": "breathing",
    "waking": "pulse",
    "followup": "pulse",
    "listening": "spin",
    "thinking": "orbit",
    "confirming": "hold",
    "speaking": "blob",
    "error": "flash",
    "muted": "frozen",
    "degraded": "dim",
}

#: Overlays: they do not replace a motion, they modify it.
OVERLAY_STATES = frozenset({"muted", "degraded"})

#: The emotional part of a reply, from `atlas_mind.mood` labels.  Every label the
#: mood engine can produce must appear here, or the hue falls back to calm (and a
#: test asserts that this table covers the mood engine's vocabulary).
MOOD_HUE_FALLBACK = 174

#: Skills that have a human phrase, Darija first.  Unknown skills are shown by
#: name — honesty beats decoration, and a wrong word is worse than a raw one.
TOOL_PHRASES: dict[str, tuple[str, str]] = {
    "system_stats": ("kanchouf l'PC", "checking the PC"),
    "open_url": ("kanhell link", "opening a link"),
    "remember": ("kankteb f notes", "writing to your notes"),
    "shutdown_pc": ("kantsedd PC", "shutting the PC down"),
    "search": ("kanqelleb", "searching"),
    "weather": ("kanchouf ljaw", "checking the weather"),
}

#: How long a caption may stay on screen (the orb's own timer uses this).
CAPTION_FADE_S = 4.0
#: Two lines, per the design language.  42 characters each is what fits at 420 px.
CAPTION_LINE_CHARS = 42
CAPTION_LINES = 2


def is_rtl(text: str) -> bool:
    """True when the first strong character is right-to-left.

    Counting Arabic letters and comparing with Latin ones gets `"salam صاحبي"`
    wrong; Unicode's bidi classes get it right, and cost one import.
    """
    for char in text:
        if char.isspace() or not char.isalpha():
            continue
        return unicodedata.bidirectional(char) in {"R", "AL", "AN"}
    return False


def clamp_caption(text: str, *, lines: int = CAPTION_LINES, width: int = CAPTION_LINE_CHARS) -> str:
    """Fit a streaming reply into `lines × width`, cutting on word boundaries.

    The orb shows the *tail* of a reply (the newest words are the interesting
    ones), which is why the newline is put in front of the last lines rather
    than the text being trimmed from the end.
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    limit = lines * width
    if len(cleaned) <= limit:
        return _wrap(cleaned, width)
    tail = cleaned[-limit:]
    if " " in tail:
        tail = tail.split(" ", 1)[1] if len(tail.split(" ", 1)[0]) < 12 else tail
    return "…" + _wrap(tail, width)


def _wrap(text: str, width: int) -> str:
    words = text.split(" ")
    out: list[str] = []
    line = ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return "\n".join(out)


def tool_phrase(skill: str, *, language: str = "ar-MA") -> str:
    """The thin "thought" line for a running skill, in the right language."""
    darija, english = TOOL_PHRASES.get(skill, (skill, skill))
    return english if str(language).startswith("en") else darija


@dataclass(slots=True)
class ThemeEngine:
    """Turns core events into orb messages.  No state, no I/O, no framework.

    `MoodRenderer` and `ThemeEngine` are the same job split two ways in the
    LEVEL-05 class list (one shapes mood, one shapes everything else); keeping
    them as one object with clear methods avoids a second source of truth for the
    hue table, which is the thing that would drift.
    """

    language: str = "ar-MA"
    microphone_muted: bool = False
    _last_mood: str = ""
    _hue: int = MOOD_HUE_FALLBACK
    _depth: int = 0
    phrases: dict[str, str] = field(default_factory=dict)

    # ── state ────────────────────────────────────────────────────────
    def state(self, state: str, *, previous: str = "") -> StateMessage:
        name = str(state)
        return StateMessage(
            state=name,
            previous=previous,
            mic_muted=self.microphone_muted or name in {s.value for s in MUTED_STATES},
        )

    def motion(self, state: str) -> str:
        """The motion the orb should play — the meaning, not the pixels."""
        return STATE_MOTIONS.get(state, STATE_MOTIONS["dormant"])

    def is_overlay(self, state: str) -> bool:
        return state in OVERLAY_STATES

    def mutes_microphone(self, state: str) -> bool:
        """True while Atlas is speaking: the orb shows the mic as off, honestly."""
        return state in {item.value for item in MUTED_STATES}

    # ── mood ─────────────────────────────────────────────────────────
    def mood(
        self,
        label: str,
        *,
        energy: float = 0.5,
        warmth: float = 0.6,
        humor_allowed: bool = True,
        intensity: float = 0.5,
    ) -> MoodMessage:
        """One mood message: hue for the colour, energy for how much it moves."""
        name = str(label or "calm")
        hue = MOOD_HUES.get(name, MOOD_HUE_FALLBACK)
        self._last_mood = name
        self._hue = hue
        # Intensity is the persona's dial; the orb still caps it, because a 1.0
        # "frustrated" orb that shakes the whole card is a bug, not a mood.
        return MoodMessage(
            mood=name,
            hue=hue,
            energy=max(0.0, min(1.0, float(energy))),
            warmth=max(0.0, min(1.0, float(warmth))),
            humor_allowed=bool(humor_allowed),
        )

    @property
    def hue(self) -> int:
        return self._hue

    def blend_hue(self, label: str) -> int:
        """Where the orb should be *travelling* — the CSS transition does the rest.

        Kept server-side so the same rule can be asserted in a test, and so the
        orb never has to know the mood vocabulary.
        """
        return MOOD_HUES.get(str(label or "calm"), self._hue)

    # ── captions ─────────────────────────────────────────────────────
    def caption(
        self, text: str, *, role: str = "atlas", final: bool = False, language: str = ""
    ) -> CaptionMessage:
        tag = language or self.language
        return CaptionMessage(
            role="user" if role == "user" else "atlas",
            text=clamp_caption(text),
            language=tag,
            final=final,
        )

    # ── tool activity ────────────────────────────────────────────────
    def tool(
        self, skill: str, *, phase: str = "started", ok: bool = True, language: str = ""
    ) -> ToolMessage:
        return ToolMessage(
            skill=skill,
            phase="finished" if phase == "finished" else "started",
            ok=bool(ok),
            spoken=tool_phrase(skill, language=language or self.language),
        )

    def confirm(self, question: str, *, timeout_s: float = 10.0) -> ConfirmMessage:
        return ConfirmMessage(question=clamp_caption(question, lines=2), timeout_s=timeout_s)

    def degraded(self, *, degraded: bool, reason: str = "", mode: str = "lean") -> DegradedMessage:
        return DegradedMessage(degraded=bool(degraded), reason=reason, mode=mode)

    def level(self, value: float) -> LevelMessage:
        return LevelMessage(value=max(0.0, min(1.0, float(value))))


#: Every state the orb must be able to draw, in the order it is presented.
#: States only the ears' loop can produce (`atlas_audio.LoopState`).  The face
#: package does not import the ears — the orb is a *viewer* and must install and
#: run without any audio stack — so its extra two words ("an answer is still in
#: flight", "Atlas is off") are declared here, and the app-level test checks this
#: union against the real enum.
LOOP_ONLY_STATES: tuple[str, ...] = ("idle", "followup", "stopped")

ORB_STATES: tuple[str, ...] = (
    *(state.value for state in State),
    *LOOP_ONLY_STATES,
    *sorted(OVERLAY_STATES),
)


def state_report() -> dict[str, str]:
    """`{state: motion}` — what `atlas ui --states` and the tests read."""
    return {state: STATE_MOTIONS[state] for state in ORB_STATES}


__all__ = [
    "CAPTION_FADE_S",
    "CAPTION_LINES",
    "CAPTION_LINE_CHARS",
    "LOOP_ONLY_STATES",
    "MOOD_HUE_FALLBACK",
    "ORB_STATES",
    "OVERLAY_STATES",
    "STATE_MOTIONS",
    "TOOL_PHRASES",
    "ThemeEngine",
    "clamp_caption",
    "is_rtl",
    "state_report",
    "tool_phrase",
]
