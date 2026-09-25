"""The UI wire protocol — one schema, generated output for the web side.

Rule from the architecture doc (§3, no-duplication #4): the browser never
hand-writes these shapes.  `typescript()` emits the `.ts` interfaces from the
pydantic models, so the orb HTML and the Python events can never drift apart.

L5 delivers the bridge, the window and the orb itself; this module is the
contract they will speak.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

PROTOCOL_VERSION = 1


class UiMessage(BaseModel):
    """Base for everything sent to the orb."""

    v: int = PROTOCOL_VERSION
    kind: str = "message"


class StateMessage(UiMessage):
    kind: Literal["state"] = "state"
    state: str = "dormant"
    previous: str = ""
    mic_muted: bool = False


class CaptionMessage(UiMessage):
    kind: Literal["caption"] = "caption"
    role: Literal["user", "atlas"] = "atlas"
    text: str = ""
    language: str = "ar-MA"
    final: bool = False


class MoodMessage(UiMessage):
    kind: Literal["mood"] = "mood"
    mood: str = "calm"
    hue: int = 174
    energy: float = 0.5
    warmth: float = 0.6
    humor_allowed: bool = True


class LevelMessage(UiMessage):
    kind: Literal["level"] = "level"
    value: float = 0.0


class ToolMessage(UiMessage):
    kind: Literal["tool"] = "tool"
    skill: str = ""
    phase: Literal["started", "finished"] = "started"
    ok: bool = True
    spoken: str = ""


class ConfirmMessage(UiMessage):
    kind: Literal["confirm"] = "confirm"
    question: str = ""
    timeout_s: float = 10.0


class DegradedMessage(UiMessage):
    kind: Literal["degraded"] = "degraded"
    degraded: bool = False
    reason: str = ""
    mode: str = "lean"


class HelloMessage(UiMessage):
    kind: Literal["hello"] = "hello"
    version: str = ""
    states: list[str] = Field(default_factory=list)
    powers: dict[str, bool] = Field(
        default_factory=lambda: {"mic": True, "cloud": False, "vault": False}
    )


MESSAGE_TYPES: list[type[UiMessage]] = [
    HelloMessage,
    StateMessage,
    CaptionMessage,
    MoodMessage,
    LevelMessage,
    ToolMessage,
    ConfirmMessage,
    DegradedMessage,
]

# Mood → orb hue (see the L5 design language: mood is colour, state is motion).
MOOD_HUES: dict[str, int] = {
    "calm": 174,
    "happy": 42,
    "focused": 212,
    "frustrated": 12,
    "serious": 250,
    "tired": 268,
}


#: The translation table: EventBus topic → the message that carries it.  This is
#: the whole vocabulary the orb understands; `UiHub` keeps its per-topic policies
#: keyed by exactly these names and a test fails if the two ever drift apart.
TOPIC_MESSAGES: dict[str, type[UiMessage]] = {
    "state.changed": StateMessage,
    "llm.token": CaptionMessage,
    "llm.done": CaptionMessage,
    "mood.changed": MoodMessage,
    "audio.level": LevelMessage,
    "tool.started": ToolMessage,
    "tool.finished": ToolMessage,
    "confirm.requested": ConfirmMessage,
    "system.degraded": DegradedMessage,
    "system.error": StateMessage,
    "speaker.matched": CaptionMessage,
}

#: Topics the orb deliberately does not draw.  Declaring them means "the face
#: ignores this" is a decision someone can review — a new core event becomes a
#: failing test rather than a feature that is silently invisible.
SILENT_TOPICS: frozenset[str] = frozenset(
    {
        "event",  # the base dataclass; never published on its own
        "timing.recorded",  # a table in `atlas timings`, not a face
        "quota.exceeded",  # the router says it in words
        "audit.entry",  # the vault and the log, never the orb
        "cost.recorded",  # ditto
        "wake.detected",  # `state.changed` already says "listening"
    }
)


def message_for_topic(topic: str) -> type[UiMessage] | None:
    """Map an EventBus topic to the UI message that carries it."""
    return TOPIC_MESSAGES.get(topic)


def typescript() -> str:
    """Generate the TypeScript interfaces the orb consumes (never hand-edited)."""
    lines = [
        f"// AUTO-GENERATED from atlas_ui.protocol (v{PROTOCOL_VERSION}) — do not edit by hand.",
        "// Regenerate with: scripts/gen_ui_protocol.py (or: atlas ui protocol)",
        "",
    ]
    for model in MESSAGE_TYPES:
        lines.append(f"export interface {model.__name__} {{")
        for name, field in model.model_fields.items():  # type: ignore[attr-defined]
            lines.append(f"  {name}: {_ts_type(field.annotation)};")
        lines.append("}")
        lines.append("")
    union = " | ".join(model.__name__ for model in MESSAGE_TYPES)
    lines.append(f"export type AtlasMessage = {union};")
    return "\n".join(lines)


def javascript() -> str:
    """Generate `orb/protocol.js` — the runtime module the orb imports.

    TypeScript cannot run in WebView2 without a build step, and the whole point of
    L5 is no build step.  So the same models emit two files: `.ts` for editors and
    type checking, `.js` for the browser.  Hand-writing either one is how they
    drift, which is why both are generated from this module.
    """
    from atlas_ui.theme import CAPTION_FADE_S  # lazy: theme imports this module

    lines = [
        f"// AUTO-GENERATED from atlas_ui.protocol (v{PROTOCOL_VERSION}) — do not edit by hand.",
        "// Regenerate with: python scripts/gen_ui_protocol.py",
        "",
        f"export const PROTOCOL_VERSION = {PROTOCOL_VERSION};",
        f"export const CAPTION_FADE_S = {CAPTION_FADE_S};",
        "",
        "export const MOOD_HUES = Object.freeze({",
    ]
    lines += [f'  {json.dumps(name)}: {hue},' for name, hue in sorted(MOOD_HUES.items())]
    lines += [
        "});",
        "",
        "export const MESSAGE_KINDS = Object.freeze([",
    ]
    lines += [f'  {json.dumps(model.model_fields["kind"].default)},' for model in MESSAGE_TYPES]
    lines += ["]);", ""]
    return "\n".join(lines)


def _ts_type(annotation: object) -> str:
    """Map a Python annotation to TypeScript (containers before primitives)."""
    text = str(annotation)
    if "dict" in text or "Mapping" in text:
        if "bool" in text:
            return "Record<string, boolean>"
        if "int" in text or "float" in text:
            return "Record<string, number>"
        return "Record<string, unknown>"
    if "list" in text:
        if "float" in text or "int" in text:
            return "number[]"
        return "string[]"
    if "bool" in text:
        return "boolean"
    if "int" in text or "float" in text:
        return "number"
    return "string"


__all__ = [
    "MESSAGE_TYPES",
    "MOOD_HUES",
    "PROTOCOL_VERSION",
    "CaptionMessage",
    "ConfirmMessage",
    "DegradedMessage",
    "HelloMessage",
    "LevelMessage",
    "MoodMessage",
    "StateMessage",
    "ToolMessage",
    "UiMessage",
    "javascript",
    "message_for_topic",
    "typescript",
]
