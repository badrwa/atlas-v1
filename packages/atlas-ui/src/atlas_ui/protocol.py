"""The UI wire protocol — one schema, generated output for the web side.

Rule from the architecture doc (§3, no-duplication #4): the browser never
hand-writes these shapes.  `typescript()` emits the `.ts` interfaces from the
pydantic models, so the orb HTML and the Python events can never drift apart.

L5 delivers the bridge, the window and the orb itself; this module is the
contract they will speak.
"""

from __future__ import annotations

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


def message_for_topic(topic: str) -> type[UiMessage] | None:
    """Map an EventBus topic to the UI message that carries it."""
    mapping: dict[str, type[UiMessage]] = {
        "state.changed": StateMessage,
        "llm.token": CaptionMessage,
        "mood.changed": MoodMessage,
        "audio.level": LevelMessage,
        "tool.started": ToolMessage,
        "tool.finished": ToolMessage,
        "confirm.requested": ConfirmMessage,
        "system.degraded": DegradedMessage,
    }
    return mapping.get(topic)


def typescript() -> str:
    """Generate the TypeScript interfaces the orb consumes (never hand-edited)."""
    lines = [
        f"// AUTO-GENERATED from atlas_ui.protocol (v{PROTOCOL_VERSION}) — do not edit by hand.",
        "// Regenerate with: python -m atlas ui-protocol > atlas-ui/orb/protocol.ts",
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
    "message_for_topic",
    "typescript",
]
