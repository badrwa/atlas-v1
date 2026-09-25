"""The reply envelope — one call that returns words *and* metadata.

Plain text streaming is the fast path and stays the default: it gives the
lowest time-to-first-token, which is what talking to Atlas feels like.  The
envelope is the *other* path, used when the caller needs to know more than the
words — which language the model actually answered in, how it read the owner's
mood, whether it wants to call a tool or ask a follow-up.

Two rules make this safe to use:

1. **Validation is not optional.** A model that returns `"mood_delta": "very
   warm indeed"` must not smuggle a paragraph into a field the orb reads as a
   colour.
2. **A bad envelope never eats a good reply.** One silent repair attempt, and if
   that fails the raw text is spoken as-is.  Losing metadata is acceptable;
   losing the answer is not.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from atlas_core.contracts import LanguageTag, ToolCallRequest

log = logging.getLogger(__name__)

# The orb has six hues (see atlas-ui).  Emotions map onto them; keep this list
# and MOOD_HUES in step — `atlas ui-protocol` is the shared contract.
EMOTIONS: tuple[str, ...] = (
    "calm",
    "happy",
    "amused",
    "playful",
    "focused",
    "serious",
    "tired",
    "frustrated",
)

MOOD_DELTAS: tuple[str, ...] = ("", "warm", "calm", "playful", "focused", "serious", "tired")


class ToolIntent(BaseModel):
    """A tool the model *wants* to call. Permission is decided later, by L7."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ReplyEnvelope(BaseModel):
    """What one structured turn returns.

    The enums here duplicate the JSON schema on purpose: a schema is a request,
    not a guarantee — providers paraphrase it, and some ignore it entirely.  The
    validation below is what actually protects the orb and the mood engine.
    """

    reply: str
    language: LanguageTag = "ar-MA"
    emotion: str = "calm"
    mood_delta: str = ""
    tool_calls: list[ToolIntent] = Field(default_factory=list)
    followup: bool = False

    @field_validator("emotion")
    @classmethod
    def _known_emotion(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in EMOTIONS:
            raise ValueError(f"unknown emotion {value!r}; use one of: {', '.join(EMOTIONS)}")
        return cleaned

    @field_validator("mood_delta")
    @classmethod
    def _known_delta(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in MOOD_DELTAS:
            raise ValueError(f"unknown mood_delta {value!r}; use one of: {', '.join(MOOD_DELTAS)}")
        return cleaned

    def to_tool_requests(self) -> list[ToolCallRequest]:
        return [
            ToolCallRequest(name=call.name, arguments=call.arguments)
            for call in self.tool_calls
        ]


class EnvelopeError(ValueError):
    """The model's output was not a usable envelope (kept for the repair prompt)."""


# ── schema handed to the providers ───────────────────────────────────
REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
            "description": "What Atlas says out loud. Plain speech, no markdown.",
        },
        "language": {
            "type": "string",
            "enum": ["ar-MA", "en-GB"],
            "description": "The language of the reply.",
        },
        "emotion": {
            "type": "string",
            "enum": list(EMOTIONS),
            "description": "How Atlas sounds right now, not how the user feels.",
        },
        "mood_delta": {
            "type": "string",
            "enum": list(MOOD_DELTAS),
            "description": "How this turn changed Atlas's mood. Empty when unchanged.",
        },
        "followup": {
            "type": "boolean",
            "description": "True when Atlas asked a question and expects an answer.",
        },
    },
    "required": ["reply", "language", "emotion"],
}
# NOTE: tools are deliberately *not* in this schema yet.  The L1 model has a
# small context and is easily distracted; listing `tool_calls` on every turn
# made it invent calls.  L7 adds the field back together with the registry that
# can actually execute (and permission-check) it — see `ToolIntent`.

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BRACES = re.compile(r"\{.*\}", re.DOTALL)


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Gemini's REST API wants OpenAPI-style types in upper case.

    Same schema, different spelling — translating it here keeps every provider
    fed from one definition instead of three hand-maintained copies.
    """
    if not isinstance(schema, dict):
        return schema
    translated: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            translated[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            translated[key] = {
                name: to_gemini_schema(sub) for name, sub in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            translated[key] = to_gemini_schema(value)
        else:
            translated[key] = value
    if translated.get("type") == "OBJECT" and "properties" not in translated:
        translated["properties"] = {}
    return translated


def _strip_to_json(raw: str) -> str:
    """Pull the JSON object out of whatever wrapper the model used."""
    text = raw.strip()
    if fence := _FENCE.search(text):
        text = fence.group(1).strip()
    if not text.startswith("{") and (braces := _BRACES.search(text)):
        text = braces.group(0)
    return text


def parse_reply(raw: str) -> ReplyEnvelope:
    """Parse a model reply into an envelope, or raise `EnvelopeError`.

    Tolerant about packaging (code fences, leading chatter), strict about fields:
    an unknown language or emotion is a validation error, not something to guess.
    """
    text = _strip_to_json(raw)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EnvelopeError(f"not JSON: {exc.msg} at line {exc.lineno}") from exc
    if not isinstance(payload, dict):
        raise EnvelopeError(f"expected a JSON object, got {type(payload).__name__}")
    try:
        return ReplyEnvelope.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(part) for part in first["loc"]) or "envelope"
        raise EnvelopeError(f"{where}: {first['msg']}") from exc


def repair_prompt(raw: str, error: str) -> str:
    """The single retry: show the model its own output and what was wrong."""
    snippet = raw.strip()[:600]
    return (
        "Your previous answer was not valid JSON for the reply schema.\n"
        f"Problem: {error}\n"
        f"Your output was:\n{snippet}\n\n"
        "Reply with the corrected JSON object only — same content, valid fields. "
        "No explanation, no code fences."
    )


def envelope_instructions(language: LanguageTag | str) -> str:
    """System-prompt addendum for structured turns (kept next to the schema)."""
    return (
        "Answer as a single JSON object with exactly these fields: "
        "reply (string, what you say out loud), "
        f"language ({language} — the language you actually used), "
        f"emotion (one of: {', '.join(EMOTIONS)}), "
        f"mood_delta (one of: {', '.join(m for m in MOOD_DELTAS if m) or 'empty'}), "
        "tool_calls (array of {name, arguments}; usually empty), "
        "followup (boolean: true only if you asked a question).\n"
        "Inside \"reply\": plain speech, no markdown, no emoji, no code fences."
    )


__all__ = [
    "EMOTIONS",
    "MOOD_DELTAS",
    "REPLY_SCHEMA",
    "EnvelopeError",
    "ReplyEnvelope",
    "ToolIntent",
    "envelope_instructions",
    "parse_reply",
    "repair_prompt",
    "to_gemini_schema",
]
