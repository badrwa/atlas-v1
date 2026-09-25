"""The reply envelope: tolerant about packaging, strict about meaning."""

from __future__ import annotations

import json

import pytest

from atlas_core.fakes import FakeProvider
from atlas_mind.envelope import (
    EMOTIONS,
    REPLY_SCHEMA,
    EnvelopeError,
    ReplyEnvelope,
    parse_reply,
    repair_prompt,
    to_gemini_schema,
)

GOOD = {
    "reply": "salam sahbi, kifach n3awnek?",
    "language": "ar-MA",
    "emotion": "happy",
    "mood_delta": "warm",
    "followup": True,
}


# ── parsing ──────────────────────────────────────────────────────────
def test_clean_json_parses() -> None:
    envelope = parse_reply(json.dumps(GOOD, ensure_ascii=False))
    assert envelope.reply.startswith("salam")
    assert envelope.language == "ar-MA"
    assert envelope.emotion == "happy"
    assert envelope.followup is True


def test_code_fences_and_chatter_are_tolerated() -> None:
    raw = f"Here you go:\n```json\n{json.dumps(GOOD, ensure_ascii=False)}\n```\nEnjoy!"
    assert parse_reply(raw).emotion == "happy"


def test_leading_prose_without_fences_is_tolerated() -> None:
    raw = "Sure! " + json.dumps(GOOD, ensure_ascii=False)
    assert parse_reply(raw).reply.startswith("salam")


def test_optional_fields_default() -> None:
    envelope = parse_reply('{"reply": "safi", "language": "ar-MA", "emotion": "calm"}')
    assert envelope.followup is False
    assert envelope.tool_calls == []
    assert envelope.mood_delta == ""


# ── rejection ────────────────────────────────────────────────────────
def test_plain_text_is_rejected_not_guessed() -> None:
    with pytest.raises(EnvelopeError):
        parse_reply("salam, kifach nta?")


def test_invented_language_is_rejected() -> None:
    bad = dict(GOOD, language="fr-FR")
    with pytest.raises(EnvelopeError) as excinfo:
        parse_reply(json.dumps(bad))
    assert "language" in str(excinfo.value)


def test_invented_emotion_is_rejected() -> None:
    """The orb maps emotions to colours — free text must not reach it."""
    bad = dict(GOOD, emotion="extremely warm indeed, like tea on a cold day")
    with pytest.raises(EnvelopeError) as excinfo:
        parse_reply(json.dumps(bad))
    assert "emotion" in str(excinfo.value)


def test_missing_reply_is_rejected() -> None:
    with pytest.raises(EnvelopeError) as excinfo:
        parse_reply('{"language": "ar-MA", "emotion": "calm"}')
    assert "reply" in str(excinfo.value)


def test_truncated_json_reports_the_position() -> None:
    with pytest.raises(EnvelopeError) as excinfo:
        parse_reply('{"reply": "salam", "language":')
    assert "line" in str(excinfo.value)


def test_a_json_array_is_not_an_envelope() -> None:
    with pytest.raises(EnvelopeError):
        parse_reply('["salam"]')


# ── the repair prompt ────────────────────────────────────────────────
def test_repair_prompt_carries_the_problem_and_the_output() -> None:
    prompt = repair_prompt('{"reply": 1}', "reply: Input should be a valid string")
    assert "Input should be a valid string" in prompt
    assert '{"reply": 1}' in prompt
    assert "JSON" in prompt


def test_repair_prompt_truncates_very_long_output() -> None:
    """A runaway answer must not become a runaway prompt (it costs TTFT)."""
    raw = "x" * 5000
    prompt = repair_prompt(raw, "boom")
    assert len(prompt) < len(raw)
    assert raw[:600] in prompt
    assert "xxxxx" * 200 not in prompt


# ── schema ───────────────────────────────────────────────────────────
def test_schema_covers_every_emotion_the_orb_knows() -> None:
    assert REPLY_SCHEMA["properties"]["emotion"]["enum"] == list(EMOTIONS)
    assert REPLY_SCHEMA["required"] == ["reply", "language", "emotion"]


def test_schema_is_valid_json_for_the_providers() -> None:
    assert json.loads(json.dumps(REPLY_SCHEMA)) == REPLY_SCHEMA
    assert to_gemini_schema(REPLY_SCHEMA)["type"] == "OBJECT"
    assert to_gemini_schema(REPLY_SCHEMA)["properties"]["reply"]["type"] == "STRING"


def test_gemini_translation_leaves_the_original_alone() -> None:
    """Providers translate their own copy; one schema keeps one meaning."""
    before = json.dumps(REPLY_SCHEMA, sort_keys=True)
    to_gemini_schema(REPLY_SCHEMA)
    assert json.dumps(REPLY_SCHEMA, sort_keys=True) == before


def test_gemini_object_without_properties_gets_an_empty_map() -> None:
    assert to_gemini_schema({"type": "object"}) == {"type": "OBJECT", "properties": {}}


def test_tool_intents_become_requests() -> None:
    envelope = parse_reply(
        '{"reply": "safi", "language": "ar-MA", "emotion": "calm",'
        ' "tool_calls": [{"name": "system_stats", "arguments": {"what": "ram"}}]}'
    )
    requests = envelope.to_tool_requests()
    assert requests[0].name == "system_stats"
    assert requests[0].arguments == {"what": "ram"}


def test_envelope_round_trips_through_a_fake_provider() -> None:
    """The shape a provider really returns: one JSON blob, streamed in pieces."""
    from atlas_core.contracts import LlmRequest
    from atlas_core.contracts import TextDelta as Delta

    payload = json.dumps(GOOD, ensure_ascii=False)
    provider = FakeProvider(
        name="fake",
        script=[[{"text": payload[:20]}, {"text": payload[20:]}], []],
    )

    async def collect() -> str:
        chunks = []
        async for event in provider.stream(LlmRequest()):
            if isinstance(event, Delta):
                chunks.append(event.text)
        return "".join(chunks)

    import asyncio

    envelope = parse_reply(asyncio.run(collect()))
    assert envelope.language == "ar-MA"
    assert isinstance(envelope, ReplyEnvelope)
