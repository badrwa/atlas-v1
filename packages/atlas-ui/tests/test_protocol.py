"""The UI protocol is a contract with a browser that cannot import Python."""

from __future__ import annotations

from atlas_ui.protocol import (
    MESSAGE_TYPES,
    MOOD_HUES,
    PROTOCOL_VERSION,
    CaptionMessage,
    StateMessage,
    message_for_topic,
    typescript,
)


def test_every_message_carries_the_protocol_version() -> None:
    for model in MESSAGE_TYPES:
        instance = model()
        assert instance.v == PROTOCOL_VERSION
        assert isinstance(instance.kind, str)


def test_generated_typescript_is_stable_and_complete() -> None:
    generated = typescript()
    assert generated.startswith("// AUTO-GENERATED")
    for model in MESSAGE_TYPES:
        assert f"export interface {model.__name__} {{" in generated
    assert "export type AtlasMessage =" in generated
    assert typescript() == generated, "generation must be deterministic"


def test_typescript_maps_python_types_sensibly() -> None:
    generated = typescript()
    assert "mic_muted: boolean;" in generated
    assert "value: number;" in generated
    assert "text: string;" in generated


def test_topic_mapping_covers_the_events_the_ui_needs() -> None:
    state_model = message_for_topic("state.changed")
    assert state_model is StateMessage
    assert message_for_topic("llm.token") is CaptionMessage
    assert state_model is not None and state_model().kind == "state"
    assert message_for_topic("nope") is None


def test_mood_hues_cover_every_mood() -> None:
    for mood in ("calm", "happy", "focused", "frustrated", "serious", "tired"):
        assert 0 <= MOOD_HUES[mood] <= 360


def test_caption_message_supports_arabic() -> None:
    message = CaptionMessage(text="salam صاحبي", language="ar-MA")
    assert message.language == "ar-MA"
    assert "صاحبي" in message.text
