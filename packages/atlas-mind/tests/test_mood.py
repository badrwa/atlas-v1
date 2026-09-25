"""Mood: predictable, decaying, and never overriding what the owner said."""

from __future__ import annotations

from atlas_mind.envelope import ReplyEnvelope
from atlas_mind.mood import EMOTION_EFFECT, MoodEngine, MoodView


def test_starts_calm_and_allows_humour() -> None:
    engine = MoodEngine()
    assert engine.view.label == "calm"
    assert engine.view.humor_allowed is True


def test_the_models_own_read_wins() -> None:
    engine = MoodEngine()
    engine.observe_envelope(ReplyEnvelope(reply="safi", emotion="frustrated", mood_delta=""))
    assert engine.view.label == "frustrated"
    assert engine.view.humor_allowed is False, "a frustrated Atlas does not joke"


def test_owner_frustration_overrides_a_playful_reply() -> None:
    engine = MoodEngine()
    engine.observe_owner("hadi machi khdam, 3yit menha")
    assert engine.view.label in {"frustrated", "tired"}
    assert engine.view.humor_allowed is False


def test_a_quiet_turn_decays_towards_calm() -> None:
    """Nothing notable happens → Atlas drifts back to calm, it does not snap."""
    engine = MoodEngine(decay=0.3)
    engine.observe_envelope(ReplyEnvelope(reply="safi", emotion="happy"))
    assert engine.view.label == "happy"

    for _ in range(6):
        engine.observe_reply("safi.")  # no markers at all

    assert engine.view.label == "calm"
    assert engine.view.humor_allowed is True


def test_markers_on_the_fast_path_are_enough_to_move_the_dial() -> None:
    """Both labels are the same idea: the orb shows its warm hue either way."""
    engine = MoodEngine()
    engine.observe_reply("haha, typical, that happened again as usual")
    assert engine.view.label in {"happy", "amused"}
    assert engine.view.humor_allowed is True
    assert engine.view.warmth > 0.6


def test_an_invented_emotion_never_reaches_the_mood_engine() -> None:
    """Validation happens at the envelope boundary, so mood always gets a known label."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReplyEnvelope(reply="safi", emotion="vibing")
    assert MoodEngine().view.label == "calm"


def test_every_emotion_the_envelope_allows_has_an_effect() -> None:
    from atlas_mind.envelope import EMOTIONS

    assert set(EMOTIONS) == set(EMOTION_EFFECT), "schema and mood table must stay in step"


def test_view_is_serialisable_for_the_ui() -> None:
    payload = MoodEngine().view.as_dict()
    assert set(payload) == {"label", "energy", "warmth", "humor_allowed", "intensity"}
    assert all(isinstance(value, (str, float, bool)) for value in payload.values())


def test_reset_returns_to_calm() -> None:
    engine = MoodEngine()
    engine.observe_owner("urgent, mochkil kbir")
    engine.reset()
    assert engine.view == MoodView()
    assert engine.history == []


def test_history_records_the_journey_for_later_tuning() -> None:
    engine = MoodEngine()
    engine.observe_envelope(ReplyEnvelope(reply="safi", emotion="happy"))
    engine.observe_reply("safi")
    assert engine.history[0] == "happy"
