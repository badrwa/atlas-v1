"""Structured turns end to end: envelope, one repair, and the plain-text path.

The rule these tests defend: **a bad envelope never eats a good reply.**  A
model that returns unparseable metadata still gets spoken; it just loses its
metadata, silently, exactly once.
"""

from __future__ import annotations

import json

from support import Turn, provider, say, session_for

from atlas_core.config import AppConfig
from atlas_core.timings import TimingRecorder
from atlas_mind.chat import ChatSession, TurnResult
from atlas_mind.mood import MoodEngine
from atlas_mind.router import ProviderRouter


def good(reply: str = "salam sahbi", **extra: object) -> Turn:
    """One structured turn: the JSON envelope a provider should return."""
    payload: dict[str, object] = {
        "reply": reply,
        "language": "ar-MA",
        "emotion": "happy",
        "mood_delta": "warm",
        "followup": False,
        **extra,
    }
    return [{"text": json.dumps(payload, ensure_ascii=False)}]


def structured(config: AppConfig, *turns: Turn, **kwargs: object) -> ChatSession:
    return session_for(config, *turns, structured=True, **kwargs)  # type: ignore[arg-type]


def empty_result() -> TurnResult:
    return TurnResult()


# ── the happy path ───────────────────────────────────────────────────
async def test_structured_turn_fills_the_whole_result(config: AppConfig) -> None:
    session = structured(config, good(), [])
    result = await session.send("salam")

    assert result.text == "salam sahbi"
    assert result.structured is True
    assert result.emotion == "happy"
    assert result.repaired is False
    assert result.mood["label"] == "happy"
    assert result.followup is False


async def test_structured_turn_yields_only_the_spoken_words(config: AppConfig) -> None:
    """The UI and the (future) voice must never see brackets and quotes."""
    session = structured(config, good("kifach n3awnek?"), [])
    spoken = "".join([delta async for delta in session.stream("salam", _result=empty_result())])

    assert spoken == "kifach n3awnek?"
    assert "{" not in spoken and '"' not in spoken


async def test_followup_flag_survives(config: AppConfig) -> None:
    session = structured(config, good("wach bghiti chi haja?", followup=True), [])
    result = await session.send("salam")
    assert result.followup is True


async def test_language_from_the_envelope_overrides_the_guess(config: AppConfig) -> None:
    """The router proposes, the model disposes: it knows what it actually wrote."""
    session = structured(config, good("Alright then.", language="en-GB"), [])
    result = await session.send("salam")
    assert result.language == "en-GB"


# ── the repair path ──────────────────────────────────────────────────
async def test_one_bad_envelope_is_repaired_silently(config: AppConfig) -> None:
    broken = [{"text": '{"reply": "salam", "emotion": "joyful", "language": "ar-MA"}'}]
    fake = provider(broken, good("salam sahbi"))
    session = ChatSession(config, ProviderRouter([fake]), structured=True)

    result = await session.send("salam")

    assert result.text == "salam sahbi"
    assert result.repaired is True
    assert len(fake.calls) == 2, "exactly one repair attempt — not a retry loop"


async def test_repair_happens_only_once(config: AppConfig) -> None:
    bad = [{"text": "not json at all"}]
    fake = provider(bad, bad)
    session = ChatSession(config, ProviderRouter([fake]), structured=True)

    result = await session.send("salam")

    assert len(fake.calls) == 2
    assert result.repaired is False, "the second failure is a fallback, not a repair"


async def test_unparseable_output_still_speaks(config: AppConfig) -> None:
    """Losing metadata is acceptable. Losing the answer is not."""
    session = structured(config, [{"text": "salam, hadchi jawab 3adi."}], [{"text": "salam."}])
    result = await session.send("salam")

    assert result.text == "salam, hadchi jawab 3adi."
    assert result.emotion == "", "a salvaged reply must not claim an emotion"


async def test_json_with_the_wrong_shape_is_salvaged_not_dumped(config: AppConfig) -> None:
    """A reply that *looks* like JSON: speak the reply field, drop the wrapper."""
    payload = '{"reply": "safi khoya", "emotion": "warm-ish"}'
    session = structured(config, [{"text": payload}], [{"text": payload}])

    result = await session.send("salam")

    assert result.text == "safi khoya"
    assert not result.text.startswith("{")


async def test_a_repaired_turn_still_records_timings(config: AppConfig) -> None:
    timings = TimingRecorder(enabled=False)
    session = structured(
        config, [{"text": "{oops}"}], good("safi"), timings=timings
    )

    result = await session.send("salam")

    assert result.repaired is True
    assert timings.records[-1].provider == "fake"


# ── the fast path stays untouched ────────────────────────────────────
async def test_fast_mode_does_not_ask_for_json(config: AppConfig) -> None:
    session = session_for(config, say("salam sahbi"), [], structured=False)
    result = await session.send("salam")

    assert result.text == "salam sahbi"
    assert result.structured is False
    assert result.emotion == ""
    assert result.mood["label"] in {"calm", "happy"}, "mood is still tracked, locally"


async def test_fast_mode_streams_more_than_one_delta(config: AppConfig) -> None:
    session = session_for(config, [{"text": "salam "}, {"text": "sahbi"}], [])
    deltas = [delta async for delta in session.stream("salam", _result=empty_result())]
    assert deltas == ["salam ", "sahbi"]


# ── mood across turns ────────────────────────────────────────────────
async def test_mood_follows_the_conversation(config: AppConfig) -> None:
    engine = MoodEngine()
    session = structured(
        config,
        good("haha, zwin bezzaf", emotion="amused", mood_delta="playful"),
        [{"text": '{"reply": "salam", "language": "ar-MA", "emotion": "frustrated"}'}],
        [],
        mood=engine,
    )

    first = await session.send("salam")
    assert first.mood["label"] == "happy"
    assert first.mood["humor_allowed"] is True

    second = await session.send("machi khdam")
    assert second.mood["label"] == "frustrated"
    assert second.mood["humor_allowed"] is False, "no jokes when things are broken"


async def test_mood_survives_a_reset(config: AppConfig) -> None:
    engine = MoodEngine()
    session = structured(config, good(emotion="tired"), [], mood=engine)
    await session.send("salam")
    session.reset()
    assert session.mood.view.label == "calm"
