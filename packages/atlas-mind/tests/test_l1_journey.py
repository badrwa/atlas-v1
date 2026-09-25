"""The L1 gate, as tests.

The level doc's "done when" is a *behaviour*, not a function: hold a coherent
conversation in Darija, switch mid-conversation to British English, survive the
network dying, and keep the pipeline's own overhead far below the model's.  Each
of those is one test below, with a fake provider standing in for the cloud so
the result is deterministic on any machine.
"""

from __future__ import annotations

import json
import time

import pytest
from support import fail, provider, say, session_for, silence

from atlas_core.config import AppConfig
from atlas_core.contracts import LanguageTag, estimate_tokens
from atlas_core.timings import TimingRecorder
from atlas_mind.chat import ChatSession
from atlas_mind.dialects import DARIJA, EN_GB
from atlas_mind.router import ProviderRouter


# ── a coherent conversation ──────────────────────────────────────────
async def test_three_turns_stay_coherent_and_in_character(config: AppConfig) -> None:
    session = session_for(
        config,
        say("salam sahbi, kifach n3awnek?"),
        say("wah, 3ndek 8 GB, mashi bezzaf walakin kafi."),
        say("safi, ghadi ndir hadchi daba."),
    )

    first = await session.send("salam")
    second = await session.send("chhal 3ndi memory?")
    third = await session.send("zid dirha")

    assert first.language == "ar-MA"
    assert second.text and third.text
    assert session.turns == 3
    assert len(session.history) == 6, "three exchanges, two messages each"


async def test_switching_to_english_changes_the_prompt_and_the_voice(config: AppConfig) -> None:
    """Requirement 4's second half: British English on request, mid-conversation."""
    session = session_for(
        config,
        say("salam sahbi."),
        say("Right then. All quiet on this end — what do you need?"),
    )

    await session.send("salam")
    assert session.current_language == "ar-MA"

    result = await session.send("speak English please")

    assert session.current_language == "en-GB"
    assert result.language == "en-GB"
    assert "Right then" in result.text

    prompt = session.persona.system_prompt(language="en-GB")
    assert EN_GB.label in prompt
    assert DARIJA.label not in prompt


async def test_the_prompt_sent_to_the_model_uses_the_switched_language(config: AppConfig) -> None:
    fake = provider(say("salam."), say("Right, sorted."))
    session = ChatSession(config, ProviderRouter([fake]))

    await session.send("salam")
    await session.send("b l'ingliziya")

    second_system = fake.calls[-1].system or ""
    assert "British English" in second_system
    assert "Moroccan Darija" not in second_system


# ── the network dying ────────────────────────────────────────────────
async def test_network_death_produces_an_honest_reply_not_a_hang(config: AppConfig) -> None:
    session = ChatSession(config, ProviderRouter([provider(fail("unavailable"))]))
    started = time.perf_counter()
    result = await session.send("salam")
    elapsed = (time.perf_counter() - started) * 1000

    assert result.provider == "canned", "the fallback must identify itself"
    assert result.text.strip(), "silence is not an acceptable answer"
    assert "سمح ليا" in result.text, "and it must be said in the owner's language"
    assert elapsed < 500, "a dead network must fail fast, not wait for a timeout"


async def test_a_429_moves_silently_to_the_next_provider(config: AppConfig) -> None:
    flaky = provider(fail("rate_limited"), name="flaky")
    healthy = provider(say("jawab men provider akhor"), name="healthy")
    session = ChatSession(config, ProviderRouter([flaky, healthy], retries=0))

    result = await session.send("salam")

    assert result.provider == "healthy"
    assert "provider akhor" in result.text
    assert len(flaky.calls) == 1, "a 429 is not retried on the same provider in one turn"


async def test_a_provider_that_says_nothing_is_not_an_answer(config: AppConfig) -> None:
    """The nastiest failure mode is not an error — it is a silent empty stream."""
    mute = provider(silence(), name="mute")
    healthy = provider(say("safi, ana hna."), name="healthy")
    session = ChatSession(config, ProviderRouter([mute, healthy]))

    result = await session.send("salam")

    assert result.provider == "healthy"
    assert result.text == "safi, ana hna."


async def test_every_provider_silent_still_speaks(config: AppConfig) -> None:
    session = session_for(config, silence(), silence())
    result = await session.send("salam")
    assert result.provider == "canned"
    assert result.text.strip(), "empty providers must not produce an empty reply"


async def test_both_languages_are_available_when_offline(config: AppConfig) -> None:
    session = ChatSession(config, ProviderRouter([provider(fail("unavailable"))]))
    session.language.route("", hint="en-GB")

    result = await session.send("hello")

    assert result.language == "en-GB"
    assert "connection" in result.text.lower(), "the canned line must speak English too"


# ── the latency budget ───────────────────────────────────────────────
async def test_pipeline_overhead_is_a_small_fraction_of_a_turn(config: AppConfig) -> None:
    """The plan demands < 50 ms of *our* overhead. Model latency is not ours."""
    session = session_for(config, *[say("safi.") for _ in range(20)])

    await session.send("salam")  # warm-up: templates and language regexes

    started = time.perf_counter()
    for _ in range(10):
        await session.send("salam")
    per_turn_ms = (time.perf_counter() - started) * 1000 / 10

    assert per_turn_ms < 50, f"pipeline overhead {per_turn_ms:.1f} ms/turn is too high"


async def test_the_context_budget_is_never_exceeded(config: AppConfig) -> None:
    """TTFT is what we are protecting: the prompt must not grow with the chat."""
    session = session_for(config, *[say("safi, hadchi jawab twil chwiya. " * 12) for _ in range(30)])

    for _ in range(12):
        await session.send("wach nta hna? " * 8)

    build = session.context.last_build
    assert build["budget"] == config.mind.context_token_budget
    total = build["system"] + build["memory"] + build["history"] + build["user"]
    assert total <= build["budget"], f"context {total} exceeded budget {build['budget']}"
    assert estimate_tokens("x") > 0


async def test_a_long_conversation_keeps_the_window_small(config: AppConfig) -> None:
    session = session_for(config, *[say("safi.") for _ in range(40)])
    for _ in range(20):
        await session.send("salam")
    assert len(session.history) <= session.context.max_turns * 2


# ── observability ────────────────────────────────────────────────────
async def test_every_turn_is_recorded_with_its_provider(config: AppConfig) -> None:
    timings = TimingRecorder(enabled=False)
    session = session_for(config, say("safi."), say("safi."), timings=timings)

    await session.send("salam")
    await session.send("kifach?")

    assert [record.provider for record in timings.records] == ["fake", "fake"]
    assert all(record.language == "ar-MA" for record in timings.records)
    assert timings.records[0].ttft_ms is not None


async def test_snapshot_is_json_serialisable_for_the_ui(config: AppConfig) -> None:
    """L5's orb reads this over the wire; it must survive json.dumps."""
    session = session_for(config, say("safi."))
    await session.send("salam")
    assert "turns" in json.dumps(session.snapshot(), ensure_ascii=False)


@pytest.mark.parametrize("language", ["ar-MA", "en-GB"])
async def test_a_turn_works_in_both_languages(config: AppConfig, language: LanguageTag) -> None:
    session = session_for(config, say("safi."))
    session.language.route("", hint=language)
    result = await session.send("salam" if language == "ar-MA" else "hello")
    assert result.language == language
    assert result.text
