"""Persona prompt, context budget and the full chat turn."""

from __future__ import annotations

import pytest

from atlas_core.config import AppConfig
from atlas_core.contracts import Message, Role, estimate_tokens
from atlas_core.events import EventBus, ReplyFinished, TokenDelta
from atlas_core.fakes import FakeProvider
from atlas_core.timings import TimingRecorder
from atlas_mind.chat import ChatSession
from atlas_mind.context import ContextBuilder
from atlas_mind.persona import MoodView, Persona
from atlas_mind.router import ProviderRouter


@pytest.fixture
def config() -> AppConfig:
    return AppConfig.model_validate({"app": {"call_name": "صاحبي", "language": "ar-MA"}})


# ── persona ──────────────────────────────────────────────────────────
def test_persona_prompt_has_identity_language_and_rules(config: AppConfig) -> None:
    prompt = Persona(config).system_prompt(language="ar-MA")
    assert "Atlas" in prompt
    assert "Moroccan Darija" in prompt
    assert "صاحبي" in prompt
    assert "Never claim you did something you did not do" in prompt


def test_persona_switches_dialect_completely(config: AppConfig) -> None:
    persona = Persona(config)
    darija = persona.system_prompt(language="ar-MA")
    british = persona.system_prompt(language="en-GB")
    assert "British English" in british
    assert "Moroccan Darija" not in british
    assert "awesome" in british  # listed as banned, not used
    assert darija != british


def test_persona_suppresses_jokes_when_mood_says_so(config: AppConfig) -> None:
    persona = Persona(config)
    calm = persona.system_prompt(language="ar-MA", mood=MoodView(label="calm"))
    serious = persona.system_prompt(
        language="ar-MA",
        mood=MoodView(label="serious", energy=0.2, warmth=0.3, humor_allowed=False),
    )
    assert "no jokes" not in calm.lower()
    assert "no jokes" in serious.lower()


def test_persona_memory_and_tools_are_included_only_when_present(config: AppConfig) -> None:
    persona = Persona(config)
    plain = persona.system_prompt(language="ar-MA")
    assert "What you remember" not in plain
    rich = persona.system_prompt(
        language="ar-MA", memory="- dentist Tuesday", tool_names=["weather", "open_app"]
    )
    assert "dentist Tuesday" in rich
    assert "weather, open_app" in rich


def test_switch_notice_is_short_and_in_the_target_language(config: AppConfig) -> None:
    notice = Persona(config).switch_notice("en-GB")
    assert "British English" in notice
    assert len(notice) < 400, "a switch notice must not cost meaningful tokens"


# ── context budget ───────────────────────────────────────────────────
def test_context_never_exceeds_the_budget() -> None:
    builder = ContextBuilder(budget_tokens=200, memory_budget=60, max_turns=20)
    history = [
        Message(role=Role.USER if i % 2 == 0 else Role.ASSISTANT, content="x" * 400)
        for i in range(20)
    ]
    request = builder.build(system="s" * 100, user_text="chno ljaw?", history=history)
    total = estimate_tokens(request.system or "") + sum(
        estimate_tokens(m.content) for m in request.messages
    )
    assert total <= 200 + 5, f"context overflowed: {total} tokens"


def test_context_keeps_the_newest_turns() -> None:
    builder = ContextBuilder(budget_tokens=120, memory_budget=0, max_turns=10)
    history = [
        Message(role=Role.USER, content="OLD TURN"),
        Message(role=Role.ASSISTANT, content="old reply"),
        Message(role=Role.USER, content="NEWEST TURN"),
    ]
    request = builder.build(system="short", user_text="daba", history=history)
    joined = " ".join(m.content for m in request.messages)
    assert "NEWEST TURN" in joined
    assert request.messages[-1].content == "daba"


def test_context_drops_memory_when_there_is_no_room() -> None:
    builder = ContextBuilder(budget_tokens=40, memory_budget=400)
    request = builder.build(system="s" * 120, user_text="salam", memory="- fact one\n- fact two")
    assert "fact one" not in " ".join(m.content for m in request.messages)


def test_context_reports_its_composition() -> None:
    builder = ContextBuilder(budget_tokens=500)
    builder.build(system="persona here", user_text="salam", memory="- fact")
    assert builder.last_build["budget"] == 500
    assert builder.last_build["turns_kept"] >= 1


# ── chat session ─────────────────────────────────────────────────────
async def test_chat_turn_streams_timed_and_evented(config: AppConfig, tmp_path) -> None:
    bus = EventBus()
    tokens = bus.subscribe("llm.token")
    finished = bus.subscribe("llm.done")
    timings = TimingRecorder(tmp_path / "timings.jsonl")
    provider = FakeProvider(name="gemini", script=[[{"text": "salam "}, {"text": "sahbi"}]])

    session = ChatSession(
        config, ProviderRouter([provider]), timings=timings, events=bus
    )
    result = await session.send("salam")

    assert result.text == "salam sahbi"
    assert result.provider == "gemini"
    assert result.language == "ar-MA"
    assert result.ttft_ms is not None and result.ttft_ms >= 0
    assert timings.records[-1].provider == "gemini"
    token_event = await tokens.get()
    done_event = await finished.get()
    assert isinstance(token_event, TokenDelta)
    assert isinstance(done_event, ReplyFinished)
    assert token_event.text == "salam "
    assert done_event.text == "salam sahbi"
    assert session.turns == 1


async def test_chat_switches_language_mid_conversation(config: AppConfig) -> None:
    provider = FakeProvider(
        name="fake",
        script=[[{"text": "salam"}], [{"text": "Right, sorted."}]],
    )
    session = ChatSession(config, ProviderRouter([provider]))
    await session.send("salam")
    assert session.current_language == "ar-MA"
    result = await session.send("speak English please")
    assert result.language == "en-GB"
    assert "British English" in (provider.calls[-1].system or "")


async def test_chat_history_is_windowed(config: AppConfig) -> None:
    provider = FakeProvider(name="fake", script=[[{"text": "ok"}]])
    session = ChatSession(config, ProviderRouter([provider]))
    session.context.max_turns = 2
    for _ in range(6):
        await session.send("salam")
    assert len(session.history) <= 4


async def test_reset_clears_conversation(config: AppConfig) -> None:
    session = ChatSession(config, ProviderRouter([FakeProvider(name="f")]))
    await session.send("salam")
    session.reset()
    assert session.history == [] and session.turns == 0


def test_snapshot_is_serialisable(config: AppConfig) -> None:
    session = ChatSession(config, ProviderRouter([FakeProvider(name="f")]))
    snapshot = session.snapshot()
    assert snapshot["providers"] == ["f"]
    assert snapshot["language"] == "ar-MA"


# ── L4: the audience ─────────────────────────────────────────────────
# The prompt is one of the two guards around the owner's memory (the caller is
# the other).  These tests pin the wording, because the failure mode is silent:
# a guest turn that quietly includes the owner's notes looks like a good answer.
def _owner_permissions():
    from atlas_core.identity import Permissions

    return Permissions.owner_of("badr")


def _stranger_permissions():
    from atlas_core.identity import Permissions

    return Permissions.stranger(score=0.31)


def test_a_stranger_prompt_forbids_personal_facts_in_both_languages(config: AppConfig) -> None:
    from atlas_core.identity import Audience

    persona = Persona(config)
    darija = persona.system_prompt(
        language="ar-MA", audience=Audience.from_permissions(_stranger_permissions())
    )
    assert "you have never heard it before" in darija
    assert "سمح ليا، هادشي خاص بصاحبي." in darija
    assert "do not hint that" in darija

    british = persona.system_prompt(
        language="en-GB", audience=Audience.from_permissions(_stranger_permissions())
    )
    assert "Sorry, that one is personal" in british
    assert "سمح ليا" not in british


def test_the_owner_prompt_is_unchanged_by_l4(config: AppConfig) -> None:
    from atlas_core.identity import Audience

    persona = Persona(config)
    with_audience = persona.system_prompt(
        language="ar-MA",
        audience=Audience.from_permissions(_owner_permissions()),
        memory="- kayt9an f atay",
    )
    without = persona.system_prompt(language="ar-MA", memory="- kayt9an f atay")
    assert with_audience == without, "an owner turn is the L3 turn, word for word"
    assert "kayt9an" in with_audience


def test_a_known_other_person_may_be_memory_but_only_their_own(config: AppConfig) -> None:
    from atlas_core.identity import Audience, Permissions

    persona = Persona(config)
    prompt = persona.system_prompt(
        language="ar-MA",
        audience=Audience.from_permissions(Permissions.guest_of("said")),
        memory="- (2026-09-25) kaykhdem f Casablanca",
    )
    assert "said" in prompt
    assert "Facts *they* tell you about themselves" in prompt
    assert "Casablanca" in prompt


async def test_the_audience_reaches_the_model_and_never_the_owner_notes(
    config: AppConfig,
) -> None:
    from atlas_core.identity import Audience

    provider = FakeProvider(name="gemini", script=[[{"text": "salam"}]])
    session = ChatSession(config, ProviderRouter([provider]))

    # The caller withholds the memory ("" here) — the prompt says why.
    await session.send(
        "chkoun nta",
        memory="",
        audience=Audience.from_permissions(_stranger_permissions()),
    )
    sent = provider.calls[-1]
    # The system prompt is its own field on `LlmRequest` — which is exactly why a
    # leak test has to look at both halves of the request.
    assert "personal" in (sent.system or "")
    assert "kayt9an" not in (sent.system or "")
    assert "kayt9an" not in " ".join(message.content for message in sent.messages)


async def test_an_owner_turn_still_receives_memory(config: AppConfig) -> None:
    from atlas_core.identity import Audience

    provider = FakeProvider(name="gemini", script=[[{"text": "salam"}]])
    session = ChatSession(config, ProviderRouter([provider]))
    await session.send(
        "chkoun nta",
        memory="- (2026-09-25) kayt9an f atay",
        audience=Audience.from_permissions(_owner_permissions()),
    )
    sent = provider.calls[-1]
    text = (sent.system or "") + "\n" + " ".join(message.content for message in sent.messages)
    assert "kayt9an" in text
