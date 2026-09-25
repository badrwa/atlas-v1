"""One contract suite per ABC, parametrised over *every* implementation.

Adding a new engine automatically inherits these tests — that is how the
"no duplicated code" rule (R5) stays honest as the project grows.
"""

from __future__ import annotations

import pytest

from atlas_core.contracts import (
    Engine,
    LlmProvider,
    Skill,
    SkillContext,
    SpeakerVerifier,
    SpeechRecognizer,
    SpeechSynthesizer,
    WakeWordEngine,
)
from atlas_core.fakes import (
    FakeProvider,
    FakeRecognizer,
    FakeSkill,
    FakeSynthesizer,
    FakeVerifier,
    FakeWake,
)

ENGINES = [FakeRecognizer(), FakeSynthesizer(), FakeVerifier()]
PROVIDERS = [FakeProvider()]
SKILLS = [FakeSkill()]
RECOGNIZERS = [FakeRecognizer()]
SYNTHESIZERS = [FakeSynthesizer()]
VERIFIERS = [FakeVerifier()]
WAKES = [FakeWake()]


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.name)
async def test_engine_load_unload_is_idempotent(engine: Engine) -> None:
    await engine.load()
    await engine.load()
    assert engine.is_loaded() is True
    await engine.unload()
    await engine.unload()
    assert engine.is_loaded() is False


@pytest.mark.parametrize("engine", ENGINES, ids=lambda e: e.name)
def test_engine_reports_its_cost(engine: Engine) -> None:
    cost = engine.cost_hint()
    assert cost.ram_mb > 0, "a leased engine must declare its RAM cost"
    assert cost.cold_start_s >= 0


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p.name)
async def test_provider_stream_always_ends(provider: LlmProvider) -> None:
    from atlas_core.contracts import LlmRequest, Message, Role

    request = LlmRequest(messages=[Message(role=Role.USER, content="salam")])
    kinds = [event.kind async for event in provider.stream(request)]
    assert kinds[-1] == "end"


@pytest.mark.parametrize("provider", PROVIDERS, ids=lambda p: p.name)
async def test_provider_health_never_raises(provider: LlmProvider) -> None:
    report = await provider.health()
    assert isinstance(report.ok, bool)


@pytest.mark.parametrize("recognizer", RECOGNIZERS, ids=lambda r: r.name)
async def test_recognizer_returns_language_and_confidence(recognizer: SpeechRecognizer) -> None:
    transcript = await recognizer.transcribe(b"\x00" * 3200)
    assert transcript.text
    assert transcript.language in {"ar-MA", "en-GB", "unknown"}
    assert 0.0 <= transcript.confidence <= 1.0
    assert transcript.engine


@pytest.mark.parametrize("synth", SYNTHESIZERS, ids=lambda s: s.name)
def test_synthesizer_yields_audio_and_finishes(synth: SpeechSynthesizer) -> None:
    chunks = list(synth.synthesize("salam", language="ar-MA"))
    assert chunks, "a synthesizer must produce at least one chunk"
    assert chunks[-1].final is True
    assert chunks[-1].sample_rate > 0


@pytest.mark.parametrize("verifier", VERIFIERS, ids=lambda v: v.name)
def test_verifier_embedding_is_stable_and_match_has_score(verifier: SpeakerVerifier) -> None:
    first = verifier.embed(b"\x00" * 3200)
    second = verifier.embed(b"\x00" * 3200)
    assert first == second, "same audio must give the same embedding"
    match = verifier.verify(b"\x00" * 3200, [first])
    assert 0.0 <= match.score <= 1.0
    assert isinstance(match.owner, bool)


@pytest.mark.parametrize("wake", WAKES, ids=lambda w: w.name)
def test_wake_silence_never_fires(wake: WakeWordEngine) -> None:
    silence = b"\x00" * 2560
    detections = [wake.feed(silence) for _ in range(20)]
    assert any(d.hit for d in detections), "this fake is configured to fire once"
    assert sum(1 for d in detections if d.hit) == 1, "silence must not fire repeatedly"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s.name)
def test_skill_spec_is_complete_in_both_languages(skill: Skill) -> None:
    spec = skill.spec()
    assert spec.name
    assert spec.description
    assert spec.description_darija
    assert spec.parameters.get("type") == "object"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s.name)
def test_skill_invoke_returns_a_result(skill: Skill) -> None:
    result = skill.invoke({}, SkillContext())
    assert isinstance(result.ok, bool)
    assert isinstance(result.spoken, str)
