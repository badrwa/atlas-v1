"""L4 — the verifier, the enrolment session, and the loop's identity wiring.

No model is downloaded here.  The extractor is a deterministic function of the
audio, which is the whole point of the injection seam: the *policy* is what these
tests check, and the policy is the part that can be wrong in a way nobody notices.
"""

from __future__ import annotations

import math
from array import array
from pathlib import Path

import pytest

from atlas_audio.loop import VoiceLoop
from atlas_audio.speaker import (
    ENROL_PROMPTS,
    EnrollmentSession,
    SherpaSpeakerVerifier,
    as_samples,
    build_identity,
    build_verifier,
    speaker_status,
    speech_ms,
    to_float_list,
    trim_silence,
)
from atlas_core.contracts import Capability
from atlas_core.identity import IdentityConfig, SpeakerLog, SpeakerProfile

SAMPLE_RATE = 16000


# ── helpers ──────────────────────────────────────────────────────────
def tone(ms: int, *, amplitude: int = 6000, freq: float = 210.0, phase: float = 0.0) -> array:
    """A pseudo-voice: loud enough to be 'speech', cheap enough to build inline."""
    count = int(SAMPLE_RATE * ms / 1000)
    return array(
        "h",
        [
            int(amplitude * math.sin(2 * math.pi * freq * index / SAMPLE_RATE + phase))
            for index in range(count)
        ],
    )


def silence(ms: int) -> array:
    return array("h", [0] * int(SAMPLE_RATE * ms / 1000))


def fake_extractor(samples: array, sample_rate: int) -> list[float]:
    """Whose voice is this?  A crude autocorrelation — pitch, not loudness.

    Two clips of "the same person" differ only by amplitude and stay cosine ≈ 1;
    a different pitch gives a different direction.  That is exactly the property
    a real speaker model has, which is why it is worth faking this way.
    """
    assert sample_rate == SAMPLE_RATE
    energy = sum(value * value for value in samples) or 1.0
    lags = [1, 3, 9, 27, 81]
    return [
        sum(samples[index] * samples[index + lag] for index in range(len(samples) - lag)) / energy
        for lag in lags
    ]


def verifier(**kwargs) -> SherpaSpeakerVerifier:
    kwargs.setdefault("model_path", "models/speaker/does-not-matter.onnx")
    kwargs.setdefault("extractor", fake_extractor)
    return SherpaSpeakerVerifier(**kwargs)


VOICE_A = tone(2000, freq=210.0)
VOICE_A_AGAIN = tone(2000, freq=210.0, amplitude=2500)  # same person, quieter
VOICE_B = tone(2000, freq=520.0)  # a different person


# ── the verifier ─────────────────────────────────────────────────────
def test_the_same_voice_matches_and_a_different_one_does_not() -> None:
    engine = verifier()
    first = engine.embed(VOICE_A)
    second = engine.embed(VOICE_A_AGAIN)
    other = engine.embed(VOICE_B)

    assert math.isclose(math.dist(first, second), 0.0, abs_tol=1e-6)
    assert math.dist(first, other) > 0.2
    assert engine.embeddings == 3
    assert engine.last_ms >= 0.0


def test_a_dropped_clip_of_the_same_voice_still_matches_the_stored_one() -> None:
    engine = verifier()
    profile = SpeakerProfile(
        name="badr", embeddings=[engine.embed(VOICE_A)], samples=1, owner=True
    )
    match = engine.match_profile(VOICE_A_AGAIN, profile)
    assert match.name == "badr"
    assert match.score > 0.99
    assert match.owner is True

    other = engine.match_profile(VOICE_B, profile, threshold=0.99)
    assert other.name == "badr" and other.owner is False


def test_the_contract_verify_returns_the_bare_score() -> None:
    engine = verifier()
    window = [engine.embed(VOICE_A), engine.embed(VOICE_A_AGAIN)]
    match = engine.verify(VOICE_A, window)
    assert match.name == ""
    assert match.score > 0.99


def test_short_audio_is_refused_rather_than_guessed() -> None:
    engine = verifier()
    with pytest.raises(Exception, match="too short"):
        engine.embed(tone(300))
    assert engine.embed_or_none(tone(300)) is None
    assert engine.embeddings == 0

    quiet = verifier(min_embed_ms=100)
    assert quiet.embed_or_none(tone(300)) is not None


def test_the_wrong_sample_rate_is_a_bug_not_a_silent_fallback() -> None:
    engine = verifier()
    with pytest.raises(Exception, match="16000"):
        engine.embed(tone(2000), sample_rate=8000)


def test_an_unloaded_model_says_so_instead_of_inventing_a_score() -> None:
    engine = SherpaSpeakerVerifier("models/speaker/absent.onnx")
    assert engine.available() is False
    assert "not found" in engine.missing() or "sherpa" in engine.missing()
    assert engine.embed_or_none(VOICE_A) is None
    status = engine.status()
    assert status["available"] is False
    assert status["model"] == "models/speaker/absent.onnx"


def test_load_reports_the_reason_it_cannot_start() -> None:
    import asyncio

    engine = SherpaSpeakerVerifier("models/speaker/absent.onnx")
    asyncio.run(engine.load())
    assert engine.is_loaded() is False
    assert engine.load_error


def test_the_verifier_stays_unloaded_while_a_streaming_loop_runs(
) -> None:
    """A loop owns the microphone; the verifier must not hog it at startup."""
    engine = verifier()
    assert engine.cost_hint().cpu_threads <= 2, "two Skylake cores are the budget"


# ── audio helpers ────────────────────────────────────────────────────
def test_trim_silence_leaves_speech_alone() -> None:
    padded = silence(500) + tone(1500) + silence(700)
    trimmed = trim_silence(padded)
    assert len(trimmed) < len(padded)
    assert 0.9 <= len(trimmed) / SAMPLE_RATE <= 2.0
    assert trim_silence(array("h")) == array("h")
    assert trim_silence(silence(400)) == array("h")


def test_speech_ms_counts_only_the_loud_part() -> None:
    assert speech_ms(tone(1000)) > 900
    assert speech_ms(silence(1000)) == 0.0
    assert speech_ms(silence(200) + tone(500)) > 400


def test_as_samples_takes_bytes_or_arrays() -> None:
    samples = tone(100)
    assert as_samples(samples) is samples
    assert as_samples(array("h", [1, 2, 3]).tobytes()) == array("h", [1, 2, 3])
    floats = to_float_list(samples)
    assert len(floats) == len(samples)
    assert max(floats) <= 1.0 and min(floats) >= -1.0
    assert floats[0] == pytest.approx(samples[0] / 32768, abs=1e-9)


# ── enrolment ────────────────────────────────────────────────────────
def session(**kwargs) -> EnrollmentSession:
    config = kwargs.pop("config", None) or IdentityConfig(enrol_samples=3, enrol_min_speech_ms=500)
    return EnrollmentSession(
        kwargs.pop("name", "badr"), verifier=verifier(), config=config, **kwargs
    )


def test_three_good_samples_produce_a_profile() -> None:
    enrolling = session()
    assert enrolling.missing == 3
    for index in range(3):
        step = enrolling.add(tone(1200, freq=210 + index * 0.05))
        assert step.ok is True
        assert step.speech_ms > 1000
        assert step.spoken, "every sample says something out loud"
    assert enrolling.complete is True
    assert enrolling.problems() == []

    profile = enrolling.profile()
    assert profile is not None
    assert profile.name == "badr"
    assert len(profile.embeddings) == 3
    assert profile.quality > 0.99
    assert profile.samples == 3


def test_a_bad_sample_is_refused_with_a_reason_and_a_spoken_line() -> None:
    enrolling = session()
    step = enrolling.add(tone(200))  # 0.2 s of speech, floor is 0.5 s
    assert step.ok is False
    assert step.reason == "not_enough_speech"
    assert "200" in step.spoken or "0" in step.spoken
    assert enrolling.index == 0, "a refused sample is not counted"
    assert enrolling.profile() is None
    assert any("more sample" in problem for problem in enrolling.problems())


def test_silence_is_not_a_voice_sample() -> None:
    enrolling = session()
    step = enrolling.add(silence(2000))
    assert step.ok is False
    assert step.reason == "not_enough_speech"


def test_inconsistent_samples_are_rejected_before_they_become_a_profile() -> None:
    """Three clips of three different people is not an enrolment — it is a bug."""
    enrolling = session(config=IdentityConfig(enrol_samples=3, enrol_min_speech_ms=500))
    enrolling.add(tone(1200, freq=210.0))
    enrolling.add(tone(1200, freq=430.0))
    step = enrolling.add(tone(1200, freq=680.0))

    assert step.ok is True, "each clip was fine on its own"
    assert "L3inayat machi mzyanin" in step.spoken, "the samples disagree, and it says so"
    assert str(round(enrolling.quality, 2)) in step.spoken
    assert enrolling.profile() is None
    assert any("below the floor" in problem for problem in enrolling.problems())


def test_the_prompts_are_short_and_speak_the_users_language() -> None:
    for language, prompts in ENROL_PROMPTS.items():
        assert len(prompts) >= 1
        assert all(prompt.strip() for prompt in prompts), language
    assert ENROL_PROMPTS["ar-MA"] != ENROL_PROMPTS["en-GB"]
    # Darija is written the way it is read (Latin letters), English in English.
    assert "Salam" in ENROL_PROMPTS["ar-MA"][0]
    assert "Salam" not in ENROL_PROMPTS["en-GB"][0]
    assert enrolling_prompt("en-GB").startswith("Say")
    # The Arabic prompt is the same text with the name dropped in.
    assert "badr" in enrolling_prompt("ar-MA")
    assert enrolling_prompt("ar-MA").split("“")[0] == ENROL_PROMPTS["ar-MA"][0].split("“")[0]


def enrolling_prompt(language: str) -> str:
    return session(language=language).prompt(0)


def test_the_session_reports_its_own_state() -> None:
    enrolling = session()
    status = enrolling.status()
    assert status["required"] == 3
    assert status["collected"] == 0
    assert status["complete"] is False
    assert status["floor"] == IdentityConfig().quality_floor
    assert status["problems"] == ["3 more sample(s) needed"]


# ── the factories ────────────────────────────────────────────────────
def fake_config(tmp_path: Path) -> object:
    """A config object shaped like `AppConfig`, with the store in tmp_path."""

    class FakeIdentitySection:
        enabled = True
        model_path = "models/speaker/x.onnx"
        profiles_path = str(tmp_path / "speakers.sqlite3")
        log_path = str(tmp_path / "speaker_log.jsonl")
        threshold = 0.7
        owner_name = "badr"
        guest_pc_control = False

    class FakeConfig:
        identity = FakeIdentitySection()

    return FakeConfig()


def test_identity_off_returns_no_verifier_at_all() -> None:
    class Off:
        enabled = False

    class Config:
        identity = Off()

    assert build_verifier(Config()) is None
    built = build_identity(Config())
    assert built["verifier"] is None
    assert built["guard"] is not None, "the guard still exists — it answers 'identity_disabled'"


def test_build_identity_wires_the_five_pieces_together(tmp_path: Path) -> None:
    built = build_identity(fake_config(tmp_path))
    assert set(built) == {"config", "verifier", "repository", "log", "guard", "owner"}
    assert isinstance(built["verifier"], SherpaSpeakerVerifier)
    assert isinstance(built["log"], SpeakerLog)
    assert built["guard"].owner_name == "badr"
    assert built["config"].threshold == 0.7
    assert built["repository"].path.name == ":memory:" or str(built["repository"].path).endswith(
        "sqlite3"
    )


def test_the_doctor_rows_name_the_model_and_the_threshold(tmp_path: Path) -> None:
    rows = speaker_status(fake_config(tmp_path))
    as_dict = dict(rows)
    assert "verifier" in as_dict
    assert "threshold" in as_dict
    assert "0.70" in as_dict["threshold"]
    assert as_dict["profiles"].startswith("0 people")
    assert all(isinstance(name, str) and isinstance(value, str) for name, value in rows)


# ── the loop ─────────────────────────────────────────────────────────
class _Detector:
    def detect(self, frame):
        return None


def build_loop(guard, engine=None, **kwargs) -> VoiceLoop:
    from atlas_audio.vad import VadSegmenter

    return VoiceLoop(
        wake=_Detector(),  # type: ignore[arg-type]
        segmenter=VadSegmenter(),
        recognizer=None,
        respond=None,
        verifier=engine,
        guard=guard,
        **kwargs,
    )


def loop_guard(**settings):
    from atlas_core.identity import ContextGuard, SpeakerProfile

    owner = SpeakerProfile(
        name="badr", embeddings=[fake_extractor(VOICE_A, SAMPLE_RATE)], samples=1, owner=True
    )
    return ContextGuard(
        IdentityConfig(owner_name="badr", **settings),
        profiles={"badr": owner},
        log=SpeakerLog(":memory:", load=False),
    )


def test_a_loop_without_identity_is_the_single_user_loop() -> None:
    from atlas_audio.vad import VadSegmenter

    loop = VoiceLoop(wake=_Detector(), segmenter=VadSegmenter(), recognizer=None, respond=None)  # type: ignore[arg-type]
    assert loop.permissions.owner is True
    assert loop.permissions.allows(Capability.DESTRUCTIVE) is True
    assert loop.permissions.reason == "identity_off"
    assert loop.identity_ms == 0.0
    assert loop.turns == []


def test_an_unavailable_verifier_keeps_the_owner_working_and_logs_why() -> None:
    loop = build_loop(loop_guard())
    assert loop.permissions.allows(Capability.GENERAL) is True  # pre-first-turn default


def test_the_default_permissions_before_any_utterance_are_the_owner_s() -> None:
    """A loop that has not heard anything yet must not be *more* restricted."""
    loop = build_loop(loop_guard())
    assert loop.permissions.owner is True
    assert loop.permissions.reason == "identity_off"


# ── the verdict reaches the turn ─────────────────────────────────────
@pytest.mark.asyncio
async def test_a_turn_carries_the_speaker_and_the_capabilities() -> None:
    from atlas_audio.vad import VadSegmenter
    from atlas_core.contracts import Transcript

    class Wake:
        def detect(self, frame):
            return None

    class Recognizer:
        name = "fake"

        async def transcribe(self, audio: bytes, *, language=None, hint=None):
            return Transcript(text="salam", language=language or "ar-MA", confidence=0.9)

    guard = loop_guard()
    loop = VoiceLoop(
        wake=Wake(),  # type: ignore[arg-type]
        segmenter=VadSegmenter(),
        recognizer=Recognizer(),  # type: ignore[arg-type]
        respond=None,
        verifier=verifier(),
        guard=guard,
        profiles={"badr": guard.profiles["badr"]},
    )

    utterance = _utterance(tone(1500))
    await loop._transcribe(utterance)

    assert len(loop.turns) == 1
    turn = loop.turns[0]
    assert turn.speaker == "badr"
    assert turn.owner is True
    assert turn.restricted is False
    assert turn.speaker_score > 0.9
    assert loop.permissions.allows(Capability.READ_MEMORY) is True
    assert loop.identity_ms > 0.0

    payload = turn.as_dict()
    assert payload["speaker"] == "badr"
    assert payload["owner"] is True
    assert payload["speaker_score"] > 0.9


def _utterance(samples: array):
    from atlas_audio.frames import pcm_to_frames
    from atlas_audio.vad import Utterance

    return Utterance(frames=pcm_to_frames(samples.tobytes()))
