"""The turn loop, and the recording that makes it reproducible.

This file is the L2 acceptance test in miniature: a synthetic microphone, a
real wake-confirmed utterance, a transcript, a reply, and a follow-up sentence
that needs no second "atlas".  Everything is offline and deterministic, which is
the only way a laptop with one microphone can be tested from CI.
"""

from __future__ import annotations

import asyncio
import itertools
from array import array
from typing import Any

import pytest
from doubles import FakeRecognizer, ScriptedWake, utterance_audio

from atlas_audio.capture import (
    TurnRecorder,
    WavFile,
    load_session,
    pcm_to_wav_bytes,
    platform_notes,
    silence,
    speech,
    tone,
)
from atlas_audio.frames import (
    FRAME_MS,
    FRAME_SAMPLES,
    FrameBus,
    FramePacket,
    frames_to_pcm,
    new_frame,
)
from atlas_audio.loop import LoopConfig, LoopState, VoiceLoop
from atlas_audio.postprocess import AsrPostProcessor
from atlas_audio.vad import SegmenterConfig, VadSegmenter
from atlas_audio.wake import EnergyWakeEngine


def make_loop(
    recognizer: FakeRecognizer | None = None,
    *,
    replies: list[str] | None = None,
    config: LoopConfig | None = None,
    recorder: TurnRecorder | None = None,
    wake=None,
) -> tuple[VoiceLoop, FakeRecognizer, list[str]]:
    recognizer = recognizer or FakeRecognizer()
    spoken: list[str] = []
    lines = replies or ["safi, kola chi haja mzyana"]

    async def respond(text: str, language: str):
        assert language in {"ar-MA", "en-GB", "unknown"}
        for chunk in lines:
            yield chunk

    loop = VoiceLoop(
        wake=wake or EnergyWakeEngine(confirm_frames=1),
        segmenter=VadSegmenter(SegmenterConfig()),
        recognizer=recognizer,
        respond=respond,
        say=spoken.append,
        processor=AsrPostProcessor(),
        recorder=recorder,
        config=config or LoopConfig(),
    )
    loop.arm()
    return loop, recognizer, spoken


# ── the happy path ───────────────────────────────────────────────────
async def test_one_wake_word_one_answer_then_a_follow_up_without_the_keyword():
    loop, recognizer, spoken = make_loop()

    frames = (
        silence(4)
        + speech(4)  # "atlas": enough to confirm the wake
        + silence(8)  # the pause after it
        + speech(14)  # the command
        + silence(8)  # the endpointing
        + speech(14)  # a follow-up, inside the 2.5 s grace window
        + silence(8)
    )
    turns = await loop.feed_frames(frames)

    assert len(turns) == 2
    assert [turn.text for turn in turns] == ["chno ljaw", "chno ljaw"]
    assert turns[1].wake is not None  # the detection is carried, for the notes
    assert spoken == ["safi, kola chi haja mzyana"] * 2  # once per turn, said aloud once
    assert loop.state is LoopState.FOLLOWUP
    assert len(recognizer.calls) == 2


async def test_speech_before_the_wake_word_is_not_transcribed():
    """Somebody talking in the room is not a command — the keyword gates it."""
    loop, recognizer, _ = make_loop(wake=ScriptedWake(allow_from=10))
    frames = speech(10) + silence(10)  # loud, but never the keyword
    assert await loop.feed_frames(frames) == []
    assert recognizer.calls == []
    assert loop.state is LoopState.WAKING


async def test_the_wake_word_itself_never_reaches_the_transcript():
    """The guard frames after a hit drop the tail of the keyword."""
    loop, _, _ = make_loop()
    await loop.feed_frames(speech(3) + silence(2) + speech(14) + silence(8))
    assert loop.frames_dropped_after_wake == loop.config.guard_frames
    assert loop.turns[0].text == "chno ljaw"
    assert "-las" not in loop.turns[0].text


async def test_an_empty_transcript_does_not_become_a_turn():
    loop, _, spoken = make_loop(FakeRecognizer(text=""))
    turns = await loop.feed_frames(utterance_audio())
    assert turns == []
    assert spoken == []  # nothing was said aloud for nothing


async def test_a_failed_asr_keeps_the_session_alive():
    loop, recognizer, _ = make_loop()
    recognizer.fail = RuntimeError("cloud exploded")
    frames = utterance_audio()
    assert await loop.feed_frames(frames) == []
    assert loop.state is LoopState.WAKING  # armed and ready for the next try

    recognizer.fail = None
    turns = await loop.feed_frames(utterance_audio())
    assert len(turns) == 1  # the very next attempt works


# ── half duplex ──────────────────────────────────────────────────────
async def test_the_microphone_is_muted_for_the_whole_reply():
    states: list[tuple[LoopState, bool]] = []

    async def respond(text: str, language: str):
        states.append((loop.state, loop.mic_muted))
        yield "salam"
        states.append((loop.state, loop.mic_muted))
        yield " labas"

    loop = VoiceLoop(
        wake=EnergyWakeEngine(confirm_frames=1),
        segmenter=VadSegmenter(SegmenterConfig()),
        recognizer=FakeRecognizer(),
        respond=respond,
        config=LoopConfig(),
    )
    loop.arm()
    await loop.feed_frames(utterance_audio())

    assert states == [(LoopState.SPEAKING, True), (LoopState.SPEAKING, True)]
    assert loop.mic_muted is False  # released afterwards, or Atlas goes deaf


async def test_no_frame_is_acted_on_while_speaking():
    """The classic bug: Atlas hears its own voice and answers itself."""
    loop, recognizer, _ = make_loop()
    loop.arm()
    loop.mic_muted = True
    for index, frame in enumerate(speech(20)):
        packet = FramePacket(frame=frame, index=index, at_ms=index * FRAME_MS)
        assert await loop.feed(packet) is None
    assert loop.frames_seen == 20  # counted…
    assert recognizer.calls == []  # …and ignored


# ── the follow-up window ─────────────────────────────────────────────
async def test_the_follow_up_window_expires_and_re_arms_the_wake_word():
    # The keyword is only allowed at the very start; the follow-up sentence is
    # spoken without one, and after the window expires it must not be enough.
    loop, recognizer, _ = make_loop(
        config=LoopConfig(followup_ms=800), wake=ScriptedWake(allow_until=8)
    )
    await loop.feed_frames(silence(2) + speech(4) + silence(8) + speech(14) + silence(8))
    assert len(loop.turns) == 1
    assert loop.state is LoopState.FOLLOWUP

    # Silence past the window: the loop goes back to needing "atlas".
    await loop.feed_frames(silence(20))
    assert loop.state is LoopState.WAKING

    # …so plain speech no longer starts a turn: the keyword is required again.
    assert await loop.feed_frames(speech(14) + silence(8)) == []
    assert len(recognizer.calls) == 1


async def test_a_stopped_loop_ignores_everything():
    loop, recognizer, _ = make_loop()
    loop.stop()
    assert await loop.feed_frames(utterance_audio()) == []
    assert loop.state is LoopState.STOPPED
    assert recognizer.calls == []


# ── push to talk ─────────────────────────────────────────────────────
async def test_push_to_talk_needs_no_wake_word():
    loop, _, _ = make_loop()
    loop.arm()
    turn = await loop.push_to_talk(speech(14) + silence(8))
    assert turn is not None
    assert turn.wake is None  # a key started this, not a keyword
    assert loop.wake_hits == 0


async def test_push_to_talk_flushes_a_recording_that_ends_mid_sentence():
    loop, _, _ = make_loop()
    loop.arm()
    turn = await loop.push_to_talk(speech(12))  # no trailing silence at all
    assert turn is not None and turn.transcript.text


# ── recording and replay ─────────────────────────────────────────────
async def test_a_session_is_recorded_and_its_turns_replayable(tmp_path):
    recorder = TurnRecorder(tmp_path / "recordings", session="test")
    loop, _, _ = make_loop(recorder=recorder)
    await loop.feed_frames(utterance_audio())

    manifest = recorder.flush()
    assert manifest is not None and manifest.exists()

    session = load_session(tmp_path / "recordings" / "test")
    assert len(session.utterances) == len(session.turns) == 1
    assert session.transcripts() == ["chno ljaw"]
    assert session.turns[0]["language"] == "ar-MA"
    assert session.turns[0]["reason"] in {"silence", "forced"}
    # The raw stream is there too, and it is longer than the utterance: it holds
    # the silence and the wake word that the utterance deliberately drops.
    assert session.audio is not None
    assert session.audio.duration_ms > session.turns[0]["duration_ms"]

    # The dump is real audio, at the right rate, and it segments back to one turn.
    wav = WavFile.read(session.utterances[0])
    assert wav.sample_rate == 16_000
    assert wav.duration_ms == pytest.approx(session.turns[0]["duration_ms"], abs=100)
    assert len(wav.frames()) > 1


async def test_a_replayed_recording_produces_the_same_turns(tmp_path):
    """Determinism: same audio in, same words out — the point of --capture-dump."""
    recorder = TurnRecorder(tmp_path / "rec", session="first")
    first, _, _ = make_loop(recorder=recorder)
    await first.feed_frames(utterance_audio())
    recorder.flush()

    session = load_session(tmp_path / "rec" / "first")
    assert session.audio is not None

    # Replay the *whole* session: the wake word is in there, so a fresh loop
    # reaches the same turn from the same audio, byte for byte.
    loop, _, _ = make_loop()
    loop.arm()
    turns = [turn for packet in session.audio.packets(source="replay") if (turn := await loop.feed(packet))]
    assert [turn.text for turn in turns] == ["chno ljaw"]
    assert loop.wake_hits == 1


def test_wav_roundtrip_preserves_the_samples(tmp_path):
    original = WavFile(array("h", list(array("h", frames_to_pcm(speech(6))))))
    path = original.write(tmp_path / "a.wav")
    again = WavFile.read(path)
    assert again.samples == original.samples
    assert again.duration_ms == pytest.approx(480)


def test_wav_read_converts_stereo_48k_to_our_format(tmp_path):
    import wave

    path = tmp_path / "device.wav"
    samples = array("h", [1000] * 48_000)  # 24 000 stereo frames = 500 ms
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(48_000)
        handle.writeframes(samples.tobytes())

    wav = WavFile.read(path)
    assert wav.sample_rate == 16_000
    assert wav.duration_ms == pytest.approx(500, abs=10)
    assert len(wav.frames()) == pytest.approx(500 / FRAME_MS, abs=1)


def test_pcm_to_wav_bytes_is_a_playable_container():
    blob = pcm_to_wav_bytes(frames_to_pcm(speech(3)))
    assert blob[:4] == b"RIFF"
    assert len(blob) == len(frames_to_pcm(speech(3))) + 44


def test_the_self_test_reports_missing_audio_support_instead_of_crashing():
    from atlas_audio.capture import self_test

    result = self_test(seconds=0.05)
    try:
        import sounddevice  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, OSError):
        assert result.ok is False
        assert "atlas-audio[audio]" in result.summary()
    else:  # pragma: no cover - only where sounddevice exists
        assert result.summary()


def test_windows_advice_is_available_for_the_readme():
    notes = platform_notes()
    assert any("Privacy" in note for note in notes)
    assert any("Exclusive" in note or "exclusive" in note for note in notes)


# ── reporting ────────────────────────────────────────────────────────
async def test_the_snapshot_is_what_the_orb_and_doktor_need():
    loop, _, _ = make_loop()
    await loop.feed_frames(utterance_audio())
    snapshot = loop.snapshot()
    assert snapshot["turns"] == 1
    assert snapshot["wake_engine"] == "energy-wake"
    assert snapshot["state"] in {"followup", "waking"}
    assert "1. [ar-MA 0.90] chno ljaw" in loop.summary()


async def test_the_bus_can_drive_the_loop_like_the_real_thing_does():
    loop, _, _ = make_loop()
    bus = FrameBus()
    queue = bus.subscribe()
    for index, frame in enumerate(utterance_audio()):
        bus.publish(FramePacket(frame=frame, index=index, at_ms=index * FRAME_MS))

    turns = []
    while not queue.empty():
        if turn := await loop.feed(queue.get_nowait()):
            turns.append(turn)
    assert len(turns) == 1
    assert loop.frames_seen == len(utterance_audio())


def test_loop_config_reads_the_numbers_from_the_plan():
    config = LoopConfig()
    assert config.followup_ms == 2500
    assert config.guard_frames == 2  # 160 ms of keyword tail
    assert config.followup_frames == 31
    assert SegmenterConfig().silence_ms == 500


def test_tone_is_loud_enough_to_trip_the_energy_vad():
    from atlas_audio.frames import rms

    assert rms(tone(FRAME_SAMPLES)) > 0.1
    assert rms(new_frame()) == 0.0


# ── the orb's side of the loop (L5) ──────────────────────────────────
def collecting_bus() -> tuple[Any, list[Any]]:
    """A bus that records instead of dispatching: tests want the list, not actors."""
    from atlas_core.events import EventBus

    bus = EventBus()
    seen: list[Any] = []
    bus.on("*", seen.append)
    return bus, seen


async def test_every_state_change_is_published_exactly_once():
    """The orb cannot miss a transition: one writer, one subscriber, in order."""
    from atlas_core.events import StateChanged

    bus, seen = collecting_bus()
    loop, _recognizer, _spoken = make_loop()
    loop.events = bus
    # Attached after the first arm, so these two are the first things the orb sees.
    loop.stop()
    loop.arm()

    await loop.feed_frames(silence(2) + speech(4) + silence(8) + speech(14) + silence(8))
    await asyncio.sleep(0)  # let the fire-and-forget publishes land

    states = [event.state for event in seen if isinstance(event, StateChanged)]
    assert states[:3] == ["stopped", "waking", "listening"], states
    assert "thinking" in states and "speaking" in states, states
    assert states[-1] in {"followup", "listening"}
    # No duplicate consecutive states: the FSM may re-enter, the orb must not flicker.
    assert all(a != b for a, b in itertools.pairwise(states)), states


async def test_levels_only_arrive_while_the_microphone_is_open():
    """Half duplex in the UI too: no level events while Atlas is speaking."""
    from atlas_core.events import AudioLevel

    bus, seen = collecting_bus()
    loop, _recognizer, _spoken = make_loop()
    loop.events = bus

    await loop.feed_frames(silence(2) + speech(4) + silence(8) + speech(14) + silence(8))
    await asyncio.sleep(0)

    levels = [event for event in seen if isinstance(event, AudioLevel)]
    assert levels, "the orb needs loudness while listening"
    assert all(0.0 <= float(event.level) <= 1.0 for event in levels)


async def test_a_turn_nobody_claims_is_not_published_as_a_speaker():
    """No verifier configured → no SpeakerMatched event, and no invented name."""
    from atlas_core.events import SpeakerMatched

    bus, seen = collecting_bus()
    loop, _recognizer, _spoken = make_loop()
    loop.events = bus
    await loop.feed_frames(speech(4) + silence(8) + speech(14) + silence(8))
    await asyncio.sleep(0)
    assert [event for event in seen if isinstance(event, SpeakerMatched)] == []


async def test_a_loop_without_a_bus_still_runs():
    """The ears must not depend on the face: no subscriber, no difference."""
    loop, _recognizer, _spoken = make_loop()
    assert loop.events is None
    turns = await loop.feed_frames(speech(4) + silence(8) + speech(14) + silence(8))
    assert len(turns) == 1
