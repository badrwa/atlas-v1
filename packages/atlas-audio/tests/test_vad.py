"""Endpointing: when does a sentence start, and when is it over?

The numbers under test are the ones the level spec fixes: 500 ms of silence to
close (350 ms snappy), 300 ms minimum, 12 s maximum, 1.5 s of pre-roll.  All
fixtures are synthetic frames, so these tests run on a machine with no mic.
"""

from __future__ import annotations

import pytest

from atlas_audio.capture import silence, speech
from atlas_audio.frames import FRAME_MS, FRAME_SAMPLES, FramePacket, new_frame, rms
from atlas_audio.vad import (
    EnergyVad,
    SegmenterConfig,
    SileroVad,
    Utterance,
    VadSegmenter,
    utterances_from_pcm,
)


def packets(frames: list) -> list[FramePacket]:
    return [
        FramePacket(frame=frame, index=index, at_ms=index * FRAME_MS, source="test")
        for index, frame in enumerate(frames)
    ]


def run(segmenter: VadSegmenter, frames: list) -> list[Utterance]:
    found = []
    for packet in packets(frames):
        if utterance := segmenter.feed(packet):
            found.append(utterance)
    return found


# ── the energy detector ──────────────────────────────────────────────
def test_energy_vad_is_a_threshold_not_a_mood():
    vad = EnergyVad(0.012)
    assert not vad.is_speech(new_frame())
    assert vad.is_speech(speech(1)[0])


def test_energy_vad_hangover_is_capped():
    """An uncapped hangover swallowed the silence run and stalled the segmenter."""
    vad = EnergyVad(0.012, hangover=2)
    for _ in range(10):
        assert vad.is_speech(speech(1)[0])
    quiet = new_frame()
    assert [vad.is_speech(quiet) for _ in range(4)] == [True, True, False, False]


def test_energy_vad_rejects_a_negative_hangover():
    with pytest.raises(ValueError):
        EnergyVad(hangover=-1)


def test_loudness_is_scale_free():
    """RMS of a quiet frame and a loud one differ by the amplitude ratio."""
    quiet = speech(1, amplitude=0.1)[0]
    loud = speech(1, amplitude=0.4)[0]
    assert rms(loud) / rms(quiet) == pytest.approx(4.0, rel=0.25)


# ── start / end ──────────────────────────────────────────────────────
def test_utterance_opens_on_speech_and_closes_on_half_a_second_of_silence():
    segmenter = VadSegmenter(SegmenterConfig())
    found = run(segmenter, silence(2) + speech(20) + silence(10))
    assert len(found) == 1
    utterance = found[0]
    assert utterance.reason == "silence"
    assert segmenter.active is False  # ready for the next sentence
    # 20 speech frames + 6 silence frames counted, the trailing silence stripped
    assert utterance.duration_ms == pytest.approx(20 * FRAME_MS)
    assert utterance.peak > 0.05


def test_a_long_quiet_lead_in_is_trimmed_to_the_onset():
    """The pre-roll is a safety net, not a licence to upload room tone."""
    segmenter = VadSegmenter(SegmenterConfig(pre_roll_ms=400))
    found = run(segmenter, silence(10) + speech(10) + silence(10))
    utterance = found[0]
    assert utterance.duration_ms == pytest.approx(10 * FRAME_MS)  # exactly the speech
    assert utterance.pre_roll_frames == 1  # the frame that opened it
    assert utterance.speech_frames == 10


def test_pre_roll_keeps_the_wake_word_tail_right_before_the_command():
    """Continuous loud audio: nothing is trimmed, because none of it is quiet."""
    segmenter = VadSegmenter(SegmenterConfig(pre_roll_ms=400, energy_threshold=0.2))
    # Constant frames, so the two thresholds cannot be confused: 6000/32768 ≈ 0.18
    # is audible (above `is_silence`) but under the 0.2 speech threshold.
    quiet = new_frame([6000] * FRAME_SAMPLES)
    loud = new_frame([20000] * FRAME_SAMPLES)
    found = run(segmenter, [quiet] * 5 + [loud] * 10 + silence(8))
    utterance = found[0]
    # The 400 ms ring holds 5 frames, and the fifth is the loud frame that opened
    # the utterance — so four frames of run-up survive, and nothing else is lost.
    assert utterance.pre_roll_frames == 5
    assert utterance.speech_frames == 10
    assert utterance.duration_ms == pytest.approx(14 * FRAME_MS)


def test_a_cough_is_not_a_command():
    segmenter = VadSegmenter(SegmenterConfig())
    found = run(segmenter, speech(2) + silence(10))
    assert found == []
    assert segmenter.active is False


def test_a_monologue_is_cut_and_the_rest_keeps_coming():
    """12 s cap: a long answer becomes two utterances instead of one huge upload."""
    config = SegmenterConfig(max_utterance_ms=1600, pre_roll_ms=0)
    segmenter = VadSegmenter(config)
    found = run(segmenter, speech(40))
    assert [utterance.reason for utterance in found] == ["max_length", "max_length"]
    assert all(utterance.duration_ms == pytest.approx(1600) for utterance in found)


def test_the_pause_threshold_is_what_makes_it_feel_fast():
    frames = speech(14) + silence(5)
    normal = run(VadSegmenter(SegmenterConfig()), frames)
    snappy = run(VadSegmenter(SegmenterConfig.snappy_config()), frames)
    assert normal == []  # 400 ms of silence is not an ending yet
    assert len(snappy) == 1  # 350 ms is
    assert SegmenterConfig.snappy_config().snappy is True


def test_flush_closes_a_recording_that_ends_mid_sentence():
    segmenter = VadSegmenter(SegmenterConfig())
    run(segmenter, speech(12))
    tail = segmenter.flush()
    assert tail is not None and tail.reason == "forced"
    assert segmenter.flush() is None  # idempotent


def test_reset_throws_away_a_half_heard_utterance():
    segmenter = VadSegmenter(SegmenterConfig())
    run(segmenter, speech(8))
    segmenter.reset()
    assert not segmenter.active and segmenter.flush() is None


# ── injected detectors ───────────────────────────────────────────────
def test_any_detector_can_drive_the_segmenter():
    """Tests, and the Silero path, both arrive through this seam."""
    calls: list[int] = []

    def detector(frame) -> bool:
        calls.append(1)
        return rms(frame) > 0.25  # deliberately stricter than the default

    segmenter = VadSegmenter(SegmenterConfig(), detector=detector)
    found = run(segmenter, speech(10, amplitude=0.4) + silence(8))
    assert calls and len(found) == 1


def test_silero_without_the_model_degrades_and_says_why():
    vad = SileroVad()
    if SileroVad.available():
        pytest.skip("sherpa-onnx is installed in this environment")
    assert vad.available() is False
    assert vad.load() is False
    assert "sherpa-onnx" in vad.load_error

    segmenter = VadSegmenter(SegmenterConfig())
    assert segmenter.use_silero(vad) is False
    # …and the segmenter still works, on the energy backend
    assert len(run(segmenter, speech(10) + silence(8))) == 1


# ── whole-recording helper ───────────────────────────────────────────
def test_utterances_from_pcm_segments_a_fixture_offline():
    pcm = _pcm(silence(2) + speech(12) + silence(8) + speech(12) + silence(8))
    found = utterances_from_pcm(pcm)
    assert len(found) == 2
    assert all(utterance.reason == "silence" for utterance in found)
    assert sum(utterance.duration_ms for utterance in found) == pytest.approx(24 * FRAME_MS)
    # and the PCM is real audio, not a placeholder
    assert len(found[0].pcm) == found[0].bytes_len


def test_utterance_exposes_its_own_duration_and_size():
    utterance = Utterance(frames=speech(5), duration_ms=400)
    assert len(utterance) == 5
    assert utterance.bytes_len == 5 * FRAME_SAMPLES * 2
    assert len(utterance.pcm) == utterance.bytes_len


def _pcm(frames: list) -> bytes:
    from atlas_audio.frames import frames_to_pcm

    return frames_to_pcm(frames)


def test_silence_helper_is_actually_silent():
    assert all(rms(frame) == 0.0 for frame in silence(3))
    assert rms(speech(1)[0]) > 0.05
