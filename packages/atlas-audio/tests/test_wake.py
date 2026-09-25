"""The wake word: it must fire when called and stay quiet when not.

Two rules the level spec is strict about, and both are tested here:

* silence must never produce a detection — the false-wake rate is the number
  that decides whether an always-on assistant is usable at all;
* a hit is *confirmed over frames*, and logged to `data/wake_log.jsonl`, because
  a threshold tuned on vibes is a threshold that drifts.
"""

from __future__ import annotations

import json
from array import array

import pytest

from atlas_audio.capture import silence, speech
from atlas_audio.frames import FRAME_SAMPLES, frame_to_bytes, new_frame
from atlas_audio.wake import (
    DEFAULT_CONFIRM_FRAMES,
    DEFAULT_THRESHOLD,
    EnergyWakeEngine,
    OpenWakeWordEngine,
    SherpaKwsEngine,
    WakeHit,
    WakeLog,
    build_wake_engine,
    wake_engine_status,
)
from atlas_core.contracts import WakeWordEngine


def loud(value: int = 12000):
    return new_frame([value] * FRAME_SAMPLES)


# ── the shared confirmation logic ────────────────────────────────────
def test_silence_never_fires():
    engine = EnergyWakeEngine()
    detections = [engine.feed(frame) for frame in silence(200)]
    assert not any(detection.hit for detection in detections)


def test_two_confirmed_frames_are_required_before_a_hit():
    engine = EnergyWakeEngine(confirm_frames=2)
    assert engine.feed(loud()).hit is False  # one loud frame is not a keyword
    assert engine.feed(loud()).hit is True


def test_a_miss_resets_the_confirmation_run():
    engine = EnergyWakeEngine(confirm_frames=2)
    engine.feed(loud())
    engine.feed(new_frame())  # the run breaks
    assert engine.feed(loud()).hit is False
    assert engine.feed(loud()).hit is True


def test_the_threshold_is_respected_in_both_directions():
    """A 0.122 RMS frame scores 0.76 — above 0.6, below 0.9, as the maths says."""
    medium = new_frame([4000] * FRAME_SAMPLES)
    strict = EnergyWakeEngine(threshold=0.9, confirm_frames=1)
    lenient = EnergyWakeEngine(threshold=0.6, confirm_frames=1)
    assert not any(strict.feed(medium).hit for _ in range(20))
    assert lenient.feed(medium).hit is True
    assert DEFAULT_THRESHOLD == 0.60


def test_engines_accept_bytes_and_frames():
    """`atlas-core`'s contract hands bytes; the loop hands frames."""
    from_bytes = EnergyWakeEngine(confirm_frames=1)
    from_frames = EnergyWakeEngine(confirm_frames=1)
    assert from_bytes.feed(frame_to_bytes(loud())).hit is True
    assert from_frames.feed(loud()).hit is True


def test_energy_engine_is_honest_about_being_a_demo():
    engine = EnergyWakeEngine()
    assert engine.name == "energy-wake"
    assert "test" in (engine.__doc__ or "") or "demo" in (engine.__doc__ or "")


# ── the Sense contract ───────────────────────────────────────────────
async def test_start_stop_are_idempotent_and_leave_nothing_running():
    engine = EnergyWakeEngine()
    assert isinstance(engine, WakeWordEngine)
    await engine.start()
    await engine.start()
    assert engine.is_healthy() is True
    assert engine.feed(loud()) is not None
    await engine.stop()
    await engine.stop()
    assert engine.is_healthy() is True


# ── optional backends degrade, they never raise ──────────────────────
def test_sherpa_engine_without_the_runtime_says_what_is_missing():
    engine = SherpaKwsEngine(keywords=("atlas",))
    if engine.runtime_available():
        pytest.skip("sherpa-onnx is installed in this environment")
    assert engine.missing
    assert engine.load() is False
    assert engine.load_error
    assert engine.feed(loud()).hit is False  # a wake engine that cannot work, silently


def test_openwakeword_without_the_package_says_what_is_missing():
    engine = OpenWakeWordEngine()
    if engine.runtime_available():
        pytest.skip("openWakeWord is installed in this environment")
    assert engine.load() is False
    assert "openwakeword" in engine.load_error


def test_build_wake_engine_degrades_all_the_way_to_energy_with_a_reason(caplog):
    """The chosen engine must be usable, and the *reason* must be in the log."""
    engine = build_wake_engine({"keywords": ["atlas"]}, keywords=["atlas"], log_sink=WakeLog(enabled=False))
    assert isinstance(engine, WakeWordEngine)
    assert engine.keywords == ("atlas",)
    if not SherpaKwsEngine.runtime_available() and not OpenWakeWordEngine.runtime_available():
        assert engine.name == "energy-wake"
        assert "no real wake engine available" in caplog.text
    # and the doctor rows explain the situation to a human
    assert any(status for _, status in wake_engine_status())


def test_wake_engine_status_reports_rows_for_doctor():
    rows = wake_engine_status()
    names = [name for name, _ in rows]
    assert names == ["sherpa-kws", "openwakeword", "energy-wake"]
    assert all(isinstance(status, str) and status for _, status in rows)
    assert any("sherpa-onnx" in status for _, status in rows) or any(
        "ready" in status for _, status in rows
    )


# ── the log ──────────────────────────────────────────────────────────
def test_every_hit_is_logged_with_its_score(tmp_path):
    path = tmp_path / "wake_log.jsonl"
    log = WakeLog(path=path)
    engine = EnergyWakeEngine(confirm_frames=1, log_sink=log)
    engine.feed(loud())

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["keyword"] == "atlas"
    assert record["score"] >= DEFAULT_THRESHOLD
    assert record["engine"] == "energy-wake"
    assert record["at"] > 0


def test_the_log_survives_a_broken_disk(tmp_path):
    """A logging failure must never break listening."""
    log = WakeLog(path=tmp_path / "nope" / "deep" / "wake.jsonl")
    engine = EnergyWakeEngine(confirm_frames=1, log_sink=log)
    assert engine.feed(loud()).hit is True  # no exception, the hit still happens


def test_false_wake_rate_is_measured_per_hour(tmp_path):
    log = WakeLog(path=tmp_path / "wake_log.jsonl", enabled=True)
    now = 1_700_000_000.0
    log.record(WakeHit(keyword="atlas", score=0.7, at=now - 3600 * 3 - 1))  # outside the window
    log.record(WakeHit(keyword="atlas", score=0.8, at=now - 60))
    log.record(WakeHit(keyword="atlas", score=0.9, at=now))

    assert log.false_wake_rate(hours=3, now=now) == pytest.approx(2 / 3)
    assert log.false_wake_rate(hours=1, now=now) == pytest.approx(2.0)
    assert log.false_wake_rate(hours=1, idle_hits=0, now=now) == 0.0  # the 4 h gate
    assert log.false_wake_rate(hours=0, now=now) == 0.0
    assert len(log.tail(2)) == 2


def test_logging_can_be_switched_off_but_hits_are_still_counted():
    log = WakeLog(enabled=False)
    log.record(WakeHit(keyword="atlas", score=0.7, at=1.0))
    assert log.hits  # kept in memory for the session summary…
    assert log.persisted is False  # …and never written to disk


def test_log_reads_back_what_it_wrote(tmp_path):
    path = tmp_path / "wake_log.jsonl"
    log = WakeLog(path=path)
    for index in range(3):
        log.record(
            WakeHit(keyword="atlas", score=0.7 + index / 10, at=100.0 + index, frame_index=index,
                    engine="energy-wake")
        )
    assert log.persisted is True
    again = WakeLog(path=path)
    assert [hit.score for hit in again.tail(10)] == pytest.approx([0.7, 0.8, 0.9])
    assert len(again.hits) == 3
    assert again.hits[0].engine == "energy-wake"  # the record survives whole


def test_a_torn_log_line_does_not_destroy_the_history(tmp_path):
    path = tmp_path / "wake_log.jsonl"
    path.write_text(
        '{"keyword": "atlas", "score": 0.7, "at": 1.0, "frame_index": 1, "engine": "x"}\n{"torn":',
        encoding="utf-8",
    )
    log = WakeLog(path=path)
    assert len(log.hits) == 1


# ── the fixture audio itself ─────────────────────────────────────────
def test_the_synthetic_helpers_are_distinguishable():
    assert array("h", speech(1)[0]) != array("h", silence(1)[0])
    assert DEFAULT_CONFIRM_FRAMES == 2
