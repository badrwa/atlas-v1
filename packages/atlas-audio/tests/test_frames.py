"""Frames, ring buffer, fan-out bus and the capture thread.

Everything here is hardware-free: the producer takes an injected reader, which
is the same injection point `atlas listen --capture-dump` uses for replay.
"""

from __future__ import annotations

import asyncio
import threading
import time
from array import array

import pytest

from atlas_audio.frames import (
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    FrameBus,
    FramePacket,
    FrameProducer,
    RingBuffer,
    downmix_and_resample,
    frame_from_bytes,
    frame_to_bytes,
    frames_to_pcm,
    is_silence,
    new_frame,
    pcm_to_frames,
    rms,
    to_numpy,
)


def loud(value: int = 12000) -> array:
    return new_frame([value] * FRAME_SAMPLES)


# ── geometry ─────────────────────────────────────────────────────────
def test_frame_geometry_matches_the_plan():
    assert (FRAME_MS, FRAME_SAMPLES, FRAME_BYTES, SAMPLE_RATE) == (80, 1280, 2560, 16000)
    assert len(new_frame()) == FRAME_SAMPLES
    assert len(new_frame().tobytes()) == FRAME_BYTES
    assert not is_silence(loud())
    assert is_silence(new_frame())


def test_short_and_long_input_are_normalised_never_dropped_silently():
    short = new_frame([1, 2, 3])
    long = new_frame(list(range(FRAME_SAMPLES * 3)))
    assert len(short) == len(long) == FRAME_SAMPLES
    assert list(short[:3]) == [1, 2, 3]


def test_pcm_roundtrip_pads_the_final_partial_frame():
    pcm = frames_to_pcm([loud(), loud()]) + b"\x10\x00" * 5
    frames = pcm_to_frames(pcm)
    assert len(frames) == 3  # the 5-sample tail becomes a padded frame, not a lost one
    assert frames_to_pcm(frames[:2]) == pcm[: FRAME_BYTES * 2]


def test_bytes_and_numpy_helpers_agree():
    frame = loud(1000)
    assert frame_from_bytes(frame_to_bytes(frame)) == frame
    assert rms(new_frame()) == 0.0
    assert 0.0 < rms(frame) < 0.1
    import numpy as np

    assert np.asarray(to_numpy(frame)).dtype == np.int16
    assert len(to_numpy(frame)) == FRAME_SAMPLES


def test_rms_rejects_neither_numpy_nor_overflow():
    assert rms(new_frame([-32768] * FRAME_SAMPLES)) == pytest.approx(1.0, abs=0.001)


# ── device conversion ────────────────────────────────────────────────
def test_downmix_takes_the_conversation_not_the_echo():
    stereo = array("h")
    for _ in range(FRAME_SAMPLES):
        stereo.extend([2000, -2000])  # opposite polarity: a phase-inverted pair
    frame = downmix_and_resample(stereo, channels=2, source_rate=16000)
    assert len(frame) == FRAME_SAMPLES
    assert rms(frame) == 0.0


def test_downsample_from_48k_keeps_one_frame_of_80ms():
    source = array("h", [3000] * (48_000 * 80 // 1000))
    frame = downmix_and_resample(source, channels=1, source_rate=48_000)
    assert len(frame) == FRAME_SAMPLES
    assert not is_silence(frame)


# ── ring buffer (the pre-roll) ───────────────────────────────────────
def test_ring_buffer_keeps_exactly_the_pre_roll_window():
    ring = RingBuffer(seconds=0.24)  # 3 frames
    for index in range(10):
        ring.push(FramePacket(frame=new_frame(), index=index, at_ms=index * FRAME_MS))
    assert len(ring) == 3
    assert [packet.index for packet in ring.snapshot()] == [7, 8, 9]
    assert ring.seconds == pytest.approx(0.24)
    assert [packet.index for packet in ring.drain()] == [7, 8, 9]
    assert len(ring) == 0


def test_ring_buffer_defaults_to_the_spec_pre_roll():
    ring = RingBuffer()  # 1.5 s from the level spec
    assert ring.capacity == 1500 // FRAME_MS


# ── bus ──────────────────────────────────────────────────────────────
async def test_bus_fans_out_to_every_subscriber():
    bus = FrameBus()
    first = bus.subscribe()
    second = bus.subscribe()
    bus.publish(FramePacket(frame=loud(), index=0))
    assert bus.published == 1 and bus.dropped == 0
    assert (await asyncio.wait_for(first.get(), 1)).index == 0
    assert (await asyncio.wait_for(second.get(), 1)).index == 0
    bus.unsubscribe(first)
    assert bus.subscriber_count == 1


async def test_bus_drops_instead_of_stalling_the_microphone():
    bus = FrameBus(maxsize=2)
    queue = bus.subscribe()
    for index in range(5):
        bus.publish(FramePacket(frame=loud(), index=index))
    assert bus.published == 5
    assert bus.dropped == 3  # a slow consumer loses frames, never the capture thread
    assert queue.qsize() == 2


async def test_bus_stream_unsubscribes_on_exit():
    bus = FrameBus()
    stream = bus.stream()
    task = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    assert bus.subscriber_count == 1
    bus.publish(FramePacket(frame=loud()))
    packet = await asyncio.wait_for(task, 1)
    assert packet.index == 0
    await stream.aclose()
    assert bus.subscriber_count == 0


# ── producer ─────────────────────────────────────────────────────────
def test_producer_publishes_injected_frames_and_stops_cleanly():
    bus = FrameBus()
    stop = threading.Event()

    def reader():
        for _ in range(5):
            if stop.is_set():
                return
            yield loud()

    producer = FrameProducer(bus, reader=reader)
    producer.start()
    deadline = time.monotonic() + 2
    while bus.published < 5 and time.monotonic() < deadline:
        time.sleep(0.01)
    producer.stop()

    assert not producer.running
    assert bus.published == 5
    assert producer.frames_captured == 5
    assert producer.error is None


def test_producer_reports_a_dead_microphone_instead_of_raising():
    seen: list[Exception] = []

    def reader():
        raise RuntimeError("device busy")
        yield  # pragma: no cover - generator marker

    producer = FrameProducer(FrameBus(), reader=reader, on_error=seen.append)
    producer.start()
    time.sleep(0.05)
    producer.stop()
    assert seen and "device busy" in str(seen[0])
    assert producer.error is not None


def test_start_is_idempotent():
    release = threading.Event()

    def reader():
        while not release.is_set():
            yield loud()

    producer = FrameProducer(FrameBus(), reader=reader)
    producer.start()
    thread = producer._thread
    producer.start()  # second start must not spawn a second capture thread
    assert producer._thread is thread
    release.set()
    producer.stop()
    assert not producer.running


def test_producer_without_sounddevice_explains_what_to_install():
    """CI has no audio stack: the error must be a sentence, not an ImportError."""
    try:
        import sounddevice  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, OSError):
        pass
    else:
        pytest.skip("sounddevice is installed here — the absence path is not reachable")

    with pytest.raises(RuntimeError, match=r"pip install 'atlas-audio\[audio\]'"):
        FrameProducer(FrameBus())._open_device()
