"""Playback: the last metre between a synthesised sentence and the speakers.

Three responsibilities, deliberately in one place because they are one gesture:

* **Write PCM to the sound card.** `sounddevice`'s *raw* output stream, so no
  numpy array is needed to hand over int16 samples — the audio core stays
  numpy-free and `array("h")` stays the one in-memory form (L2's rule).
* **Shape the audio**: convert to 16 kHz once per sentence, fade the ends so the
  joins do not click, and apply the mood's gain.
* **Own the microphone while speaking** (`MicGate`).  Half duplex is the plan's
  hard rule; putting the gate here — next to the thing that makes noise — means
  no future caller can forget it.

Metering rides along for free: the samples are already in hand, so the orb's
amplitude is computed from what was *actually played*, not from a second capture
that would drift.
"""

from __future__ import annotations

import logging
from array import array
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from atlas_audio.frames import FRAME_MS, SAMPLE_RATE, resample_pcm
from atlas_core.errors import Unsupported

log = logging.getLogger(__name__)

#: 40 ms at the edges at 16 kHz.  Long enough to kill the click, short enough
#: that a one-word reply is not mostly fade.
FADE_FRAMES = int(SAMPLE_RATE * 0.040)


def fade(samples: array, *, frames: int = FADE_FRAMES) -> array:
    """Linear fade in/out, in place-safe fashion.  No-op on very short buffers."""
    if len(samples) <= frames * 2:
        return samples
    for index in range(frames):
        weight = index / frames
        samples[index] = int(samples[index] * weight)
        samples[-1 - index] = int(samples[-1 - index] * weight)
    return samples


def apply_gain(samples: array, gain: float) -> array:
    """Scale, clamping — a loud gain must clip, not wrap around to noise."""
    if gain == 1.0:
        return samples
    for index, value in enumerate(samples):
        scaled = int(value * gain)
        samples[index] = 32767 if scaled > 32767 else (-32768 if scaled < -32768 else scaled)
    return samples


def peak_level(samples: array) -> float:
    """0..1 amplitude for the orb — cheap, and it is the *played* signal."""
    if not samples:
        return 0.0
    biggest = max(max(samples), -min(samples))
    return min(1.0, biggest / 32767.0)


def to_pcm_bytes(samples: array) -> bytes:
    return samples.tobytes()


# ── devices ──────────────────────────────────────────────────────────
class SoundDeviceWriter:
    """`sounddevice.RawOutputStream` — bytes in, sound out.

    Opened once and kept open: opening a stream per sentence costs ~30 ms and
    adds a click at every sentence boundary, which is exactly the "seamless join"
    the plan asks for.  `latency="high"` is the right trade on a laptop running a
    language model, an ASR model and a conversation at the same time.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        device: str | int | None = None,
        blocksize: int = 0,
        latency: str = "high",
    ) -> None:
        self.sample_rate = sample_rate
        self.device = device
        self.blocksize = blocksize
        self.latency = latency
        self.stream: Any = None
        self.bytes_written = 0

    @staticmethod
    def available() -> bool:
        try:
            import sounddevice  # noqa: F401
        except (ImportError, OSError):
            # OSError: sounddevice installed but no PortAudio DLL — a real state
            # on a fresh Windows box, and not something to crash over.
            return False
        return True

    def open(self) -> None:
        if self.stream is not None:
            return
        if not self.available():
            raise Unsupported("sounddevice not installed — pip install 'atlas-audio[audio]'")
        import sounddevice as sd

        self.stream = sd.RawOutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.blocksize,
            latency=self.latency,
            device=self.device,
        )
        self.stream.start()

    def write(self, samples: array) -> None:
        if not samples:
            return
        self.open()
        self.stream.write(samples.tobytes())
        self.bytes_written += len(samples) * 2

    def close(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        finally:
            self.stream = None


class NullWriter:
    """A writer that swallows audio — for `--dry-run`, tests and CI."""

    def __init__(self) -> None:
        self.samples = 0
        self.bytes_written = 0
        self.closed = False

    def write(self, samples: array) -> None:
        self.samples += len(samples)
        self.bytes_written += len(samples) * 2

    def close(self) -> None:
        self.closed = True

    @property
    def seconds(self) -> float:
        return self.samples / SAMPLE_RATE


# ── half duplex ──────────────────────────────────────────────────────
class MicGate:
    """Closes the microphone while Atlas talks, and reopens it after the tail.

    It holds *intent*, not state: the callbacks are the loop's own `mute` /
    `unmute`, which are reference counted (L3 fix), so two overlapping reasons to
    stay quiet cannot open the mic early.  Idempotent on purpose — the mouth
    closes it before every sentence's first byte.
    """

    def __init__(self, *, on_close: Any = None, on_open: Any = None) -> None:
        self.on_close = on_close
        self.on_open = on_open
        self.muted = False
        self.closes = 0

    def close(self) -> None:
        if self.muted:
            return
        self.muted = True
        self.closes += 1
        if self.on_close:
            self.on_close()

    def open(self) -> None:
        if not self.muted:
            return
        self.muted = False
        if self.on_open:
            self.on_open()

    @contextmanager
    def guarded(self) -> Iterator[MicGate]:
        """Close for the duration of the block, always reopening."""
        self.close()
        try:
            yield self
        finally:
            self.open()


class DuckingController:
    """Master gain: base level, a duck for quiet hours, and a stop-the-music hook.

    OS-level ducking of *other* applications is a WASAPI session thing and is not
    pretended here; what this owns is Atlas's own level, which is what the mood
    table, the quiet-hours rule and `preferences.md` all want to influence.
    """

    def __init__(self, *, base_gain: float = 1.0, duck_factor: float = 0.6) -> None:
        self.base_gain = max(0.0, min(2.0, base_gain))
        self.duck_factor = max(0.0, min(1.0, duck_factor))
        self.ducked = False

    @property
    def gain(self) -> float:
        return self.base_gain * (self.duck_factor if self.ducked else 1.0)

    def duck(self) -> None:
        self.ducked = True

    def restore(self) -> None:
        self.ducked = False

    def as_dict(self) -> dict[str, object]:
        return {"base_gain": round(self.base_gain, 2), "ducked": self.ducked, "gain": round(self.gain, 3)}


@dataclass(slots=True)
class PlaybackResult:
    """What one sentence cost — the numbers `bench_tts.py` reports."""

    seconds: float = 0.0
    bytes_written: int = 0
    peak: float = 0.0
    converted: bool = False  # had to resample (engine rate ≠ 16 kHz)

    def as_dict(self) -> dict[str, object]:
        return {
            "seconds": round(self.seconds, 3),
            "bytes": self.bytes_written,
            "peak": round(self.peak, 3),
            "converted": self.converted,
        }


class AudioPlayer:
    """Turns PCM into sound, with the gate closed for exactly as long as it lasts."""

    def __init__(
        self,
        writer: Any = None,
        *,
        sample_rate: int = SAMPLE_RATE,
        gain: float = 1.0,
        ducking: DuckingController | None = None,
        gate: MicGate | None = None,
        fade_ms: int = 40,
        level_hz: int = 30,
        on_level: Any = None,
    ) -> None:
        self.writer = writer if writer is not None else SoundDeviceWriter(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.ducking = ducking or DuckingController(base_gain=gain)
        self.gate = gate or MicGate()
        self.fade_frames = int(sample_rate * fade_ms / 1000)
        self.level_hz = level_hz
        self.on_level = on_level
        self.played_sentences = 0
        self.played_seconds = 0.0
        self.stopped = False

    @property
    def gain(self) -> float:
        return self.ducking.gain

    @property
    def available(self) -> bool:
        return bool(getattr(self.writer, "available", lambda: True)())

    def stop(self) -> None:
        """Finish the current sentence, drop the rest (interruption policy)."""
        self.stopped = True

    def reset(self) -> None:
        self.stopped = False

    def play(self, pcm: bytes, *, sample_rate: int = SAMPLE_RATE, gain: float = 1.0) -> PlaybackResult:
        """Play one already-synthesised sentence.  Blocking, by design."""
        if not pcm:
            return PlaybackResult()
        samples = array("h")
        samples.frombytes(pcm)
        return self.play_samples(samples, sample_rate=sample_rate, gain=gain)

    def play_samples(self, samples: array, *, sample_rate: int = SAMPLE_RATE, gain: float = 1.0) -> PlaybackResult:
        result = PlaybackResult()
        if not samples:
            return result

        if sample_rate != self.sample_rate:
            # One conversion over the whole sentence — the L2 lesson: resampling
            # per chunk stretches the audio and desynchronises the joins.
            samples = resample_pcm(samples, source_rate=sample_rate)
            result.converted = True

        samples = apply_gain(samples, self.gain * gain)
        fade(samples, frames=self.fade_frames)

        result.peak = self._meter(samples, sample_rate=self.sample_rate)
        result.seconds = len(samples) / self.sample_rate
        # The gate is closed *before* the first byte reaches the device and only
        # reopened once the last one has: anything else and the microphone hears
        # the tail of Atlas's own voice.
        with self.gate.guarded():
            self.writer.write(samples)
        result.bytes_written = len(samples) * 2
        self.played_sentences += 1
        self.played_seconds += result.seconds
        return result

    def _meter(self, samples: array, *, sample_rate: int) -> float:
        """Emit level samples at ~`level_hz` and return the sentence's peak."""
        if not self.on_level:
            return peak_level(samples)
        window = max(1, int(sample_rate / self.level_hz))
        peak = 0.0
        for start in range(0, len(samples), window):
            level = peak_level(samples[start : start + window])
            peak = max(peak, level)
            self.on_level(level)
        return peak

    def play_tones(self, frames: int = 3, *, freq: float = 880.0, amplitude: float = 0.25) -> PlaybackResult:
        """The wake chime.  Synthesised, not shipped as a file — 80 lines of WAV
        in the repo for a beep would be silly."""
        import math

        total = int(self.sample_rate * FRAME_MS / 1000) * max(1, frames)
        samples = array(
            "h",
            (
                int(amplitude * 32767 * math.sin(2 * math.pi * freq * index / self.sample_rate))
                for index in range(total)
            ),
        )
        return self.play_samples(samples, sample_rate=self.sample_rate)

    def say(self, chunks: Iterable[Any]) -> list[PlaybackResult]:
        """Play a synthesizer's `AudioChunk` iterator to the end."""
        results: list[PlaybackResult] = []
        for chunk in chunks:
            if self.stopped:
                break
            if chunk.pcm:
                results.append(self.play(chunk.pcm, sample_rate=chunk.sample_rate))
        return results

    def close(self) -> None:
        if hasattr(self.writer, "close"):
            self.writer.close()
        self.gate.open()

    def as_dict(self) -> dict[str, object]:
        return {
            "sentences": self.played_sentences,
            "seconds": round(self.played_seconds, 2),
            "gain": round(self.gain, 3),
            "muted": self.gate.muted,
        }


__all__ = [
    "FADE_FRAMES",
    "AudioPlayer",
    "DuckingController",
    "MicGate",
    "NullWriter",
    "PlaybackResult",
    "SoundDeviceWriter",
    "apply_gain",
    "fade",
    "peak_level",
    "to_pcm_bytes",
]
