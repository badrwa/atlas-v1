"""Frames, the ring buffer, and the bus that fans them out.

Three decisions worth knowing about before reading the code:

**16 kHz mono, always.** Everything upstream (a 48 kHz headset, a stereo
webcam) is converted at the edge, here. Models want 16 kHz mono; a project that
lets sampling rates leak inward is a project with a different bug every week.

**Frames are `array('h')`, not numpy.** Atlas's whole audio core has to work on
a machine where the audio extras were never installed — `atlas doctor` and the
test suite run without numpy. Pure-Python int16 maths over 80 ms frames is
16 000 samples/second: nothing, next to the model that follows it. `to_numpy()`
exists for the two places that genuinely want it (faster-whisper, sounddevice).

**The ring buffer is the reason wake word detection works at all.** When the
wake word fires, the words that came *with* it are already in the past. 1.5 s of
pre-roll is prepended to the utterance, or "atlas, chno ljaw?" loses "atlas".
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from array import array
from collections import deque
from collections.abc import AsyncGenerator, Callable, Iterable, Iterator
from dataclasses import dataclass

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
FRAME_MS = 80
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1280
FRAME_BYTES = FRAME_SAMPLES * 2  # int16
SAMPLE_WIDTH = 2
MAX_INT16 = 32_767


# ── frame type ───────────────────────────────────────────────────────
Frame = array  # array('h') — one int16 sample per element


def new_frame(samples: Iterable[int] = ()) -> Frame:
    """Exactly one frame: short input is zero-padded, long input is truncated.

    Exact length is an invariant the rest of the audio layer relies on — RMS,
    duration arithmetic, `FRAME_BYTES` and the segmenter's frame counting all
    assume it, so a frame can never be any other size.
    """
    frame = array("h", samples)
    if len(frame) < FRAME_SAMPLES:
        frame.extend([0] * (FRAME_SAMPLES - len(frame)))
    elif len(frame) > FRAME_SAMPLES:
        del frame[FRAME_SAMPLES:]
    return frame


def frame_from_bytes(raw: bytes) -> Frame:
    """PCM16 little-endian bytes → frame (truncated/padded to one frame)."""
    frame = array("h")
    frame.frombytes(raw[:FRAME_BYTES])
    return new_frame(frame)


def frame_to_bytes(frame: Frame) -> bytes:
    return frame.tobytes()


def frames_to_pcm(frames: Iterable[Frame]) -> bytes:
    return b"".join(frame_to_bytes(frame) for frame in frames)


def pcm_to_frames(pcm: bytes) -> list[Frame]:
    """Split PCM16 into whole frames; the tail is padded, never dropped silently."""
    frames: list[Frame] = []
    for start in range(0, len(pcm), FRAME_BYTES):
        frames.append(frame_from_bytes(pcm[start : start + FRAME_BYTES]))
    return frames


def to_numpy(frame: Frame):
    """numpy view of a frame — only for the libraries that require it."""
    import numpy as np

    return np.frombuffer(frame.tobytes(), dtype="<i2")


def resample_pcm(
    data: array, channels: int = 1, source_rate: int = SAMPLE_RATE
) -> array:
    """Downmix and resample a whole buffer to 16 kHz mono.

    Averaging channels then decimating is crude on purpose: a proper resampler
    is a dependency, and the models are trained on far worse telephone audio.
    What matters is that the conversion happens *here*, once — and over the whole
    buffer, not per frame, because resampling each frame separately loses the
    phase between frames and silently stretches the audio.
    """
    if channels > 1:
        mono = array("h")
        for index in range(0, len(data) - channels + 1, channels):
            total = sum(data[index + offset] for offset in range(channels))
            mono.append(int(total / channels))
        data = mono

    if source_rate == SAMPLE_RATE or not data:
        return data

    ratio = source_rate / SAMPLE_RATE
    resampled = array("h")
    position = 0.0
    while int(position) < len(data):
        resampled.append(data[int(position)])
        position += ratio
    return resampled


def downmix_and_resample(
    data: array, channels: int = 1, source_rate: int = SAMPLE_RATE
) -> Frame:
    """One device block in, exactly one canonical frame out."""
    return new_frame(resample_pcm(data, channels=channels, source_rate=source_rate))


def rms(frame: Frame) -> float:
    """Loudness of one frame, 0..1 — the number the VAD and wake engines use."""
    if not frame:
        return 0.0
    total = 0
    for sample in frame:
        total += sample * sample
    return (total / len(frame)) ** 0.5 / MAX_INT16


def is_silence(frame: Frame, threshold: float = 0.01) -> bool:
    return rms(frame) < threshold


@dataclass(slots=True)
class FramePacket:
    """A frame plus where it came from — needed to replay a session exactly."""

    frame: Frame
    index: int = 0
    at_ms: float = 0.0
    source: str = "mic"


# ── pre-roll ─────────────────────────────────────────────────────────
class RingBuffer:
    """Fixed-length history of frames. The wake word needs the recent past."""

    def __init__(self, seconds: float = 1.5, *, frame_ms: int = FRAME_MS) -> None:
        self.capacity = max(1, int(seconds * 1000 / frame_ms))
        self._frames: deque[FramePacket] = deque(maxlen=self.capacity)

    def push(self, packet: FramePacket) -> None:
        self._frames.append(packet)

    def snapshot(self) -> list[FramePacket]:
        return list(self._frames)

    def drain(self) -> list[FramePacket]:
        """Take everything and empty the buffer (used when an utterance starts)."""
        frames = list(self._frames)
        self._frames.clear()
        return frames

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def seconds(self) -> float:
        return len(self._frames) * FRAME_MS / 1000


# ── bus ──────────────────────────────────────────────────────────────
class FrameBus:
    """One producer thread, many consumers, no blocking on any of them.

    Consumers are async iterators.  A slow consumer (a model loading, a disk
    write) drops frames instead of stalling the microphone: for a voice
    assistant, missing a frame is recoverable, stuttering the capture is not.
    """

    def __init__(self, *, maxsize: int = 64) -> None:
        self.maxsize = maxsize
        self._subscribers: list[asyncio.Queue[FramePacket]] = []
        self._lock = threading.Lock()
        self.dropped = 0
        self.published = 0

    def subscribe(self, *, maxsize: int | None = None) -> asyncio.Queue[FramePacket]:
        queue: asyncio.Queue[FramePacket] = asyncio.Queue(maxsize=maxsize or self.maxsize)
        with self._lock:
            self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[FramePacket]) -> None:
        with self._lock, contextlib.suppress(ValueError):
            self._subscribers.remove(queue)

    def publish(self, packet: FramePacket) -> int:
        """Hand a frame to every subscriber. Returns how many took it."""
        self.published += 1
        delivered = 0
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(packet)
                delivered += 1
            except asyncio.QueueFull:
                self.dropped += 1
        return delivered

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    async def stream(self, *, maxsize: int | None = None) -> AsyncGenerator[FramePacket, None]:
        queue = self.subscribe(maxsize=maxsize)
        try:
            while True:
                yield await queue.get()
        finally:
            self.unsubscribe(queue)


class FrameProducer:
    """Reads the microphone on a thread and publishes frames to the bus.

    `reader` is injected: tests pass a function that yields synthetic frames, so
    the whole capture path — including pacing and thread handoff — is exercised
    without a sound card.
    """

    def __init__(
        self,
        bus: FrameBus,
        *,
        reader: Callable[[], Iterator[Frame]] | None = None,
        device: str | int | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = FRAME_MS,
        channels: int = 1,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.bus = bus
        self.device = device
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.channels = channels
        self.on_error = on_error
        self._reader = reader
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.frames_captured = 0
        self.error: Exception | None = None

    # ── lifecycle ────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="atlas-capture", daemon=True)
        self._thread.start()
        log.info("capture_started device=%s rate=%s", self.device, self.sample_rate)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── the thread ───────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            frames = self._reader() if self._reader is not None else self._open_device()
            for index, frame in enumerate(frames):
                if self._stop.is_set():
                    break
                self.frames_captured += 1
                self.bus.publish(
                    FramePacket(
                        frame=frame,
                        index=index,
                        at_ms=index * self.frame_ms,
                        source="mic",
                    )
                )
        except Exception as exc:
            self.error = exc
            log.warning("capture_failed error=%s", exc)
            if self.on_error:
                self.on_error(exc)

    def _open_device(self) -> Iterator[Frame]:
        """The real microphone. Raises a sentence, not a traceback, when absent."""
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "microphone support is not installed — run: "
                "pip install 'atlas-audio[audio]'  (then: atlas audio devices)"
            ) from exc

        blocksize = int(self.sample_rate * self.frame_ms / 1000)

        def generator() -> Iterator[Frame]:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=blocksize,
                device=self.device,
                channels=self.channels,
                dtype="int16",
            ) as stream:
                while not self._stop.is_set():
                    raw, _overflowed = stream.read(blocksize)
                    yield downmix_and_resample(
                        array("h", raw), channels=self.channels, source_rate=self.sample_rate
                    )

        return generator()


__all__ = [
    "FRAME_BYTES",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "MAX_INT16",
    "SAMPLE_RATE",
    "Frame",
    "FrameBus",
    "FramePacket",
    "FrameProducer",
    "RingBuffer",
    "downmix_and_resample",
    "frame_from_bytes",
    "frame_to_bytes",
    "frames_to_pcm",
    "is_silence",
    "new_frame",
    "pcm_to_frames",
    "resample_pcm",
    "rms",
    "to_numpy",
]
