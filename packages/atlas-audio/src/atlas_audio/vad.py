"""Voice activity detection and endpointing.

Why not just use a VAD library: because *endpointing* is the product decision
here, and it is ours, not the library's.  A voice assistant that cuts you off
mid-sentence is worse than one that waits a beat too long.  So the model (Silero
via sherpa-onnx, or WebRTC) produces one number per frame — "speech, yes/no" —
and this file decides when an utterance begins, when it ends, and what is
included.

Two backends ship, chosen at call time:

* `energy`  — pure Python, no dependency, works on any machine.  Honest about
  being crude: it is right in a quiet room and wrong next to a fan.
* `silero`  — sherpa-onnx when installed (`pip install 'atlas-audio[local]'`).
  This is what the plan wants in production.

The segmenter is deliberately synchronous and free of I/O: feed it frames, get
utterances.  That is what makes `FakeMic` replay tests possible.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from atlas_audio.frames import (
    FRAME_MS,
    FRAME_SAMPLES,
    Frame,
    FramePacket,
    frames_to_pcm,
    is_silence,
    rms,
)

log = logging.getLogger(__name__)

DEFAULT_SILENCE_MS = 500  # how long a pause means "done talking"
SNAPPY_SILENCE_MS = 350
DEFAULT_MIN_UTTERANCE_MS = 300  # below this it is a cough
DEFAULT_MAX_UTTERANCE_MS = 12_000  # above this, cut it and let them continue


@dataclass(slots=True)
class Utterance:
    """One thing the owner said, ready for ASR."""

    frames: list[Frame]
    started_ms: float = 0.0
    duration_ms: float = 0.0
    reason: str = "silence"  # "silence" | "max_length" | "forced"
    pre_roll_frames: int = 0
    peak: float = 0.0
    speech_frames: int = 0  # how much of `frames` was actually voice

    @property
    def pcm(self) -> bytes:
        return frames_to_pcm(self.frames)

    @property
    def bytes_len(self) -> int:
        return len(self.frames) * FRAME_SAMPLES * 2

    def __len__(self) -> int:
        return len(self.frames)


@dataclass
class SegmenterConfig:
    silence_ms: int = DEFAULT_SILENCE_MS
    min_utterance_ms: int = DEFAULT_MIN_UTTERANCE_MS
    max_utterance_ms: int = DEFAULT_MAX_UTTERANCE_MS
    energy_threshold: float = 0.012
    frame_ms: int = FRAME_MS
    pre_roll_ms: int = 1500
    # How much quiet may sit in front of the first syllable.  The pre-roll ring
    # is there so nothing is clipped; uploading 1.5 s of room tone with every
    # command is not.
    lead_in_ms: int = 200
    snappy: bool = False

    @classmethod
    def from_config(cls, config: Any = None) -> SegmenterConfig:
        """Read `[vad]` (and `[audio]`) — one place where the numbers come from."""
        vad = getattr(config, "vad", None)
        audio = getattr(config, "audio", None)
        return cls(
            silence_ms=int(getattr(vad, "silence_ms", DEFAULT_SILENCE_MS)),
            min_utterance_ms=int(getattr(vad, "min_utterance_ms", DEFAULT_MIN_UTTERANCE_MS)),
            max_utterance_ms=int(getattr(vad, "max_utterance_ms", DEFAULT_MAX_UTTERANCE_MS)),
            energy_threshold=float(getattr(vad, "energy_threshold", 0.012)),
            frame_ms=int(getattr(audio, "frame_ms", FRAME_MS)),
            pre_roll_ms=int(getattr(audio, "pre_roll_ms", 1500)),
            lead_in_ms=int(getattr(vad, "lead_in_ms", 200)),
        )

    @classmethod
    def snappy_config(cls) -> SegmenterConfig:
        """The 350 ms variant: feels fast, occasionally cuts a thoughtful pause."""
        return cls(silence_ms=SNAPPY_SILENCE_MS, snappy=True)

    @property
    def frames_per_silence(self) -> int:
        return max(1, self.silence_ms // self.frame_ms)

    @property
    def min_frames(self) -> int:
        return max(1, self.min_utterance_ms // self.frame_ms)

    @property
    def max_frames(self) -> int:
        return max(1, self.max_utterance_ms // self.frame_ms)

    @property
    def lead_in_frames(self) -> int:
        return max(0, self.lead_in_ms // self.frame_ms)


class EnergyVad:
    """Crude, dependency-free speech detector: loud enough, or not.

    `hangover` is the number of quiet frames still called speech (a plosive's
    gap, a breath).  It is **capped**, and defaults to zero, because the
    `VadSegmenter` already owns pause accounting: an uncapped hangover here
    would swallow the silence run and the utterance would never close — which
    is exactly the bug this comment exists to prevent.
    """

    name = "energy-vad"

    def __init__(self, threshold: float = 0.012, *, hangover: int = 0) -> None:
        if hangover < 0:
            raise ValueError("hangover cannot be negative")
        self.threshold = threshold
        self.hangover = hangover
        self._loud_run = 0

    def is_speech(self, frame: Frame) -> bool:
        if rms(frame) >= self.threshold:
            self._loud_run = min(self._loud_run + 1, self.hangover)
            return True
        if self._loud_run:
            self._loud_run -= 1
            return True  # a brief dip inside a word is not silence
        return False

    def reset(self) -> None:
        self._loud_run = 0


class SileroVad:
    """Silero VAD through sherpa-onnx — the production backend.

    Loading a VAD model per frame would be absurd, so the session is created
    once and reused.  When sherpa-onnx is absent the class stays importable and
    says exactly what to install: `atlas doctor` reports it, and the segmenter
    falls back to the energy backend instead of failing the whole loop.
    """

    name = "silero-vad"
    ram_mb = 20

    def __init__(self, *, threshold: float = 0.5, model_path: str = "") -> None:
        self.threshold = threshold
        self.model_path = model_path
        # sherpa-onnx's Python classes are untyped: `Any` here beats a pile of
        # ignore comments, and the call sites are guarded by `available()`.
        self._session: Any = None
        self._load_error = ""

    @staticmethod
    def available() -> bool:
        try:
            import sherpa_onnx  # type: ignore[import-not-found]

            return hasattr(sherpa_onnx, "VadModel")
        except ImportError:
            return False

    @property
    def load_error(self) -> str:
        return self._load_error

    def load(self) -> bool:
        """False when unavailable (never raises): the caller degrades to energy."""
        if self._session is not None:
            return True
        if not self.available():
            self._load_error = (
                "sherpa-onnx not installed — pip install 'atlas-audio[local]' "
                "(or keep the energy VAD, which is what CI uses)"
            )
            return False
        try:  # pragma: no cover - needs the real model
            import sherpa_onnx

            config = sherpa_onnx.VadModelConfig()
            config.silero_vad.model = self.model_path or "models/silero_vad.onnx"
            config.silero_vad.threshold = self.threshold
            config.sample_rate = 16_000
            self._session = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)
            return True
        except Exception as exc:
            self._load_error = f"silero load failed: {exc}"
            return False

    def is_speech(self, frame: Frame) -> bool:  # pragma: no cover - needs the model
        if self._session is None and not self.load():
            return not is_silence(frame)
        import numpy as np

        samples = np.frombuffer(frame.tobytes(), dtype="<i2").astype("float32") / 32768.0
        self._session.accept_waveform(samples)
        return bool(self._session.is_speech_detected())


class VadSegmenter:
    """Turns a frame stream into utterances.

    Feed every frame; get an `Utterance` back on the frame that ends one.  State
    is small and explicit so the behaviour is testable frame by frame.
    """

    def __init__(
        self,
        config: SegmenterConfig | None = None,
        *,
        detector: Callable[[Frame], bool] | None = None,
    ) -> None:
        self.config = config or SegmenterConfig()
        self._detector = detector
        self._vad: EnergyVad | SileroVad | None = None
        self.reset()

    # ── backend ──────────────────────────────────────────────────────
    def use_silero(self, vad: SileroVad) -> bool:
        """Prefer Silero when it loads. Returns whether the swap happened."""
        if vad.load():
            self._vad = vad
            self._detector = vad.is_speech
            log.info("vad_backend backend=silero")
            return True
        log.info("vad_backend backend=energy reason=%s", vad.load_error)
        return False

    def _speech(self, frame: Frame) -> bool:
        if self._detector is not None:
            return bool(self._detector(frame))
        if self._vad is None:
            self._vad = EnergyVad(self.config.energy_threshold)
        return bool(self._vad.is_speech(frame))

    # ── state ────────────────────────────────────────────────────────
    def reset(self) -> None:
        self.active = False
        self.frames: list[Frame] = []
        self._silence_run = 0
        self._elapsed_ms = 0.0
        self._started_ms = 0.0
        self._pre_roll: list[FramePacket] = []
        self._pre_roll_used = 0
        self._speech_frames = 0
        self.peak = 0.0

    @property
    def pre_roll_frames(self) -> int:
        return int(self.config.pre_roll_ms // self.config.frame_ms)

    def feed(self, packet: FramePacket) -> Utterance | None:
        """One frame in; an `Utterance` out on the frame that closes one."""
        frame = packet.frame
        self._elapsed_ms += self.config.frame_ms
        speech = self._speech(frame)

        if not self.active:
            # Keep the recent past so the first syllable (and the wake word right
            # before it) is not lost.  Bounded, so this cannot grow.
            self._pre_roll.append(packet)
            keep = max(1, self.pre_roll_frames)  # never drop the frame that triggers
            del self._pre_roll[: max(0, len(self._pre_roll) - keep)]
            if speech:
                self.active = True
                self._started_ms = max(0.0, packet.at_ms - (len(self._pre_roll) - 1) * self.config.frame_ms)
                self.frames = [item.frame for item in self._pre_roll]
                self._pre_roll_used = len(self._pre_roll)
                self._silence_run = 0
                self._speech_frames = 1  # this frame is the one that opened it
                self.peak = rms(frame)
            return None

        self.frames.append(frame)
        self.peak = max(self.peak, rms(frame))

        if speech:
            self._silence_run = 0
            self._speech_frames += 1
        else:
            self._silence_run += 1

        if self._silence_run >= self.config.frames_per_silence:
            return self._close("silence")
        if len(self.frames) >= self.config.max_frames:
            return self._close("max_length")
        return None

    def flush(self) -> Utterance | None:
        """Nobody is going to speak again — close whatever is open."""
        if not self.active or not self.frames:
            self.reset()
            return None
        return self._close("forced")

    # ── closing ──────────────────────────────────────────────────────
    def _close(self, reason: str) -> Utterance | None:
        frames = self.frames
        speech_frames = self._speech_frames
        # Drop the trailing silence: sending it to a cloud ASR wastes money and
        # two thirds of a second of everyone's time.
        keep = frames[: max(0, len(frames) - self._silence_run)]
        # …and keep only the onset of the lead-in, not the whole quiet room.
        lead = self.config.lead_in_frames
        trimmed = 0
        while len(keep) > lead and is_silence(keep[0]):
            keep.pop(0)
            trimmed += 1
        # Of what survived, how much arrived before the first syllable?  This is
        # the wake-word tail in the normal path, and it is what the notes quote.
        pre_roll_frames = max(0, min(self._pre_roll_used - trimmed, len(keep)))
        # "Too short" is measured in *speech*, not in frames: padding a cough up
        # to the minimum would make the minimum meaningless.
        too_short = speech_frames < self.config.min_frames
        utterance = Utterance(
            frames=keep,
            started_ms=self._started_ms,
            duration_ms=len(keep) * self.config.frame_ms,
            reason=reason,
            pre_roll_frames=pre_roll_frames,
            peak=self.peak,
            speech_frames=speech_frames,
        )
        self.reset()
        if too_short:
            log.debug("utterance_dropped reason=too_short speech_frames=%s", speech_frames)
            return None
        log.debug(
            "utterance_closed reason=%s frames=%s duration_ms=%.0f peak=%.3f",
            reason,
            len(keep),
            utterance.duration_ms,
            utterance.peak,
        )
        return utterance


def utterances_from_pcm(pcm: bytes, config: SegmenterConfig | None = None) -> list[Utterance]:
    """Segment a whole recording (a WAV fixture, a capture dump) offline."""
    from atlas_audio.frames import pcm_to_frames

    segmenter = VadSegmenter(config)
    found: list[Utterance] = []
    for index, frame in enumerate(pcm_to_frames(pcm)):
        result = segmenter.feed(
            FramePacket(frame=frame, index=index, at_ms=index * (config or SegmenterConfig()).frame_ms)
        )
        if result:
            found.append(result)
    if tail := segmenter.flush():
        found.append(tail)
    return found


__all__ = [
    "DEFAULT_MAX_UTTERANCE_MS",
    "DEFAULT_MIN_UTTERANCE_MS",
    "DEFAULT_SILENCE_MS",
    "SNAPPY_SILENCE_MS",
    "EnergyVad",
    "SegmenterConfig",
    "SileroVad",
    "Utterance",
    "VadSegmenter",
    "utterances_from_pcm",
]
