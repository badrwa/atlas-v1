"""Recording and replay — the reason this level can be tested at all.

Two jobs:

* **`TurnRecorder`** writes a session (`--capture-dump`) as frames plus a small
  JSON sidecar.  A real recording of a real failure is worth more than any
  synthetic fixture: it becomes the regression corpus, and it can be replayed
  with no microphone and no network.
* **`WavFile`** reads and writes 16 kHz mono WAV using the standard library
  (`wave`), so `atlas listen --replay file.wav` works on a machine with no
  audio dependencies installed at all.

The self-test (`atlas audio test`) records a second of speech and plays it back.
It is the fastest way to find out that Windows has muted your microphone.
"""

from __future__ import annotations

import json
import logging
import sys
import wave
from array import array
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from atlas_audio.frames import (
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    Frame,
    FramePacket,
    frame_to_bytes,
    new_frame,
    pcm_to_frames,
    resample_pcm,
)

log = logging.getLogger(__name__)


# ── WAV ──────────────────────────────────────────────────────────────
@dataclass
class WavFile:
    """16 kHz mono PCM16 in, out, and around — stdlib only."""

    samples: array
    sample_rate: int = SAMPLE_RATE

    @classmethod
    def read(cls, path: str | Path) -> WavFile:
        """Read any PCM WAV, converting to 16 kHz mono on the way in."""
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())

        if width != 2:
            raise ValueError(f"{path}: only 16-bit PCM is supported (got {width * 8}-bit)")
        samples = array("h")
        samples.frombytes(raw)

        if channels > 1 or rate != SAMPLE_RATE:
            # One pass over the whole buffer: per-chunk resampling would stretch
            # the audio (and this is the same function the live mic path uses).
            samples = resample_pcm(samples, channels=channels, source_rate=rate)
        return cls(samples=samples, sample_rate=SAMPLE_RATE)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(target), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(self.samples.tobytes())
        return target

    @property
    def duration_ms(self) -> float:
        return len(self.samples) / self.sample_rate * 1000

    def frames(self) -> list[Frame]:
        return pcm_to_frames(self.samples.tobytes())

    def packets(self, *, source: str = "file") -> list[FramePacket]:
        return [
            FramePacket(frame=frame, index=index, at_ms=index * FRAME_MS, source=source)
            for index, frame in enumerate(self.frames())
        ]

    def __len__(self) -> int:
        return len(self.samples)


def pcm_to_wav_bytes(pcm: bytes, *, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Raw PCM → WAV bytes in memory (used by the cloud ASR uploads)."""
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


# ── synthetic audio (tests, self-test, demos) ────────────────────────
def tone(frames: int, *, freq: float = 440.0, amplitude: float = 0.3, sample_rate: int = SAMPLE_RATE) -> Frame:
    """A loud frame — stands in for speech in segmentation tests."""
    import math

    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * freq * index / sample_rate))
        for index in range(min(frames, FRAME_SAMPLES))
    ]
    return new_frame(samples)


def silence(count: int = 1) -> list[Frame]:
    return [new_frame() for _ in range(count)]


def speech(count: int = 12, *, amplitude: float = 0.35) -> list[Frame]:
    """`count` loud frames with a little variation, so it is not a pure tone."""
    import math

    out: list[Frame] = []
    for index in range(count):
        wobble = 0.6 + 0.4 * math.sin(index / 2)
        out.append(tone(FRAME_SAMPLES, freq=180 + 20 * (index % 5), amplitude=amplitude * wobble))
    return out


# ── recording a session ──────────────────────────────────────────────
@dataclass
class RecordedTurn:
    """One utterance as it happened — replayable, and evidence in the notes."""

    index: int
    pcm_path: str
    duration_ms: float
    transcript: str = ""
    language: str = ""
    confidence: float = 0.0
    engine: str = ""
    reason: str = ""


class TurnRecorder:
    """Dumps frames and transcripts so a session can be replayed exactly.

    Layout:

        recordings/<session>/session.wav   (whole session, when `include_raw`)
        recordings/<session>/turn-0001.wav (one file per utterance)
        recordings/<session>/session.json  (manifest: frames, turns, transcripts)

    Two levels of detail, deliberately.  Per-utterance WAVs are small and are
    what `bench_asr.py` replays.  The raw session WAV is written **only** when
    `--capture-dump` asks for it, because it contains the wake word — and the
    wake word is exactly what a replay needs to reproduce a failure.  It is
    streamed to disk frame by frame, so recording four idle hours costs disk, not
    RAM.
    """

    def __init__(
        self,
        root: str | Path = "recordings",
        *,
        session: str = "",
        enabled: bool = True,
        include_raw: bool = True,
    ) -> None:
        from datetime import datetime

        self.enabled = enabled
        self.include_raw = include_raw
        self.session = session or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.root = Path(root) / self.session
        self.turns: list[RecordedTurn] = []
        self.frames_seen = 0
        self._raw: wave.Wave_write | None = None

    # ── the raw stream ───────────────────────────────────────────────
    def push_frame(self, frame: Frame) -> None:
        """Every frame the microphone saw, in order — including the idle ones."""
        if not (self.enabled and self.include_raw):
            return
        if self._raw is None:
            self.root.mkdir(parents=True, exist_ok=True)
            # Deliberately not a context manager: the handle lives for the whole
            # session and is closed by `close_raw()`. SIM115 is wrong here.
            self._raw = wave.open(str(self.root / "session.wav"), "wb")  # noqa: SIM115
            self._raw.setnchannels(1)
            self._raw.setsampwidth(2)
            self._raw.setframerate(SAMPLE_RATE)
        self._raw.writeframes(frame_to_bytes(frame))

    @property
    def path(self) -> Path:
        return self.root

    def add_turn(
        self,
        pcm: bytes,
        *,
        transcript: str = "",
        language: str = "",
        confidence: float = 0.0,
        engine: str = "",
        reason: str = "",
    ) -> RecordedTurn:
        turn = RecordedTurn(
            index=len(self.turns),
            pcm_path="",
            duration_ms=len(pcm) / 2 / SAMPLE_RATE * 1000,
            transcript=transcript,
            language=language,
            confidence=confidence,
            engine=engine,
            reason=reason,
        )
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            target = self.root / f"turn-{len(self.turns):04d}.wav"
            WavFile(array("h", _as_int16(pcm))).write(target)
            turn.pcm_path = str(target)
        self.turns.append(turn)
        return turn

    def close_raw(self) -> Path | None:
        """Finish the session WAV (the headers are patched on close)."""
        if self._raw is None:
            return None
        path = Path(self._raw._file.name)  # type: ignore[attr-defined]
        self._raw.close()
        self._raw = None
        return path

    def flush(self) -> Path | None:
        if not self.enabled or not self.turns:
            self.close_raw()
            return None
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = self.root / "session.json"
        self.close_raw()
        manifest.write_text(
            json.dumps(
                {
                    "session": self.session,
                    "frames": self.frames_seen,
                    "raw": "session.wav" if (self.root / "session.wav").exists() else "",
                    "turns": [turn.__dict__ for turn in self.turns],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest

    def replay(self) -> Iterator[tuple[Path, RecordedTurn]]:
        """Yield each recorded turn's WAV and metadata, in order."""
        for turn in self.turns:
            if turn.pcm_path and Path(turn.pcm_path).exists():
                yield Path(turn.pcm_path), turn


@dataclass
class Session:
    """A recorded session: what was said, and optionally the whole audio."""

    directory: Path
    turns: list[dict] = field(default_factory=list)
    utterances: list[Path] = field(default_factory=list)
    audio: WavFile | None = None

    @property
    def duration_ms(self) -> float:
        return self.audio.duration_ms if self.audio else 0.0

    def transcripts(self) -> list[str]:
        return [turn.get("transcript", "") for turn in self.turns]


def load_session(path: str | Path, *, with_audio: bool = True) -> Session:
    """Read a session manifest (a directory or the JSON file itself)."""
    target = Path(path)
    manifest = target / "session.json" if target.is_dir() else target
    if not manifest.exists():
        raise FileNotFoundError(f"no session manifest at {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    base = manifest.parent
    utterances = [
        base / Path(turn["pcm_path"]).name
        for turn in data.get("turns", [])
        if turn.get("pcm_path")
    ]
    audio = None
    if with_audio and data.get("raw"):
        raw = base / Path(data["raw"]).name
        if raw.exists():
            audio = WavFile.read(raw)
    return Session(
        directory=base, turns=list(data.get("turns", [])), utterances=utterances, audio=audio
    )


def _as_int16(pcm: bytes) -> list[int]:
    samples = array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    return list(samples)


# ── hardware self-test ───────────────────────────────────────────────
@dataclass
class SelfTestResult:
    recorded_frames: int = 0
    peak: float = 0.0
    played_frames: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and self.recorded_frames > 0

    def summary(self) -> str:
        if self.problems:
            return " · ".join(self.problems)
        if self.peak < 0.01:
            return "recorded, but almost silent — check the mic volume and Windows privacy settings"
        return f"recorded {self.recorded_frames} frames (peak {self.peak:.2f}), playback done"


def self_test(*, seconds: float = 1.0, device: str | int | None = None) -> SelfTestResult:
    """Record and immediately play back — the fastest way to find a muted mic."""
    import time

    result = SelfTestResult()
    try:
        import sounddevice as sd  # type: ignore[import-not-found]
    except (ImportError, OSError):
        result.problems.append("audio support not installed — pip install 'atlas-audio[audio]'")
        return result

    from atlas_audio.frames import rms

    block = int(SAMPLE_RATE * seconds)
    try:
        captured = sd.rec(block, samplerate=SAMPLE_RATE, channels=1, dtype="int16", device=device)
        sd.wait()
    except Exception as exc:
        result.problems.append(f"capture failed: {exc}")
        return result

    samples = array("h")
    samples.frombytes(captured.tobytes())
    frames = pcm_to_frames(samples.tobytes())
    result.recorded_frames = len(frames)
    result.peak = max((rms(frame) for frame in frames), default=0.0)
    if result.peak < 0.005:
        result.problems.append("input is silent — is another app holding the microphone?")

    try:
        sd.play(captured, samplerate=SAMPLE_RATE, device=device)
        sd.wait()
        result.played_frames = len(frames)
    except Exception as exc:
        result.problems.append(f"playback failed: {exc}")
    time.sleep(0.05)
    return result


def platform_notes() -> list[str]:
    """Windows-specific truths the plan calls out, surfaced where they matter."""
    notes = [
        "Use a headset or a quiet room: Windows 'audio enhancements' and AGC wreck VAD.",
        "Settings → Privacy → Microphone → let desktop apps listen.",
        "Exclusive mode steals the device: Sound → your mic → Advanced → uncheck it.",
    ]
    if sys.platform == "win32":  # pragma: no cover - Windows only
        notes.append("If capture fails, close Teams/Zoom/OBS — one of them holds the mic.")
    return notes


__all__ = [
    "RecordedTurn",
    "SelfTestResult",
    "Session",
    "TurnRecorder",
    "WavFile",
    "load_session",
    "pcm_to_wav_bytes",
    "platform_notes",
    "self_test",
    "silence",
    "speech",
    "tone",
]
