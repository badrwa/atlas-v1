"""Wake word detection — the always-on part, so it has to be cheap and honest.

Two implementations, one contract (`WakeWordEngine.feed(frame) -> Detection`):

* `SherpaKwsEngine` — sherpa-onnx keyword spotting. The plan's primary: one
  runtime for KWS/VAD/ASR, ~30 MB resident, Windows x64 ONNX models.
* `OpenWakeWordEngine` — the alternative, and the one whose *custom* model
  ("atlas") can be trained free on Colab from synthetic speech.

A third ships for tests and first-run demos: `EnergyWakeEngine`, which fires on
loudness. It is *not* a wake word engine and says so in its own docs — it exists
so the rest of the pipeline (pre-roll, VAD, ASR, half-duplex) can be driven
end-to-end without a model file.

Rules that matter more than the models:

* **Two-frame confirmation.** A single loud frame is a door, not a name.
* **Thresholds are not tuned down to make demos work.** The plan says it
  explicitly: lowering the threshold ships a TV-activated assistant. Every
  detection is logged with its score so the *data* decides, not the mood.
"""

from __future__ import annotations

import logging
import time
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas_audio.frames import Frame, frame_from_bytes, rms
from atlas_core.contracts import Detection, WakeWordEngine
from atlas_core.jsonl import JsonlLog

log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.60
DEFAULT_CONFIRM_FRAMES = 2


@dataclass(slots=True)
class WakeHit:
    """One detection, as the log records it — scores are how thresholds get tuned."""

    keyword: str
    score: float
    at: float
    frame_index: int = 0
    engine: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "keyword": self.keyword,
            "score": round(self.score, 4),
            "at": self.at,
            "frame_index": self.frame_index,
            "engine": self.engine,
        }


class WakeLog(JsonlLog[WakeHit]):
    """Append-only `data/wake_log.jsonl` — loaded on start, never fatal.

    Loading matters here: the L2 gate is "zero false wakes in four idle hours",
    so the log has to survive the restarts inside those four hours.
    """

    def __init__(self, path: str | Path = "data/wake_log.jsonl", *, enabled: bool = True) -> None:
        super().__init__(path, enabled=enabled, load=True)

    def _from_dict(self, data: dict[str, Any]) -> WakeHit:
        return WakeHit(**data)

    @property
    def hits(self) -> list[WakeHit]:
        """The wake log's word for its records — the name the notes use."""
        return self.records

    def record(self, hit: WakeHit) -> None:
        """A wake-hit's `add`. Kept as a separate name because it reads better."""
        self.add(hit)

    def false_wake_rate(
        self, *, hours: float, idle_hits: int | None = None, now: float | None = None
    ) -> float:
        """Hits per hour over an idle window — the number the L2 gate measures.

        `idle_hits` exists because the 4-hour gate is measured by a watcher that
        timed its own window; the default counts the log instead, which is what
        `atlas listen --status` can do after the fact.
        """
        if hours <= 0:
            return 0.0
        if idle_hits is None:
            cutoff = (time.time() if now is None else now) - hours * 3600
            idle_hits = sum(1 for hit in self.hits if hit.at >= cutoff)
        return idle_hits / hours

    def tail(self, limit: int = 20) -> list[WakeHit]:
        return self.hits[-limit:]


class _ConfirmingWake(WakeWordEngine):
    """Shared logic: score each frame, confirm over N frames, log every hit."""

    keywords: tuple[str, ...] = ("atlas",)
    confirm_frames: int = DEFAULT_CONFIRM_FRAMES

    def __init__(
        self,
        *,
        keywords: tuple[str, ...] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        confirm_frames: int | None = None,
        log_sink: WakeLog | None = None,
    ) -> None:
        if keywords:
            self.keywords = tuple(keywords)
        self.threshold = threshold
        self.confirm_frames = confirm_frames or DEFAULT_CONFIRM_FRAMES
        self.log = log_sink or WakeLog(enabled=False)
        self._run = 0
        self._run_keyword = ""
        self._frames_seen = 0
        self._started = False

    # ── Sense contract ───────────────────────────────────────────────
    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._started = False
        self.reset()

    def is_healthy(self) -> bool:
        return True

    def reset(self) -> None:
        self._run = 0
        self._run_keyword = ""

    # ── WakeWordEngine contract ──────────────────────────────────────
    def feed(self, frame: bytes | Frame) -> Detection:
        """`atlas-core`'s contract hands us bytes; the loop hands us frames.

        Both are accepted, and bytes are widened here rather than at every call
        site — one normalisation point, no duplicated decoding logic.
        """
        self._frames_seen += 1
        keyword, score = self._score(frame if isinstance(frame, array) else frame_from_bytes(frame))
        if keyword and score >= self.threshold:
            if keyword == self._run_keyword:
                self._run += 1
            else:
                self._run_keyword, self._run = keyword, 1
            if self._run >= self.confirm_frames:
                hit = WakeHit(
                    keyword=keyword,
                    score=score,
                    at=time.time(),
                    frame_index=self._frames_seen,
                    engine=self.name,
                )
                self.log.record(hit)
                self.reset()
                return Detection(hit=True, keyword=keyword, score=score)
        else:
            self.reset()
        return Detection(hit=False, keyword=self._run_keyword, score=score)

    def _score(self, frame: Frame) -> tuple[str, float]:
        raise NotImplementedError


class SherpaKwsEngine(_ConfirmingWake):
    """sherpa-onnx keyword spotting — the always-on engine the plan picks.

    Without sherpa-onnx (or without the keyword model file) this class stays
    importable and `available()` says exactly what is missing, so `atlas doctor`
    can report it in one line.
    """

    name = "sherpa-kws"
    ram_mb = 30

    def __init__(
        self,
        *,
        model_dir: str | Path = "models/kws",
        keywords_file: str | Path = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_dir = Path(model_dir)
        self.keywords_file = Path(keywords_file) if keywords_file else self.model_dir / "keywords.txt"
        self._spotter: Any = None
        self._stream = None
        self._load_error = ""

    # ── availability ─────────────────────────────────────────────────
    @staticmethod
    def runtime_available() -> bool:
        try:
            import sherpa_onnx  # type: ignore[import-not-found]

            return hasattr(sherpa_onnx, "KeywordSpotter")
        except ImportError:
            return False

    @property
    def missing(self) -> list[str]:
        gaps: list[str] = []
        if not self.runtime_available():
            gaps.append("sherpa-onnx (pip install 'atlas-audio[local]')")
        for path, what in ((self.model_dir, "keyword model dir"), (self.keywords_file, "keywords.txt")):
            if not Path(path).exists():
                gaps.append(f"{what} at {path}")
        return gaps

    @property
    def load_error(self) -> str:
        return self._load_error

    def load(self) -> bool:
        if self._spotter is not None:
            return True
        if self.missing:
            self._load_error = "; ".join(self.missing)
            log.info("kws_unavailable reason=%s", self._load_error)
            return False
        try:  # pragma: no cover - needs the real model
            import sherpa_onnx

            config = sherpa_onnx.KeywordSpotterConfig(
                model=sherpa_onnx.OnlineModelConfig(
                    transducer=sherpa_onnx.OnlineTransducerModelConfig(
                        encoder=str(self.model_dir / "encoder.onnx"),
                        decoder=str(self.model_dir / "decoder.onnx"),
                        joiner=str(self.model_dir / "joiner.onnx"),
                    ),
                    tokens=str(self.model_dir / "tokens.txt"),
                ),
                keywords_file=str(self.keywords_file),
                num_threads=1,
            )
            self._spotter = sherpa_onnx.KeywordSpotter(config)
            return True
        except Exception as exc:
            self._load_error = f"kws load failed: {exc}"
            return False

    # ── scoring ──────────────────────────────────────────────────────
    def _score(self, frame: Frame) -> tuple[str, float]:  # pragma: no cover - needs the model
        if self._spotter is None and not self.load():
            return "", 0.0
        import numpy as np

        samples = np.frombuffer(frame.tobytes(), dtype="<i2").astype("float32") / 32768.0
        stream = self._stream or self._spotter.create_stream()
        self._stream = stream
        stream.accept_waveform(sample_rate=16_000, waveform=samples)
        while self._spotter.is_ready(stream):
            self._spotter.decode_stream(stream)
            result = self._spotter.get_result(stream)
            if result:
                return result, 0.9
        return "", 0.0


class OpenWakeWordEngine(_ConfirmingWake):
    """openWakeWord — `hey_jarvis` ships pretrained; "atlas" is trained in L9.

    Kept behind the same contract as sherpa, because the plan calls for comparing
    them on real recordings rather than arguing about them.
    """

    name = "openwakeword"
    ram_mb = 45

    def __init__(self, *, model_path: str | Path = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model_path = Path(model_path) if model_path else None
        self._model: Any = None
        self._load_error = ""

    @staticmethod
    def runtime_available() -> bool:
        try:
            import openwakeword  # type: ignore[import-not-found]

            return hasattr(openwakeword, "Model")
        except ImportError:
            return False

    @property
    def missing(self) -> list[str]:
        gaps: list[str] = []
        if not self.runtime_available():
            gaps.append("openwakeword (pip install 'atlas-audio[openwakeword]')")
        if self.model_path and not self.model_path.exists():
            gaps.append(f"wake model at {self.model_path}")
        return gaps

    @property
    def load_error(self) -> str:
        return self._load_error

    def load(self) -> bool:
        if self._model is not None:
            return True
        if self.missing:
            self._load_error = "; ".join(self.missing)
            return False
        try:  # pragma: no cover - needs the real model
            from openwakeword.model import Model

            self._model = Model(wakeword_models=[str(self.model_path)] if self.model_path else None)
            return True
        except Exception as exc:
            self._load_error = f"openwakeword load failed: {exc}"
            return False

    def _score(self, frame: Frame) -> tuple[str, float]:  # pragma: no cover - needs the model
        if self._model is None and not self.load():
            return "", 0.0
        import numpy as np

        samples = np.frombuffer(frame.tobytes(), dtype="<i2")
        scores = self._model.predict(samples)
        if not scores:
            return "", 0.0
        keyword = max(scores, key=lambda name: scores[name])
        return keyword, float(scores[keyword])


class EnergyWakeEngine(_ConfirmingWake):
    """Loudness-based 'wake' — for tests and first-run demos, never for real use.

    It cannot tell "atlas" from a dropped spoon.  What it *can* do is drive the
    whole L2 pipeline on any machine with no models installed, which is exactly
    what the fixutres need.
    """

    name = "energy-wake"

    def __init__(self, *, rms_threshold: float = 0.08, **kwargs: Any) -> None:
        super().__init__(threshold=kwargs.pop("threshold", 0.5), **kwargs)
        self.rms_threshold = rms_threshold

    def _score(self, frame: Frame) -> tuple[str, float]:
        loudness = rms(frame)
        if loudness < self.rms_threshold:
            return "", loudness
        # Map loudness onto the 0..1 confidence scale the real engines use.
        score = min(1.0, loudness / (self.rms_threshold * 2))
        return self.keywords[0], score


def build_wake_engine(
    config: Any = None,
    *,
    keywords: Sequence[str] | None = None,
    prefer: str = "auto",
    log_sink: WakeLog | None = None,
) -> WakeWordEngine:
    """Pick the best available wake engine, and say which one it picked.

    `prefer="sherpa"` still degrades to energy rather than failing: the L2 gate
    is about the loop working, and a machine without ONNX models should still be
    able to run `atlas listen --ptt`.
    """
    threshold = getattr(getattr(config, "wake", None), "threshold", DEFAULT_THRESHOLD)
    confirm = getattr(getattr(config, "wake", None), "confirm_frames", DEFAULT_CONFIRM_FRAMES)
    keywords = tuple(
        keywords or getattr(getattr(config, "wake", None), "keywords", None) or ("atlas",)
    )

    candidates: list[WakeWordEngine] = []
    if prefer in {"auto", "sherpa"}:
        candidates.append(
            SherpaKwsEngine(keywords=keywords, threshold=threshold, confirm_frames=confirm, log_sink=log_sink)
        )
    if prefer in {"auto", "openwakeword"}:
        candidates.append(
            OpenWakeWordEngine(
                keywords=keywords, threshold=threshold, confirm_frames=confirm, log_sink=log_sink
            )
        )

    for candidate in candidates:
        load = getattr(candidate, "load", None)
        if load is None or load():
            return candidate
        log.info("wake_engine_skipped engine=%s reason=%s", candidate.name, getattr(candidate, "load_error", ""))

    log.warning("wake_engine_fallback engine=energy-wake — no real wake engine available")
    return EnergyWakeEngine(keywords=keywords, confirm_frames=confirm, log_sink=log_sink)


def wake_engine_status(config: Any = None) -> list[tuple[str, str]]:
    """(engine, status) rows for `atlas doctor` and `atlas listen --status`."""
    rows: list[tuple[str, str]] = []
    for engine in (
        SherpaKwsEngine(),
        OpenWakeWordEngine(),
        EnergyWakeEngine(),
    ):
        missing = getattr(engine, "missing", None)
        if missing is None:
            rows.append((engine.name, "available (test/demo only)"))
        else:
            rows.append((engine.name, "ready" if not missing else "missing " + ", ".join(missing)))
    return rows


__all__ = [
    "DEFAULT_CONFIRM_FRAMES",
    "DEFAULT_THRESHOLD",
    "EnergyWakeEngine",
    "OpenWakeWordEngine",
    "SherpaKwsEngine",
    "WakeHit",
    "WakeLog",
    "build_wake_engine",
    "wake_engine_status",
]
