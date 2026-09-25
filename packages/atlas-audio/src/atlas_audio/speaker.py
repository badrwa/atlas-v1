"""The verifier and the enrolment flow — the L4 engines.

Two objects, and the split matters:

* `SherpaSpeakerVerifier` answers "what does this voice look like as numbers?"
  through sherpa-onnx's speaker-embedding models (ERes2Net / 3D-Speaker family,
  ~90 MB, CPU, no torch).  The extractor is injectable, which is what lets the
  enrolment flow, the threshold sweep and every test in this package run in CI
  without the model.
* `EnrollmentSession` owns the *quality* of a profile: three clips, each with
  enough real speech, trimmed, embedded, and scored pairwise.  A profile is only
  as good as the worst pair in it, and a "yes, enrolled" that produced a bad
  profile is worse than asking again.

Why this file is in `atlas-audio` and not in `atlas-core`: it imports the audio
layer (frames, RMS, trimming) and it owns a model.  `atlas_core.identity` keeps
the policy (`ContextGuard`), the storage (`ProfileRepository`) and the log; this
keeps the signal processing.  Both sides of that line are tested independently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atlas_audio.frames import FRAME_MS, FRAME_SAMPLES, SAMPLE_RATE, pcm_to_frames, rms
from atlas_core.contracts import LanguageTag, ResourceCost, SpeakerMatch, SpeakerVerifier
from atlas_core.engines import LoadedFlag
from atlas_core.errors import Unsupported
from atlas_core.identity import (
    ContextGuard,
    IdentityConfig,
    ProfileRepository,
    SpeakerLog,
    SpeakerProfile,
    best_match,
)

log = logging.getLogger(__name__)

#: The plan's budget for the embedding model, and what a two-core laptop can
#: afford while ASR and the LLM are also awake.
DEFAULT_MODEL = "models/speaker/3dspeaker_speech_eres2net_base.onnx"
#: Below this there is not enough voice to compare — the plan's rule about the
#: television saying a single word.
MIN_EMBED_MS = 600.0


# ── signal handling ──────────────────────────────────────────────────
def trim_silence(samples: array, *, threshold: float = 0.012) -> array:
    """Drop leading and trailing silence, frame by frame.

    A trailing "mmm" from the microphone or the hiss of a room should not become
    part of a voice print: embeddings average over the whole clip, so silence is
    not neutral, it is a dilution.
    """
    frames = pcm_to_frames(samples.tobytes())
    first = next((index for index, frame in enumerate(frames) if rms(frame) >= threshold), None)
    if first is None:
        return array("h")
    last = len(frames) - 1
    while last > first and rms(frames[last]) < threshold:
        last -= 1
    start = first * FRAME_SAMPLES
    end = min((last + 1) * FRAME_SAMPLES, len(samples))
    return array("h", samples[start:end])


def speech_ms(samples: array, *, threshold: float = 0.012) -> float:
    """Milliseconds of *actual speech* in a clip (silence does not count)."""
    frames = pcm_to_frames(samples.tobytes())
    voiced = sum(1 for frame in frames if rms(frame) >= threshold)
    return voiced * FRAME_MS


def as_samples(audio: bytes | array) -> array:
    """Accept either form the rest of the audio layer uses (`FeedFrame` rule)."""
    if isinstance(audio, array):
        return audio
    samples = array("h")
    samples.frombytes(audio)
    return samples


def to_float_list(samples: array) -> list[float]:
    """int16 → floats in [-1, 1).  Stdlib only: the math is the same everywhere."""
    return [sample / 32768.0 for sample in samples]


def _best_of(embedding: list[float], window: list[list[float]]) -> float:
    """Highest similarity of one embedding against a window of others."""
    return max((best_match(embedding, SpeakerProfile(name="", embeddings=[sample])) for sample in window), default=0.0)


# ── the verifier ─────────────────────────────────────────────────────
class SherpaSpeakerVerifier(LoadedFlag, SpeakerVerifier):
    """Speaker embeddings through sherpa-onnx.

    `extractor` is injected in tests (a callable `(samples, sample_rate) -> list`
    or any object with `compute`) and `model_factory` in production.  The class
    never imports numpy at module scope: the audio core keeps working with
    `array("h")` and only the model call pays for numpy.
    """

    name = "speaker-verify"
    ram_mb = 90
    cold_start_s = 1.5

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        *,
        num_threads: int = 2,
        provider: str = "cpu",
        extractor: Any = None,
        model_factory: Callable[[str, int, str], Any] | None = None,
        min_embed_ms: float = MIN_EMBED_MS,
    ) -> None:
        self.model_path = str(model_path)
        self.num_threads = num_threads
        self.provider = provider
        self._extractor = extractor
        self._factory = model_factory
        self.min_embed_ms = min_embed_ms
        self.load_error = ""
        self.last_ms = 0.0
        self.embeddings = 0

    # ── engine lifecycle ─────────────────────────────────────────────
    async def load(self) -> None:
        if self._extractor is not None:
            self._loaded = True
            return
        try:
            factory = self._factory or _build_extractor
            self._extractor = await asyncio.to_thread(
                factory, self.model_path, self.num_threads, self.provider
            )
            self.load_error = ""
            self._loaded = True
            log.info("speaker_model_loaded model=%s", self.model_path)
        except Exception as exc:
            self.load_error = str(exc)
            self._extractor = None
            self._loaded = False
            log.warning("speaker_model_failed model=%s error=%s", self.model_path, exc)

    async def unload(self) -> None:
        """Release the model.  An injected extractor is released too — it is ours."""
        self._extractor = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._extractor is not None

    def cost_hint(self) -> ResourceCost:
        return ResourceCost(ram_mb=type(self).ram_mb, cold_start_s=type(self).cold_start_s)

    def available(self) -> bool:
        """Cheap precondition — an injected extractor, a model file, or a factory."""
        if self._extractor is not None:
            return True
        if self._factory is not None:
            return True
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return False
        return Path(self.model_path).exists()

    def missing(self) -> str:
        if self.available() or self.is_loaded():
            return ""
        try:
            import sherpa_onnx  # noqa: F401
        except ImportError:
            return "sherpa-onnx is not installed — pip install 'atlas-audio[local]'"
        return (
            f"speaker model not found at {self.model_path} — download a 3D-Speaker/ERes2Net "
            "ONNX model into models/speaker/ (see docs/levels/NOTES-L04.md)"
        )

    # ── the contract ─────────────────────────────────────────────────
    def embed(self, audio: bytes | array, *, sample_rate: int = SAMPLE_RATE) -> list[float]:
        """One embedding per utterance (or clip).  Deterministic, and cheap."""
        started = time.perf_counter()
        samples = as_samples(audio)
        if sample_rate != SAMPLE_RATE:
            # 16 kHz is what these models were trained on; anything else is a bug
            # in the caller, and silently accepting it would produce a profile
            # that matches nothing.
            raise Unsupported(
                f"speaker embeddings need {SAMPLE_RATE} Hz audio (got {sample_rate})"
            )
        duration = len(samples) / SAMPLE_RATE * 1000
        if duration < self.min_embed_ms:
            raise Unsupported(
                f"too short to identify: {duration:.0f} ms of audio, need {self.min_embed_ms:.0f} ms"
            )
        if self._extractor is None:
            raise Unsupported(self.missing() or "speaker model is not loaded")

        vector = self._compute(samples)
        self.last_ms = (time.perf_counter() - started) * 1000
        self.embeddings += 1
        return [float(value) for value in vector]

    def verify(
        self, audio: bytes | array, profile: list[list[float]], *, threshold: float = 0.65
    ) -> SpeakerMatch:
        """Score one utterance against a list of embeddings (a profile's window)."""
        embedding = self.embed(audio)
        score = _best_of(embedding, profile)
        return SpeakerMatch(name="", score=score, owner=score >= threshold)

    def match_profile(
        self, audio: bytes | array, profile: SpeakerProfile, *, threshold: float = 0.65
    ) -> SpeakerMatch:
        """Like `verify`, but keeps the profile's name — what the loop uses."""
        embedding = self.embed(audio)
        score = best_match(embedding, profile)
        return SpeakerMatch(name=profile.name, score=score, owner=profile.owner and score >= threshold)

    # ── internals ────────────────────────────────────────────────────
    def _compute(self, samples: array) -> Iterable[float]:
        extractor = self._extractor
        if hasattr(extractor, "compute"):
            return extractor.compute(samples, SAMPLE_RATE)
        return extractor(samples, SAMPLE_RATE)

    def embed_or_none(self, audio: bytes | array) -> list[float] | None:
        """`embed` for callers that treat "cannot identify" as a normal outcome."""
        try:
            return self.embed(audio)
        except Unsupported as exc:
            log.info("speaker_embed_skipped reason=%s", exc)
            return None

    def status(self) -> dict[str, object]:
        return {
            "name": self.name,
            "model": self.model_path,
            "available": self.available(),
            "loaded": self.is_loaded(),
            "missing": self.missing(),
            "ram_mb": self.ram_mb,
            "last_ms": round(self.last_ms, 1),
            "embeddings": self.embeddings,
            "error": self.load_error,
        }


def _build_extractor(model_path: str, num_threads: int, provider: str) -> Any:
    """The real sherpa-onnx extractor.  numpy is imported *here*, not above."""
    import numpy as np
    import sherpa_onnx

    if not Path(model_path).exists():
        raise FileNotFoundError(model_path)
    config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=model_path, num_threads=num_threads, provider=provider
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
    if not extractor.is_ready():
        raise RuntimeError(f"sherpa-onnx could not load {model_path}")

    class _Extractor:
        """One adapter, so the rest of the file does not know sherpa's API."""

        def compute(self, samples: array, sample_rate: int) -> list[float]:
            stream = extractor.create_stream()
            floats = np.asarray(to_float_list(samples), dtype=np.float32)
            stream.accept_waveform(sample_rate, floats)
            stream.input_finished()
            return list(extractor.compute(stream))

    return _Extractor()


# ── enrolment ────────────────────────────────────────────────────────
@dataclass(slots=True)
class EnrollmentStep:
    """The answer to one enrolment sample — what the CLI reads out loud."""

    index: int
    ok: bool
    reason: str = ""
    speech_ms: float = 0.0
    score: float = 0.0
    spoken: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "ok": self.ok,
            "reason": self.reason,
            "speech_ms": round(self.speech_ms, 1),
            "spoken": self.spoken,
        }


#: What the owner is asked to read.  Varied on purpose: a profile built from one
#: sentence said identically three times is a profile of that sentence.
ENROL_PROMPTS: dict[str, tuple[str, ...]] = {
    "ar-MA": (
        "Goul: “Salam, smiti … , w hada sowti l Atlas. Kanhder b Darija.”",
        "Daba 3awed besshot: “Lyoum ljaw mzyan, bghit nsejjel had l'compte.”",
        "W daba besshot akhir: “Atlas, 3refti sowti? Khellini nkellem m3ak.”",
    ),
    "en-GB": (
        "Say: “Hello, this is my voice. Atlas and I are getting introduced.”",
        "Now a little differently: “Today is a good day, and I am enrolling my voice.”",
        "And one more time: “Atlas, you know my voice. Let us get to work.”",
    ),
}


class EnrollmentSession:
    """Three clips, one profile — and a refusal when the clips are not good enough.

    The quality gate is the difference between an assistant that recognises you
    and one that recognises everyone: pairwise cosine across the samples catches
    "I enrolled in a noisy room with music on" *before* it becomes a profile, and
    the spoken message tells the person what to fix.
    """

    def __init__(
        self,
        name: str,
        *,
        verifier: SherpaSpeakerVerifier,
        config: IdentityConfig | None = None,
        language: LanguageTag | str = "ar-MA",
    ) -> None:
        self.name = name.strip()
        self.verifier = verifier
        config = config or IdentityConfig()
        self.required = max(1, config.enrol_samples)
        self.min_speech_ms = config.enrol_min_speech_ms
        self.quality_floor = config.quality_floor
        self.language = language
        self.steps: list[EnrollmentStep] = []
        self.embeddings: list[list[float]] = []
        self.started_at = time.time()

    # ── progress ─────────────────────────────────────────────────────
    @property
    def index(self) -> int:
        return len(self.embeddings)

    @property
    def missing(self) -> int:
        return max(0, self.required - self.index)

    @property
    def complete(self) -> bool:
        return self.index >= self.required

    def prompt(self, index: int | None = None) -> str:
        prompts = ENROL_PROMPTS.get(str(self.language), ENROL_PROMPTS["ar-MA"])
        index = self.index if index is None else index
        return prompts[index % len(prompts)].replace("…", self.name or "…")

    @property
    def quality(self) -> float:
        """Worst pairwise similarity between the samples — the honest number.

        An average would hide one bad clip, and one bad clip is enough to make
        the profile reject the owner on a quiet day.
        """
        if len(self.embeddings) < 2:
            return 0.0
        worst = 1.0
        for index, left in enumerate(self.embeddings):
            for right in self.embeddings[index + 1 :]:
                worst = min(worst, _best_of(left, [right]))
        return worst

    # ── samples ──────────────────────────────────────────────────────
    def add(self, audio: bytes | array) -> EnrollmentStep:
        """Take one sample: trim, measure, embed, and say what happened."""
        index = self.index
        samples = trim_silence(as_samples(audio))
        voiced = speech_ms(samples)
        if voiced < self.min_speech_ms:
            step = EnrollmentStep(
                index=index,
                ok=False,
                reason="not_enough_speech",
                speech_ms=voiced,
                spoken=self._spoken("short", extra=f"{voiced:.0f}"),
            )
            self.steps.append(step)
            return step
        embedding = self.verifier.embed_or_none(samples)
        if embedding is None:
            step = EnrollmentStep(
                index=index,
                ok=False,
                reason="no_embedding",
                speech_ms=voiced,
                spoken=self._spoken("failed"),
            )
            self.steps.append(step)
            return step

        self.embeddings.append(embedding)
        score = self.quality
        step = EnrollmentStep(index=index, ok=True, speech_ms=voiced, score=score, spoken="")
        self.steps.append(step)
        if self.complete and score < self.quality_floor:
            step.spoken = self._spoken("quality", extra=f"{score:.2f}")
        elif self.complete:
            step.spoken = self._spoken("done", extra=str(index + 1))
        else:
            step.spoken = self._spoken("next")
        return step

    def add_many(self, clips: Iterable[bytes | array]) -> list[EnrollmentStep]:
        return [self.add(clip) for clip in clips if self.index < self.required]

    # ── finishing ────────────────────────────────────────────────────
    def profile(self, *, force: bool = False) -> SpeakerProfile | None:
        """The profile, or `None` when the samples did not clear the floor."""
        if not self.complete:
            return None
        quality = self.quality
        if quality < self.quality_floor and not force:
            return None
        return SpeakerProfile(
            name=self.name,
            embeddings=[list(vector) for vector in self.embeddings],
            quality=quality,
            samples=self.index,
            note=f"enrolled in {self.required} samples",
        )

    def problems(self) -> list[str]:
        """Why this session cannot produce a profile yet — for `atlas identity`."""
        problems: list[str] = []
        if self.missing:
            problems.append(f"{self.missing} more sample(s) needed")
        if self.complete and self.quality < self.quality_floor:
            problems.append(
                f"quality {self.quality:.2f} is below the floor {self.quality_floor:.2f} — "
                "quiet room, same microphone, speak normally"
            )
        return problems

    def status(self) -> dict[str, object]:
        return {
            "name": self.name,
            "required": self.required,
            "collected": self.index,
            "quality": round(self.quality, 3),
            "floor": self.quality_floor,
            "min_speech_ms": self.min_speech_ms,
            "complete": self.complete,
            "problems": self.problems(),
        }

    def _spoken(self, kind: str, *, extra: str = "") -> str:
        english = str(self.language).startswith("en")
        if kind == "short":
            return (
                f"Too short — {float(extra):.0f} ms of speech, I need "
                f"{self.min_speech_ms:.0f} ms. Try again."
                if english
                else f"Qssir bezzaf — {float(extra):.0f} ms. 3awed, hder chwiya ktar."
            )
        if kind == "failed":
            return (
                "I could not read that voice clip."
                if english
                else "Ma 9ditch n9ra had sowt."
            )
        if kind == "quality":
            return (
                f"The samples disagree with each other ({extra}) — quieter room, please."
                if english
                else f"L3inayat machi mzyanin ({extra}) — 3awed f blasa hdya."
            )
        if kind == "done":
            return (
                f"Enrolled, {self.name}. I know your voice now."
                if english
                else f"Safi {self.name}, 3reft sowtek daba."
            )
        return (
            f"Got it — {self.index} of {self.required}. Next sentence."
            if english
            else f"Wakha — {self.index} men {self.required}. Ljoumla li man ba3d."
        )


# ── factory ──────────────────────────────────────────────────────────
def build_verifier(
    config: Any = None,
    *,
    extractor: Any = None,
    model_path: str | None = None,
) -> SherpaSpeakerVerifier | None:
    """The verifier this machine can actually run, or `None` when identity is off.

    `None` is an answer, not a failure: with `[identity] enabled = false` Atlas
    behaves exactly like L3 (single user, full capabilities) and nothing loads.
    """
    identity = IdentityConfig.from_config(config)
    if not identity.enabled:
        return None
    path = model_path or str(
        getattr(getattr(config, "identity", None), "model_path", DEFAULT_MODEL) or DEFAULT_MODEL
    )
    return SherpaSpeakerVerifier(path, extractor=extractor)


def build_identity(
    config: Any = None,
    *,
    verifier: Any = None,
    repository: ProfileRepository | None = None,
    log: SpeakerLog | None = None,
    extractor: Any = None,
) -> dict[str, Any]:
    """Everything one process needs for identity: guard, store, verifier, log."""
    identity = IdentityConfig.from_config(config)
    repository = repository or ProfileRepository(identity.profiles_path, window=identity.window)
    log = log if log is not None else SpeakerLog(identity.log_path)
    verifier = verifier or build_verifier(config, extractor=extractor)
    guard = ContextGuard(identity, repository=repository, log=log)
    return {
        "config": identity,
        "verifier": verifier,
        "repository": repository,
        "log": log,
        "guard": guard,
        "owner": guard.owner_name,
    }


def speaker_status(config: Any = None) -> list[tuple[str, str]]:
    """Rows for `atlas doctor` and `atlas identity status`."""
    identity = IdentityConfig.from_config(config)
    rows: list[tuple[str, str]] = [
        ("identity", "enabled" if identity.enabled else "disabled — single user, full access"),
    ]
    if not identity.enabled:
        return rows
    verifier = build_verifier(config)
    if verifier is None:  # pragma: no cover - guarded above
        return rows
    rows.append(("verifier", "available" if verifier.available() else verifier.missing()))
    repository = ProfileRepository(identity.profiles_path, window=identity.window)
    stats = repository.stats()
    rows.append(
        (
            "profiles",
            f"{stats['people']} people · owner {stats['owner'] or '—'} · "
            f"{stats['samples']} samples · dim {stats['dimension'] or '—'}",
        )
    )
    rows.append(
        ("threshold", f"{identity.threshold:.2f} (trust needs ≥ {identity.trust_min_ms / 1000:.1f}s)")
    )
    return rows


__all__ = [
    "DEFAULT_MODEL",
    "ENROL_PROMPTS",
    "MIN_EMBED_MS",
    "EnrollmentSession",
    "EnrollmentStep",
    "SherpaSpeakerVerifier",
    "as_samples",
    "build_identity",
    "build_verifier",
    "speaker_status",
    "speech_ms",
    "to_float_list",
    "trim_silence",
]
