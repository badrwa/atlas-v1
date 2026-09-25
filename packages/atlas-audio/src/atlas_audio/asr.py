"""Speech recognition: cloud first, local when the wire is cut.

Three implementations behind one contract (`SpeechRecognizer.transcribe`):

* `CloudRecognizer` — two backends in one class, because they differ only in
  request shape: **Gemini native audio** (best measured Darija accuracy) and
  **Groq whisper-large-v3-turbo** (fast, huge free quota).  Which one served a
  transcript is recorded, not guessed.
* `LocalWhisperRecognizer` — faster-whisper INT8, the Bunker path.  ~1.5 GB
  resident for a small model, so it is *leased*, never kept warm by accident.
* `SherpaOfflineRecognizer` — sherpa-onnx offline, for when an ONNX Darija
  export exists (MoulSot/Qwen3-ASR).  Honest stub until then.

`RecognizerFactory` owns the policy from the plan:

    Darija → cloud (online + quota) → local Darija → multilingual small
    English → cloud (online) → small.en → multilingual small

and the promise that matters on an 8 GB laptop: **never two ASR engines at
once.**  The lease is the mechanism, not a comment.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from atlas_core.contracts import (
    Engine,
    HealthReport,
    LanguageTag,
    ResourceCost,
    SpeechRecognizer,
    Transcript,
)
from atlas_core.engines import LoadedFlag
from atlas_core.errors import ProviderUnavailable, RateLimited, Unsupported
from atlas_core.http import OwnedHttpClient
from atlas_core.resources import ResourceLease

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 6.0


def wav_header(pcm: bytes, *, sample_rate: int = 16_000, channels: int = 1) -> bytes:
    """Wrap raw PCM16 in a WAV container — several APIs insist on one."""
    import struct

    bits = 16
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    data_size = len(pcm)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
        + b"data"
        + struct.pack("<I", data_size)
        + pcm
    )


@dataclass(slots=True)
class AsrAttempt:
    """One backend's outcome — kept so failures are visible in the notes."""

    engine: str
    ok: bool
    ms: float
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"engine": self.engine, "ok": self.ok, "ms": round(self.ms, 1), "detail": self.detail}


class CloudRecognizer(LoadedFlag, SpeechRecognizer, OwnedHttpClient):
    """Cloud ASR with a same-class backup backend and an honest fallback chain."""

    name = "cloud-asr"

    def __init__(
        self,
        *,
        gemini_key: str = "",
        groq_key: str = "",
        gemini_model: str = "gemini-2.5-flash",
        groq_model: str = "whisper-large-v3-turbo",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
        hotwords: Callable[[], str] | None = None,
        offline_check: Callable[[], bool] | None = None,
        order: tuple[str, ...] = ("gemini", "groq"),
    ) -> None:
        self.order = tuple(order) or ("gemini", "groq")
        self.gemini_key = gemini_key
        self.groq_key = groq_key
        self.gemini_model = gemini_model
        self.groq_model = groq_model
        OwnedHttpClient.__init__(self, client, timeout_s=timeout_s)
        # The Darija lexicon is passed as a hotword prompt: names, apps and
        # places are what ASR gets wrong, and they are exactly what we know.
        self._hotwords = hotwords or (lambda: "")
        self._offline_check = offline_check
        self.attempts: list[AsrAttempt] = []
        self._loaded = False

    # ── plumbing ─────────────────────────────────────────────────────
    # load/unload/is_loaded come from LoadedFlag: a cloud recogniser has nothing
    # to load, but it still has to answer the `Engine` contract honestly.
    def cost_hint(self) -> ResourceCost:
        return ResourceCost(ram_mb=5, cold_start_s=0.0)

    @property
    def backends(self) -> list[str]:
        """Configured backends, in the order `asr.cloud_order` asks for."""
        available = {"gemini": bool(self.gemini_key), "groq": bool(self.groq_key)}
        return [name for name in self.order if available.get(name)]

    async def health(self) -> HealthReport:
        if not self.backends:
            return HealthReport(ok=False, detail="no cloud ASR key configured")
        if self._offline_check and self._offline_check():
            return HealthReport(ok=False, detail="offline")
        return HealthReport(ok=True, detail="backends: " + ", ".join(self.backends))

    # ── the contract ─────────────────────────────────────────────────
    async def transcribe(
        self, audio: bytes, *, language: LanguageTag = "unknown", sample_rate: int = 16_000
    ) -> Transcript:
        if self._offline_check and self._offline_check():
            raise ProviderUnavailable("cloud ASR offline")

        errors: list[str] = []
        rate_limited = 0
        for backend in self.backends:
            started = time.perf_counter()
            try:
                if backend == "gemini":
                    result = await self._gemini(audio, language=language, sample_rate=sample_rate)
                else:
                    result = await self._groq(audio, language=language, sample_rate=sample_rate)
            except (ProviderUnavailable, RateLimited) as exc:
                if isinstance(exc, RateLimited):
                    rate_limited += 1
                errors.append(f"{backend}: {exc}")
                self.attempts.append(
                    AsrAttempt(backend, False, (time.perf_counter() - started) * 1000, str(exc))
                )
                continue
            except httpx.HTTPError as exc:
                errors.append(f"{backend}: {exc}")
                self.attempts.append(
                    AsrAttempt(backend, False, (time.perf_counter() - started) * 1000, str(exc))
                )
                continue

            ms = (time.perf_counter() - started) * 1000
            self.attempts.append(AsrAttempt(backend, True, ms))
            log.info("asr_cloud engine=%s language=%s ms=%.0f chars=%s", backend, language, ms, len(result.text))
            return Transcript(
                text=result.text,
                language=result.language or (language if language != "unknown" else "unknown"),
                confidence=result.confidence,
                engine=f"cloud:{backend}",
                duration_ms=ms,
            )

        detail = "all cloud ASR backends failed — " + "; ".join(errors or ["none configured"])
        if self.backends and rate_limited == len(self.backends):
            # Telling the truth here is what lets Atlas say "quota" instead of
            # "something went wrong", and lets the factory reach for a local model.
            raise RateLimited(detail)
        raise ProviderUnavailable(detail)

    # ── backends ─────────────────────────────────────────────────────
    async def _gemini(
        self, audio: bytes, *, language: LanguageTag, sample_rate: int
    ) -> Transcript:
        prompt = (
            "Transcribe this audio exactly as spoken. It is Moroccan Darija or British "
            "English. Reply with the transcript only — no translation, no commentary."
        )
        if hotwords := self._hotwords():
            prompt += f"\nNames and words to expect: {hotwords}"

        payload: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "audio/wav",
                                "data": _b64(wav_header(audio, sample_rate=sample_rate)),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 256},
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent"
        )
        response = await self.client.post(
            url, headers={"x-goog-api-key": self.gemini_key}, json=payload
        )
        if response.status_code in (401, 403):
            raise ProviderUnavailable(f"gemini asr auth ({response.status_code})")
        if response.status_code == 429:
            raise RateLimited("gemini asr rate limited")
        if response.status_code >= 400:
            raise ProviderUnavailable(f"gemini asr http {response.status_code}")

        body = response.json()
        text = ""
        for candidate in body.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                text += part.get("text", "")
        return Transcript(text=text.strip(), language=language if language != "unknown" else "unknown",
                          confidence=0.8)

    async def _groq(self, audio: bytes, *, language: LanguageTag, sample_rate: int) -> Transcript:
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        # Groq applies Whisper's ISO-639-1 hint; a wrong hint is worse than none.
        hint = {"ar-MA": "ar", "en-GB": "en"}.get(str(language), "")
        data: dict[str, str] = {"model": self.groq_model, "response_format": "json", "temperature": "0"}
        if hint:
            data["language"] = hint
        if hotwords := self._hotwords():
            data["prompt"] = f"Names and words to expect: {hotwords}"

        files = {
            "file": ("utterance.wav", wav_header(audio, sample_rate=sample_rate), "audio/wav"),
        }
        response = await self.client.post(
            url, headers={"Authorization": f"Bearer {self.groq_key}"}, data=data, files=files
        )
        if response.status_code in (401, 403):
            raise ProviderUnavailable(f"groq asr auth ({response.status_code})")
        if response.status_code == 429:
            raise RateLimited("groq asr rate limited")
        if response.status_code >= 400:
            raise ProviderUnavailable(f"groq asr http {response.status_code}")

        body = response.json()
        confidence = 0.8
        if segments := body.get("segments"):
            no_speech = [seg.get("no_speech_prob", 0.0) for seg in segments]
            confidence = max(0.0, min(1.0, 1.0 - sum(no_speech) / len(no_speech)))
        return Transcript(
            text=str(body.get("text", "")).strip(),
            language=language if language != "unknown" else "unknown",
            confidence=confidence,
        )


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


class LocalWhisperRecognizer(SpeechRecognizer):
    """faster-whisper INT8 on the CPU — the offline path.

    Model choice is a *policy* decision, not a constant: a Darija fine-tune for
    Moroccan speech, `small.en` for English, and a multilingual small model as
    the last resort.  `compute_type="int8"` and `cpu_threads=3` (two cores, four
    threads: leave one for the rest of the system) come straight from the plan's
    hardware verdict.
    """

    name = "faster-whisper"
    ram_mb = 1200  # small-int8 resident; the lease is what keeps this honest

    def __init__(
        self,
        model_path: str | Path = "models/whisper-darija-ct2",
        *,
        language: LanguageTag = "ar-MA",
        cpu_threads: int = 3,
        compute_type: str = "int8",
        beam_size: int = 1,
        initial_prompt: str = "",
        model_factory: Callable[[], Any] | None = None,
        ram_mb: int | None = None,
        key: str = "",
    ) -> None:
        self.model_path = str(model_path)
        # The lease keys engines by `name`, and this class is instantiated once
        # per model — three instances sharing one name would collide in
        # ResourceLease.entries and evict each other's bookkeeping.
        self.key = key or Path(self.model_path).stem or "whisper"
        self.name = f"whisper:{self.key}"
        self.language = language
        self.cpu_threads = cpu_threads
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.initial_prompt = initial_prompt
        self._factory = model_factory
        self._model: Any = None
        self.ram_mb = ram_mb if ram_mb is not None else type(self).ram_mb
        self.load_error = ""

    # ── Engine contract ──────────────────────────────────────────────
    def is_loaded(self) -> bool:
        return self._model is not None

    def cost_hint(self) -> ResourceCost:
        return ResourceCost(ram_mb=self.ram_mb, cpu_threads=self.cpu_threads, cold_start_s=6.0)

    async def load(self) -> None:
        if self._model is not None:
            return
        if self._factory is not None:  # injected (tests, or a preloaded model)
            self._model = self._factory()
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise Unsupported(
                "faster-whisper is not installed — run: pip install 'atlas-audio[local]'"
            ) from exc

        if _is_missing_local_model(self.model_path):
            raise Unsupported(
                f"local ASR model not found at {self.model_path} — see docs/levels/NOTES-L02.md"
            )
        self._model = WhisperModel(
            self.model_path,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
        )
        log.info("asr_local_loaded model=%s ram_mb=%s", self.model_path, self.ram_mb)

    async def unload(self) -> None:
        self._model = None

    async def health(self) -> HealthReport:
        if self.is_loaded():
            return HealthReport(ok=True, detail=f"{Path(self.model_path).name} ({self.compute_type})")
        if self.load_error:
            return HealthReport(ok=False, detail=self.load_error)
        try:
            import faster_whisper  # type: ignore[import-not-found]  # noqa: F401

            return HealthReport(ok=True, detail=f"{Path(self.model_path).name} available")
        except ImportError:
            return HealthReport(ok=False, detail="faster-whisper not installed")

    # ── the contract ─────────────────────────────────────────────────
    async def transcribe(
        self, audio: bytes, *, language: LanguageTag = "unknown", sample_rate: int = 16_000
    ) -> Transcript:
        if self._model is None:
            await self.load()
        started = time.perf_counter()
        import asyncio

        # faster-whisper is synchronous and CPU-bound: run it off the event loop
        # so the UI and the wake engine keep breathing while it thinks.
        text, detected, confidence = await asyncio.to_thread(
            self._decode, audio, language, sample_rate
        )
        ms = (time.perf_counter() - started) * 1000
        log.info("asr_local ms=%.0f chars=%s", ms, len(text))
        return Transcript(
            text=text,
            language=detected,
            confidence=confidence,
            engine=f"local:{Path(self.model_path).name}",
            duration_ms=ms,
        )

    def _decode(
        self, audio: bytes, language: LanguageTag, sample_rate: int = 16_000
    ) -> tuple[str, LanguageTag, float]:
        import numpy as np

        samples = np.frombuffer(audio, dtype="<i2").astype("float32") / 32768.0
        # A 30 s cap: nobody talks that long in one breath, and the VAD already
        # cuts at 12 s — this only guards against a corrupt buffer.
        samples = samples[: max(1, sample_rate) * 30]
        segments, info = self._model.transcribe(
            samples,
            language={"ar-MA": "ar", "en-GB": "en"}.get(str(language)),
            beam_size=self.beam_size,
            # Each utterance is independent: carrying context across turns made
            # Atlas repeat the previous sentence when the audio was clipped.
            condition_on_previous_text=False,
            vad_filter=False,  # we already segmented — doing it twice loses words
            initial_prompt=self.initial_prompt or None,
        )
        parts: list[str] = []
        probabilities: list[float] = []
        for segment in segments:
            parts.append(segment.text)
            probabilities.append(float(getattr(segment, "avg_logprob", -1.0)))

        text = "".join(parts).strip()
        probability = sum(probabilities) / len(probabilities) if probabilities else -1.0
        confidence = max(0.0, min(1.0, 1.0 + probability))  # logprob 0 → 1.0
        detected = _language_from_whisper(getattr(info, "language", ""), text, language)
        return text, detected, confidence


class SherpaOfflineRecognizer(LoadedFlag, SpeechRecognizer):
    """sherpa-onnx offline ASR — the single-runtime path for Bunker mode.

    Waiting on an ONNX Darija export (L9).  Until one exists this class reports
    exactly what is missing rather than pretending to hear.
    """

    name = "sherpa-asr"

    def __init__(self, model_dir: str | Path = "models/sherpa-asr", **_: Any) -> None:
        self.model_dir = Path(model_dir)
        self._loaded = False

    @property
    def missing(self) -> list[str]:
        gaps = []
        try:
            import sherpa_onnx  # type: ignore[import-not-found]

            if not hasattr(sherpa_onnx, "OfflineRecognizer"):
                gaps.append("sherpa-onnx offline recogniser")
        except ImportError:
            gaps.append("sherpa-onnx (pip install 'atlas-audio[local]')")
        if not self.model_dir.exists():
            gaps.append(f"ONNX ASR model at {self.model_dir} (see L9)")
        return gaps

    def is_loaded(self) -> bool:
        return self._loaded

    def cost_hint(self) -> ResourceCost:
        return ResourceCost(ram_mb=800, cpu_threads=3, cold_start_s=5.0)

    async def load(self) -> None:
        if gaps := self.missing:
            raise Unsupported("sherpa ASR unavailable: " + "; ".join(gaps))
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def health(self) -> HealthReport:
        gaps = self.missing
        return HealthReport(ok=not gaps, detail="; ".join(gaps) or "ready")

    async def transcribe(
        self, audio: bytes, *, language: LanguageTag = "unknown", sample_rate: int = 16_000
    ) -> Transcript:
        if gaps := self.missing:
            raise Unsupported("sherpa ASR unavailable: " + "; ".join(gaps))
        raise Unsupported("sherpa ASR transcription lands with the L9 ONNX export")  # pragma: no cover


# ── policy ───────────────────────────────────────────────────────────
MODES = ("cloud_first", "local_first", "cloud_only", "local_only")


@dataclass
class RecognizerPolicy:
    """The one place engine preference is written down.

    Darija: the community Darija fine-tune first when offline, then a
    multilingual small.  English: the English-only small model, which is
    markedly better than multilingual for en-GB, then multilingual.  `cloud` is
    dropped entirely in `local_only` rather than tried and failed — the point of
    Bunker mode is that no audio leaves the machine.
    """

    mode: str = "cloud_first"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown ASR mode {self.mode!r} — expected one of {MODES}")

    @classmethod
    def from_config(cls, config: Any = None) -> RecognizerPolicy:
        """`[asr]` → the effective policy, `cloud_audio = false` included.

        Both `build_recognizers` and `atlas listen --status` go through here, so
        the mode the CLI prints is the mode the factory will actually use.
        """
        asr = getattr(config, "asr", None)
        mode = getattr(asr, "mode", "cloud_first")
        if not bool(getattr(asr, "cloud_audio", True)):
            mode = "local_only"
        return cls(mode=mode)

    @property
    def prefer_cloud(self) -> bool:
        return self.mode in ("cloud_first", "cloud_only")

    def locals_for(self, language: LanguageTag) -> list[str]:
        if language == "en-GB":
            return ["local:english", "local:multilingual"]
        return ["local:darija", "local:multilingual"]

    def order(self, language: LanguageTag) -> list[str]:
        """Ordered preference list — first available wins."""
        local = self.locals_for(language)
        if self.mode == "cloud_only":
            return ["cloud"]
        if self.mode == "local_only":
            return local
        if self.mode == "local_first":
            return [*local, "cloud"]
        return ["cloud", *local]


class RecognizerFactory:
    """Builds the right recogniser, leasing local engines so only one is warm.

    The 8 GB rule is enforced here: acquiring a second local ASR engine releases
    the first.  `ResourceLease` decides whether it even *fits* — a lease denial
    is a normal outcome that falls back to the cloud, not an exception a user
    ever sees.
    """

    def __init__(
        self,
        *,
        cloud: SpeechRecognizer | None = None,
        local_models: Mapping[str, SpeechRecognizer] | None = None,
        lease: ResourceLease | None = None,
        online: Callable[[], bool] | None = None,
        policy: RecognizerPolicy | None = None,
    ) -> None:
        self.cloud = cloud
        self.local_models: dict[str, SpeechRecognizer] = dict(local_models or {})
        self.lease = lease or ResourceLease()
        self._online = online or (lambda: True)
        self.policy = policy or RecognizerPolicy()
        self.last_choice = ""
        self.denials: list[str] = []

    # ── selection ────────────────────────────────────────────────────
    @property
    def offline(self) -> bool:
        """The policy knows whether the cloud may be tried at all."""
        return not (self.policy.prefer_cloud and self._online())

    def candidates(self, language: LanguageTag) -> list[str]:
        return [name for name in self.policy.order(language) if self._exists(name)]

    def _exists(self, name: str) -> bool:
        if name == "cloud":
            return self.cloud is not None
        return name in self.local_models

    async def choose(self, language: LanguageTag) -> SpeechRecognizer:
        """First candidate that is usable right now, cloud included."""
        errors: list[str] = []
        for name in self.candidates(language):
            if name == "cloud":
                if not self._online():
                    errors.append("cloud: offline")
                    continue
                assert self.cloud is not None
                self.last_choice = "cloud"
                return self.cloud

            recognizer = self.local_models[name]
            try:
                await self._acquire(recognizer)
            except Exception as exc:
                self.denials.append(f"{name}: {exc}")
                errors.append(f"{name}: {exc}")
                continue
            self.last_choice = name
            return recognizer

        raise Unsupported("no ASR engine available — " + "; ".join(errors or ["nothing configured"]))

    async def _acquire(self, recognizer: Engine) -> None:
        """Load `recognizer`, unloading any other local engine first."""
        for name, other in self.local_models.items():
            if other is recognizer:
                continue
            if self.lease.is_loaded(other):
                await self.lease.release(other)
                log.info("asr_evicted engine=%s to make room for %s", name, recognizer.name)
        await self.lease.acquire(recognizer)

    async def transcribe(
        self, audio: bytes, *, language: LanguageTag = "unknown", sample_rate: int = 16_000
    ) -> Transcript:
        recognizer = await self.choose(language)
        try:
            return await recognizer.transcribe(audio, language=language, sample_rate=sample_rate)
        except (ProviderUnavailable, RateLimited) as exc:
            # The wire died mid-conversation: fall back to whatever is local.
            log.warning("asr_cloud_failed error=%s — trying local", exc)
            self.denials.append(f"cloud: {exc}")
            for name, local in self.local_models.items():
                if local is recognizer:
                    continue
                try:
                    await self._acquire(local)
                except Exception:
                    continue
                self.last_choice = name
                return await local.transcribe(audio, language=language, sample_rate=sample_rate)
            raise

    async def aclose(self) -> None:
        await self.lease.release_all()
        if self.cloud is not None and hasattr(self.cloud, "aclose"):
            await self.cloud.aclose()  # type: ignore[attr-defined]

    def snapshot(self) -> dict[str, object]:
        return {
            "cloud": self.cloud.name if self.cloud else None,
            "local": sorted(self.local_models),
            "loaded": self.lease.loaded_names(),
            "resident_mb": self.lease.resident_mb(),
            "last_choice": self.last_choice,
            "denials": self.denials[-5:],
        }


def build_recognizers(config: Any, *, lexicon_prompt: Callable[[], str] | None = None) -> dict[str, Any]:
    """Everything `atlas listen` needs, built from config, missing pieces included."""
    from atlas_audio.postprocess import AsrPostProcessor

    gemini = next((p for p in getattr(config, "providers", []) if p.name == "gemini"), None)
    groq = next((p for p in getattr(config, "providers", []) if p.name == "groq"), None)
    processor = AsrPostProcessor.from_config(config)
    hotwords = lexicon_prompt or (lambda: processor.lexicon_prompt())

    asr_config = getattr(config, "asr", None)
    cloud = CloudRecognizer(
        gemini_key=gemini.api_key() if gemini else "",
        groq_key=groq.api_key() if groq else "",
        gemini_model=(gemini.model if gemini else "gemini-2.5-flash") or "gemini-2.5-flash",
        hotwords=hotwords,
        order=tuple(getattr(asr_config, "cloud_order", ("gemini", "groq"))),
    )
    local = {
        "local:darija": LocalWhisperRecognizer(
            getattr(asr_config, "darija_model", "models/whisper-darija-ct2"),
            key="darija",
        ),
        "local:english": LocalWhisperRecognizer(
            getattr(asr_config, "english_model", "small.en"),
            language="en-GB",
            ram_mb=500,
            key="english",
        ),
        "local:multilingual": LocalWhisperRecognizer("small", ram_mb=500, key="multilingual"),
    }
    # A hard stop, stated once: with `cloud_audio = false` no candidate is
    # allowed to send audio anywhere, whatever the mode says.
    policy = RecognizerPolicy.from_config(config)
    factory = RecognizerFactory(cloud=cloud, local_models=local, policy=policy)
    return {"cloud": cloud, "local": local, "processor": processor, "factory": factory}


def _is_missing_local_model(value: str) -> bool:
    """A path we clearly meant to use, that is not there.

    `models/x` or `~/models/x` is a path, so absence is an error worth stating.
    `ychafiqui/whisper-small-darija` is a Hugging Face repo id — faster-whisper
    downloads it on first use, so "not on disk" is the normal state.
    """
    if Path(value).exists():
        return False
    if value.startswith(("models/", "models\\", "./", "../", "~", "/")):
        return True
    return "\\" in value


def _language_from_whisper(detected: str, text: str, hint: LanguageTag) -> LanguageTag:
    """Whisper says `ar` or `en`; Atlas speaks `ar-MA` or `en-GB`.

    An explicit hint wins (we asked for a language), then the model's own
    verdict, then the text itself — three sources, in decreasing authority.
    """
    if hint != "unknown":
        return hint
    if detected.startswith("ar"):
        return "ar-MA"
    if detected.startswith("en"):
        return "en-GB"
    from atlas_audio.postprocess import detect_language

    return detect_language(text)


__all__ = [
    "AsrAttempt",
    "CloudRecognizer",
    "LocalWhisperRecognizer",
    "RecognizerFactory",
    "RecognizerPolicy",
    "SherpaOfflineRecognizer",
    "build_recognizers",
    "wav_header",
]
