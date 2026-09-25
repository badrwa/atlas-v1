"""Engine slots for L2/L3 — declared, validated, and honestly not implemented.

Why ship empty classes?  Because the contracts are the architecture (plan §4.1):
declaring them now means `atlas doctor` can report exactly what is missing, the
factory has fixed extension points, and L2/L3 become "fill in these five classes"
instead of "design an audio layer".

Every one of these raises `Unsupported` on `load()` until its level lands.
"""

from __future__ import annotations

from collections.abc import Iterator

from atlas_core.contracts import (
    AudioChunk,
    Detection,
    LanguageTag,
    ResourceCost,
    SpeakerMatch,
    Transcript,
)
from atlas_core.errors import Unsupported

NOT_YET = "not implemented until {level} — see docs/levels/{file}"


class _PlannedEngine:
    """Shared behaviour: knows its budget, refuses to pretend it works."""

    name = "planned"
    level = "L2"
    level_file = "LEVEL-02-ears.md"
    ram_mb = 0
    cold_start_s = 0.0

    def __init__(self) -> None:
        self._loaded = False

    async def load(self) -> None:
        raise Unsupported(NOT_YET.format(level=self.level, file=self.level_file))

    async def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def cost_hint(self) -> ResourceCost:
        return ResourceCost(ram_mb=self.ram_mb, cold_start_s=self.cold_start_s)


class SherpaKws(_PlannedEngine):
    """Wake word 'atlas' via sherpa-onnx keyword spotting (L2)."""

    name = "sherpa-kws"
    ram_mb = 30


class SileroVad(_PlannedEngine):
    """Voice activity detection + endpointing (L2)."""

    name = "silero-vad"
    ram_mb = 20


class LocalWhisperRecognizer(_PlannedEngine):
    """Darija/English ASR via faster-whisper INT8 (L2) — Bunker mode."""

    name = "whisper-local"
    ram_mb = 400
    cold_start_s = 3.0

    async def transcribe(
        self, audio: bytes, *, language: LanguageTag = "unknown", sample_rate: int = 16_000
    ) -> Transcript:
        raise Unsupported(NOT_YET.format(level="L2", file="LEVEL-02-ears.md"))


class CloudRecognizer(_PlannedEngine):
    """Darija-first ASR through Gemini / Groq (L2) — Lean mode."""

    name = "asr-cloud"
    ram_mb = 0

    async def transcribe(
        self, audio: bytes, *, language: LanguageTag = "unknown", sample_rate: int = 16_000
    ) -> Transcript:
        raise Unsupported(NOT_YET.format(level="L2", file="LEVEL-02-ears.md"))


class SherpaSpeakerVerifier(_PlannedEngine):
    """Voice identity via speaker embeddings (L4)."""

    name = "speaker-verify"
    level = "L4"
    level_file = "LEVEL-04-identity.md"
    ram_mb = 90

    def embed(self, audio: bytes, *, sample_rate: int = 16_000) -> list[float]:
        raise Unsupported(NOT_YET.format(level="L4", file="LEVEL-04-identity.md"))

    def verify(
        self, audio: bytes, profile: list[list[float]], *, threshold: float = 0.65
    ) -> SpeakerMatch:
        raise Unsupported(NOT_YET.format(level="L4", file="LEVEL-04-identity.md"))


class _PlannedVoice(_PlannedEngine):
    """Shared body for the two voices that do not exist yet.

    Piper and the Darija sidecar differ in *what they are* (language, model,
    RAM budget) — not in how they refuse to work before L3 lands.
    """

    def synthesize(
        self, text: str, *, voice: str = "", language: LanguageTag = "ar-MA"
    ) -> Iterator[AudioChunk]:
        raise Unsupported(NOT_YET.format(level=self.level, file=self.level_file))


class PiperSynthesizer(_PlannedVoice):
    """British English voice, ONNX, realtime on this CPU (L3)."""

    name = "piper"
    level = "L3"
    level_file = "LEVEL-03-mouth.md"
    ram_mb = 150
    cold_start_s = 0.5


class DarijaTtsSidecar(_PlannedVoice):
    """Moroccan Darija voice (DarijaTTS-500M via llama.cpp) in an isolated venv (L3)."""

    name = "darija-tts"
    level = "L3"
    level_file = "LEVEL-03-mouth.md"
    ram_mb = 700
    cold_start_s = 8.0


class WakeWordFeed:
    """Minimal wake-word holder used by doctor/CLI before L2 exists."""

    name = "wake"

    def __init__(self, keyword: str = "atlas") -> None:
        self.keyword = keyword

    def feed(self, frame: bytes) -> Detection:  # pragma: no cover - L2 replaces this
        return Detection()


__all__ = [
    "CloudRecognizer",
    "DarijaTtsSidecar",
    "LocalWhisperRecognizer",
    "PiperSynthesizer",
    "SherpaKws",
    "SherpaSpeakerVerifier",
    "SileroVad",
    "WakeWordFeed",
]
