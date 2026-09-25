"""Engine slots that are still *planned* — declared, sized, and honestly empty.

Why keep empty classes at all?  Because the contracts are the architecture
(plan §4.1): declaring them means `atlas doctor` can report exactly what is
missing, the factory has fixed extension points, and L3/L4 become "fill in these
classes" instead of "design an audio layer".

**L2 is now real** — `frames`, `vad`, `wake`, `asr`, `postprocess` and `loop`
carry the working implementations (`SileroVad`, `EnergyVad`, `SherpaKwsEngine`,
`CloudRecognizer`, `LocalWhisperRecognizer`).  This file keeps only what later
levels owe: the British voice (L3), the Darija voice sidecar (L3) and speaker
verification (L4).  Two definitions of a `CloudRecognizer` in one package would
be a bug, not a plan.
"""

from __future__ import annotations

from collections.abc import Iterator

from atlas_core.contracts import AudioChunk, LanguageTag, ResourceCost, SpeakerMatch
from atlas_core.errors import Unsupported

NOT_YET = "not implemented until {level} — see docs/levels/{file}"


class _PlannedEngine:
    """Shared behaviour: knows its budget, refuses to pretend it works."""

    name = "planned"
    level = "L3"
    level_file = "LEVEL-03-mouth.md"
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


__all__ = [
    "DarijaTtsSidecar",
    "PiperSynthesizer",
    "SherpaSpeakerVerifier",
]
