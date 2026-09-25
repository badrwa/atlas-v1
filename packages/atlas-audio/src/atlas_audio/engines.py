"""The one engine slot that is still *planned*: speaker identity (L4).

Why keep an empty class at all?  Because the contracts are the architecture
(plan §4.1): declaring it means `atlas doctor` can report exactly what is missing,
the factory has a fixed extension point, and L4 becomes "fill in this class"
instead of "design an identity layer".

**L2 and L3 are real now** and their implementations live in their own modules —
`wake.py`, `vad.py`, `asr.py`, `tts.py`, `speech.py`, `playback.py`.  The three
stubs that used to sit here (`PiperSynthesizer`, `DarijaTtsSidecar`,
`SherpaSpeakerVerifier`) are gone: the first two are the real engines now, and
having *two* definitions of a class called `PiperSynthesizer` in one package
would be a bug with a docstring, not a plan.
"""

from __future__ import annotations

from atlas_core.contracts import ResourceCost, SpeakerMatch
from atlas_core.errors import Unsupported

NOT_YET = "not implemented until {level} — see docs/levels/{file}"


class SherpaSpeakerVerifier:
    """Voice identity via speaker embeddings (L4).

    Budgeted from the plan: ~90 MB resident for the embedding model, loaded on
    demand through `ResourceLease` the first time an unknown speaker is heard.
    """

    name = "speaker-verify"
    level = "L4"
    level_file = "LEVEL-04-identity.md"
    ram_mb = 90
    cold_start_s = 1.5

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

    def embed(self, audio: bytes, *, sample_rate: int = 16_000) -> list[float]:
        raise Unsupported(NOT_YET.format(level=self.level, file=self.level_file))

    def verify(
        self, audio: bytes, profile: list[list[float]], *, threshold: float = 0.65
    ) -> SpeakerMatch:
        raise Unsupported(NOT_YET.format(level=self.level, file=self.level_file))


__all__ = ["SherpaSpeakerVerifier"]
