"""Test doubles for the audio layer.

The energy wake engine is a fine *stand-in* for a real keyword spotter when a
test wants "something loud was said", but it is useless when the test is about
*whether the loop was listening for a wake word at all* — it fires on any speech
at all.  `ScriptedWake` is the double that fixes that: it is a real
`WakeWordEngine`, and the test decides when a keyword is allowed to exist.
"""

from __future__ import annotations

from atlas_audio.capture import silence, speech
from atlas_audio.wake import EnergyWakeEngine
from atlas_core.contracts import Detection, LanguageTag, Transcript


class ScriptedWake:
    """A wake engine under the test's control: it can veto any frame.

    `allow_from` / `allow_until` are frame indices.  Outside that window every
    frame is reported as a miss *regardless of loudness*, which is what makes it
    possible to say "somebody spoke, but it was not the keyword".
    """

    name = "scripted-wake"

    def __init__(
        self,
        *,
        allow_from: int = 0,
        allow_until: int | None = None,
        confirm_frames: int = 1,
    ) -> None:
        self.allow_from = allow_from
        self.allow_until = allow_until
        self._inner = EnergyWakeEngine(confirm_frames=confirm_frames)
        self._seen = 0

    @property
    def keywords(self) -> tuple[str, ...]:
        return self._inner.keywords

    async def start(self) -> None:
        await self._inner.start()

    async def stop(self) -> None:
        await self._inner.stop()

    def is_healthy(self) -> bool:
        return True

    def feed(self, frame) -> Detection:
        self._seen += 1
        if self._seen <= self.allow_from:
            return Detection(hit=False)
        if self.allow_until is not None and self._seen > self.allow_until:
            return Detection(hit=False)
        return self._inner.feed(frame)


class FakeRecognizer:
    """A recogniser that returns what the test says, and records what it got."""

    name = "fake-asr"

    def __init__(
        self, text: str = "chno ljaw", language: LanguageTag = "ar-MA", confidence: float = 0.9
    ) -> None:
        self.text = text
        self.language = language
        self.confidence = confidence
        self.calls: list[int] = []
        self.fail: Exception | None = None

    async def transcribe(
        self, audio: bytes, *, language: LanguageTag = "unknown", sample_rate: int = 16000
    ) -> Transcript:
        self.calls.append(len(audio))
        if self.fail is not None:
            raise self.fail
        text = self.text
        if self.calls and text == "__echo__":  # pragma: no cover - convenience
            text = f"call-{len(self.calls)}"
        return Transcript(
            text=text, language=self.language, confidence=self.confidence, engine=self.name
        )

    async def load(self) -> None: ...

    async def unload(self) -> None: ...

    def is_loaded(self) -> bool:
        return True

    def cost_hint(self):
        from atlas_core.contracts import ResourceCost

        return ResourceCost()


def utterance_audio(*, lead: int = 3, speech_frames: int = 14) -> list:
    """Wake-word audio, a pause, then a command — how a person actually speaks."""
    return silence(lead) + speech(4) + silence(8) + speech(speech_frames) + silence(8)


__all__ = ["FakeRecognizer", "ScriptedWake", "utterance_audio"]
