"""Fakes for every contract.

Rule of the level docs: *every new capability ships with a fake*, so CI can run
a whole conversation with no microphone, no cloud key and no GPU.  These are
also the reference behaviour for the contract test suites.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any, ClassVar

from atlas_core.contracts import (
    AudioChunk,
    DeclaredSkill,
    Detection,
    HealthReport,
    LlmEvent,
    LlmProvider,
    LlmRequest,
    Permission,
    QuotaState,
    ResourceCost,
    Sense,
    SkillContext,
    SkillResult,
    SpeakerMatch,
    SpeakerVerifier,
    SpeechRecognizer,
    SpeechSynthesizer,
    StreamEnd,
    StructuredResult,
    TextDelta,
    ToolCallRequest,
    Transcript,
    WakeWordEngine,
)
from atlas_core.errors import ProviderUnavailable, RateLimited


# ── LLM ──────────────────────────────────────────────────────────────
class FakeProvider(LlmProvider):
    """Scriptable provider.

    ``script`` is a list of "turns"; each turn is a list of events to emit.
    Special events: ``{"error": "rate_limited"}``, ``{"error": "unavailable"}``,
    ``{"delay": 0.01}``.  Used by router, decorator and tool-loop tests.
    """

    supports_tools = True

    def __init__(
        self,
        name: str = "fake",
        script: list[list[dict[str, Any]]] | None = None,
        *,
        model: str = "fake-1",
        fail_times: int = 0,
        healthy: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.script = script or [[{"text": "salam, kifach nta?"}]]
        self.fail_times = fail_times
        self.healthy = healthy
        self.calls: list[LlmRequest] = []
        self._turn = 0

    async def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]:
        self.calls.append(request)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RateLimited("fake 429", retry_after_s=0.0)

        step = self.script[min(self._turn, len(self.script) - 1)]
        self._turn += 1
        for item in step:
            if "error" in item:
                kind = item["error"]
                if kind == "rate_limited":
                    raise RateLimited("fake 429", retry_after_s=0.0)
                raise ProviderUnavailable(f"fake {kind}")
            if "delay" in item:
                await asyncio.sleep(item["delay"])
            if "text" in item:
                yield TextDelta(text=item["text"])
            if "structured" in item:
                yield StructuredResult(data=item["structured"])
            if "tool" in item:
                yield ToolCallRequest(name=item["tool"], arguments=item.get("args", {}))
        yield StreamEnd()

    async def health(self) -> HealthReport:
        return HealthReport(ok=self.healthy, detail="fake", latency_ms=1.0)

    def quota_state(self) -> QuotaState:
        return QuotaState(used_today=0, daily_cap=None)


# ── audio ────────────────────────────────────────────────────────────
class FakeSense(Sense):
    def __init__(self, name: str = "fake-sense") -> None:
        self.name = name
        self.started = False
        self.healthy = True

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    def is_healthy(self) -> bool:
        return self.healthy


class FakeWake(WakeWordEngine):
    """Fires on the Nth frame, or never — deterministic for tests."""

    name = "fake-wake"

    def __init__(self, hit_at: int | None = 1, keyword: str = "atlas") -> None:
        self.hit_at = hit_at
        self.keyword = keyword
        self.frames = 0
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False

    def is_healthy(self) -> bool:
        return True

    def feed(self, frame: bytes) -> Detection:
        self.frames += 1
        if self.hit_at is not None and self.frames == self.hit_at:
            return Detection(hit=True, keyword=self.keyword, score=0.99)
        return Detection()


class FakeEngineMixin:
    """The engine lifecycle every fake shares: load, unload, is_loaded, cost_hint.

    Real engines do not share this — an ONNX session and an HTTP client have
    genuinely different lifecycles — but all three fakes model the *same
    contract*, so writing it three times would be three copies of one idea.
    """

    _cold_start_s: float = 1.0

    def _init_engine(self, cost_mb: int) -> None:
        self._loaded = False
        self._cost_mb = cost_mb

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def cost_hint(self) -> ResourceCost:
        return ResourceCost(ram_mb=self._cost_mb, cold_start_s=self._cold_start_s)


class FakeRecognizer(FakeEngineMixin, SpeechRecognizer):
    name = "fake-asr"

    def __init__(self, text: str = "chno ljaw?", language: str = "ar-MA", cost_mb: int = 300) -> None:
        self.text = text
        self.language = language
        self.transcribe_calls = 0
        self._init_engine(cost_mb)

    async def transcribe(
        self, audio: bytes, *, language: str = "unknown", sample_rate: int = 16_000
    ) -> Transcript:
        self.transcribe_calls += 1
        return Transcript(
            text=self.text,
            language=self.language,  # type: ignore[arg-type]
            confidence=0.95,
            engine=self.name,
            duration_ms=42.0,
        )


class FakeSynthesizer(FakeEngineMixin, SpeechSynthesizer):
    name = "fake-tts"
    _cold_start_s = 0.5

    def __init__(self, cost_mb: int = 120) -> None:
        self.spoken: list[str] = []
        self._init_engine(cost_mb)

    def synthesize(self, text: str, *, voice: str = "", language: str = "ar-MA") -> Iterator[AudioChunk]:
        self.spoken.append(text)
        yield AudioChunk(pcm=b"\x00\x01" * 32, final=True)


class FakeVerifier(FakeEngineMixin, SpeakerVerifier):
    name = "fake-speaker"
    _cold_start_s = 0.4

    def __init__(self, owner: bool = True, score: float = 0.91) -> None:
        self.owner = owner
        self.score = score
        self._init_engine(90)

    def embed(self, audio: bytes, *, sample_rate: int = 16_000) -> list[float]:
        return [0.1, 0.2, 0.3]

    def verify(
        self, audio: bytes, profile: list[list[float]], *, threshold: float = 0.65
    ) -> SpeakerMatch:
        return SpeakerMatch(name="badr" if self.owner else "", score=self.score, owner=self.owner)


# ── skills ───────────────────────────────────────────────────────────
class FakeSkill(DeclaredSkill):
    name = "fake_skill"
    permission = Permission.SAFE
    description = "a fake skill for tests"
    description_darija = "مهارة تجريبية"
    parameters: ClassVar[dict[str, Any]] = {"type": "object", "properties": {}}

    def __init__(
        self, result: SkillResult | None = None, *, permission: Permission | None = None
    ) -> None:
        # Only override the class attribute when explicitly asked, so subclasses
        # like BlockedSkill keep their declared permission.
        if permission is not None:
            self.permission = permission
        self.result = result or SkillResult(ok=True, spoken="safi", data={"echo": True})
        self.calls: list[tuple[dict[str, Any], SkillContext]] = []

    def invoke(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        self.calls.append((args, ctx))
        return self.result


# ── misc ─────────────────────────────────────────────────────────────
class FakeClock:
    """Deterministic clock for FSM timeouts and lease TTL tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> float:
        self._now += seconds
        return self._now

    def monotonic(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:  # synchronous advance for tests
        self.advance(seconds)


class Timer:
    """Tiny wall-clock stopwatch used by pipelines and benchmarks."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def restart(self) -> Timer:
        self._start = time.perf_counter()
        return self


__all__ = [
    "FakeClock",
    "FakeProvider",
    "FakeRecognizer",
    "FakeSense",
    "FakeSkill",
    "FakeSynthesizer",
    "FakeVerifier",
    "FakeWake",
    "Timer",
]
