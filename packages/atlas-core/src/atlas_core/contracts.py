"""The contracts: every concept in Atlas has exactly one abstraction.

If a component needs to talk to "speech in", it depends on ``SpeechRecognizer``
and nothing else.  Implementations live in ``atlas-audio`` / ``atlas-mind`` /
``atlas-skills``; this module stays dependency-free so every package can import
it without dragging a runtime along.

Each ABC documents its **contract test** — the shared suite under
``tests/contracts/`` that every implementation must pass (architecture doc §3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from array import array
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────────────────
#  Shared value types
# ─────────────────────────────────────────────────────────────────────

LanguageTag = Literal["ar-MA", "en-GB", "unknown"]


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Permission(StrEnum):
    """How dangerous a capability is. See docs/levels/LEVEL-07-hands.md."""

    SAFE = "safe"
    CONFIRM = "confirm"
    BLOCKED = "blocked"


class Message(BaseModel):
    role: Role
    content: str
    name: str | None = None


class ToolSpec(BaseModel):
    """A capability as the LLM sees it — one line per language (TTFT matters)."""

    name: str
    description: str
    description_darija: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    permission: Permission = Permission.SAFE


class LlmRequest(BaseModel):
    messages: list[Message] = Field(default_factory=list)
    system: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    temperature: float = 0.7
    max_output_tokens: int = 512
    json_schema: dict[str, Any] | None = None
    # The language this turn is expected in.  Providers mostly ignore it (the
    # system prompt already carries the style), but anything that has to *speak
    # without a model* — the offline fallback, an error line — needs to know.
    language: str = ""


# ── streaming events emitted by providers ────────────────────────────
class LlmEvent(BaseModel):
    """Base class for everything a provider streams back."""

    kind: str = "event"


class TextDelta(LlmEvent):
    kind: Literal["text"] = "text"
    text: str = ""


class StructuredResult(LlmEvent):
    kind: Literal["structured"] = "structured"
    data: dict[str, Any] = Field(default_factory=dict)


class ToolCallRequest(LlmEvent):
    kind: Literal["tool_call"] = "tool_call"
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)


class Usage(LlmEvent):
    kind: Literal["usage"] = "usage"
    prompt_tokens: int = 0
    completion_tokens: int = 0


class StreamEnd(LlmEvent):
    kind: Literal["end"] = "end"
    reason: str = "stop"


@dataclass(slots=True)
class HealthReport:
    ok: bool
    detail: str = ""
    latency_ms: float | None = None


@dataclass(slots=True)
class QuotaState:
    used_today: int = 0
    daily_cap: int | None = None

    @property
    def exhausted(self) -> bool:
        return self.daily_cap is not None and self.used_today >= self.daily_cap


@dataclass(slots=True)
class ResourceCost:
    """What loading this engine costs the machine (used by the lease governor)."""

    ram_mb: int = 0
    cpu_threads: int = 1
    disk_mb: int = 0
    cold_start_s: float = 0.0


@dataclass(slots=True)
class Transcript:
    text: str
    language: LanguageTag = "unknown"
    confidence: float = 0.0
    engine: str = ""
    duration_ms: float = 0.0


@dataclass(slots=True)
class AudioChunk:
    pcm: bytes
    sample_rate: int = 16000
    final: bool = False


@dataclass(slots=True)
class Detection:
    hit: bool = False
    keyword: str = ""
    score: float = 0.0


@dataclass(slots=True)
class SpeakerMatch:
    name: str = ""
    score: float = 0.0
    owner: bool = False


@dataclass(slots=True)
class SkillResult:
    ok: bool
    spoken: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    needs_confirmation: bool = False
    retry_hint: str = ""


@dataclass(slots=True)
class SkillContext:
    """Everything a skill is allowed to know about the current turn."""

    speaker: str = ""
    owner: bool = False
    language: LanguageTag = "ar-MA"
    dry_run: bool = False
    vault_path: str = ""
    workspace_dir: str = ""
    events: Any = None
    confirmed: bool = False


# ─────────────────────────────────────────────────────────────────────
#  ABCs
# ─────────────────────────────────────────────────────────────────────


class Sense(ABC):
    """Something that observes the world and must be started/stopped.

    Contract test: `start()` is idempotent, `stop()` after `start()` leaves no
    threads/tasks behind, `is_healthy()` is honest (False when it knows it's broken).
    """

    name: str = "sense"

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def is_healthy(self) -> bool: ...


class Engine(ABC):
    """A heavy local model that is *leased*, never resident (see ResourceLease).

    Contract test: `load()` twice is a no-op, `unload()` when unloaded is a
    no-op, `is_loaded()` reflects reality, `cost_hint()` is non-zero for models.
    """

    name: str = "engine"

    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def unload(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def cost_hint(self) -> ResourceCost: ...


class LlmProvider(ABC):
    """A brain.  Cloud, local, or fake — the router cannot tell the difference.

    Contract test: `stream()` yields at least one `StreamEnd`, never raises for
    ordinary provider failures (it yields an error event or raises ProviderError
    *before* emitting text), and `health()` never raises.
    """

    name: str = "provider"
    model: str = ""
    supports_tools: bool = False

    @abstractmethod
    def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]: ...

    @abstractmethod
    async def health(self) -> HealthReport: ...

    def quota_state(self) -> QuotaState:
        return QuotaState()


class SpeechRecognizer(Engine):
    """Speech in.  Implementations: cloud, faster-whisper, sherpa-onnx."""

    name = "recognizer"

    @abstractmethod
    async def transcribe(
        self, audio: bytes, *, language: LanguageTag = "unknown", sample_rate: int = 16000
    ) -> Transcript: ...


class SpeechSynthesizer(Engine):
    """Speech out.  Implementations: Piper, DarijaTTS sidecar, SAPI."""

    name = "synthesizer"

    @abstractmethod
    def synthesize(
        self, text: str, *, voice: str = "", language: LanguageTag = "ar-MA"
    ) -> Iterator[AudioChunk]: ...


#: One audio frame, ready for a sense.  L2 fixed the shape — 80 ms of 16 kHz
#: mono PCM — and int16 samples (`array("h")`) are the in-memory form the capture
#: path uses, while `bytes` is what crosses a process or an API boundary.  The
#: contract accepts both so neither side has to convert on the hot path.
FeedFrame = bytes | array


class WakeWordEngine(Sense):
    """Always-on, tiny, local.  Contract test: silence never fires a wake."""

    name = "wake"
    #: What this engine is listening for — shown by `atlas doctor`, logged with
    #: every hit, and the reason a hit is believable or not.
    keywords: tuple[str, ...] = ()

    @abstractmethod
    def feed(self, frame: FeedFrame) -> Detection: ...


class SpeakerVerifier(Engine):
    """Voice identity.  Contract test: identical clips → similarity ≈ 1.0."""

    name = "verifier"

    @abstractmethod
    def embed(self, audio: bytes, *, sample_rate: int = 16000) -> list[float]: ...

    @abstractmethod
    def verify(self, audio: bytes, profile: list[list[float]], *, threshold: float = 0.65) -> SpeakerMatch: ...


class Skill(ABC):
    """A capability Atlas can perform.  Contract test: bad args never execute."""

    name: str = "skill"
    permission: Permission = Permission.SAFE
    owner_only: bool = False

    @abstractmethod
    def spec(self) -> ToolSpec: ...

    @abstractmethod
    def invoke(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult: ...


class DeclaredSkill(Skill):
    """A skill that describes itself with plain class attributes.

    Skills differ in *what they do*, not in how they announce themselves.  Without
    this, every capability repeats the same `spec()` body — and a tool schema is
    exactly the thing that drifts out of sync once it is copy-pasted.
    """

    description: str = ""
    description_darija: str = ""
    parameters: ClassVar[dict[str, Any]] = {}

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            description_darija=self.description_darija,
            parameters=dict(self.parameters),
            permission=self.permission,
        )


class Store(ABC):
    """Memory.  Implementations: vault (Markdown) and SQLite (FTS5)."""

    name: str = "store"

    @abstractmethod
    def get(self, key: str) -> str | None: ...

    @abstractmethod
    def set(self, key: str, value: str) -> None: ...

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]: ...


# ─────────────────────────────────────────────────────────────────────
#  Protocols (structural, for things that don't need to inherit)
# ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class Resettable(Protocol):
    def reset(self) -> None: ...


@runtime_checkable
class Stoppable(Protocol):
    def stop(self) -> None: ...


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (Arabic/Darija is denser than English).

    Deliberately conservative: underestimating costs a little latency,
    overestimating costs context.  We under-promise in Latin and split the
    difference elsewhere.
    """
    if not text:
        return 0
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06ff")
    latin = len(text) - arabic
    return int(arabic / 2.2 + latin / 3.6) + 1


def trim_to_tokens(chunks: Iterable[str], budget: int, *, sep: str = "\n") -> str:
    """Join `chunks` newest-last, keeping only what fits the budget."""
    kept: list[str] = []
    used = 0
    for chunk in reversed(list(chunks)):
        cost = estimate_tokens(chunk)
        if used + cost > budget:
            break
        kept.append(chunk)
        used += cost
    return sep.join(reversed(kept))


__all__ = [
    "AudioChunk",
    "Detection",
    "Engine",
    "HealthReport",
    "LanguageTag",
    "LlmEvent",
    "LlmProvider",
    "LlmRequest",
    "Message",
    "Permission",
    "QuotaState",
    "Resettable",
    "ResourceCost",
    "Role",
    "Sense",
    "Skill",
    "SkillContext",
    "SkillResult",
    "SpeakerMatch",
    "SpeakerVerifier",
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "Stoppable",
    "Store",
    "StreamEnd",
    "StructuredResult",
    "TextDelta",
    "ToolCallRequest",
    "ToolSpec",
    "Transcript",
    "Usage",
    "WakeWordEngine",
    "estimate_tokens",
    "trim_to_tokens",
]
