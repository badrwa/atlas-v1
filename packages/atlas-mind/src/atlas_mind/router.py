"""The router: quota → retry → timing → fallback chain.

Decorators wrap *any* `LlmProvider`, so retry/quota/timing logic exists exactly
once no matter how many providers are configured (R5).  The router itself only
decides **who answers**:

    1. first usable provider in `mind.provider_order`
    2. retried once on 429/5xx/timeout *before any text was spoken*
    3. next provider if it fails before producing text
    4. honest canned reply in the user's language if everyone is down

Once text has been emitted we never switch providers — you cannot un-speak a
sentence.  Failures after that point are reported as a truncated stream.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path

from atlas_core.contracts import (
    HealthReport,
    LlmEvent,
    LlmProvider,
    LlmRequest,
    QuotaState,
    StreamEnd,
    TextDelta,
)
from atlas_core.errors import ProviderError, QuotaExhausted, RateLimited
from atlas_core.events import EventBus, TimingRecorded

log = logging.getLogger(__name__)

OFFLINE_LINES: dict[str, str] = {
    "ar-MA": "سمح ليا، ماشي متصل بالأنترنيت دابا. عيط ليا من بعد ولا شغل المود المحلي.",
    "en-GB": "Sorry, I've lost the connection. Try again in a moment, or switch me to local mode.",
    "unknown": "Sorry, I can't reach my brain right now.",
}


# ─────────────────────────────────────────────────────────────────────
#  Quota
# ─────────────────────────────────────────────────────────────────────
class QuotaTracker:
    """Daily per-provider request budget, persisted so restarts don't reset it."""

    def __init__(self, path: str | Path | None = "data/quota.json") -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, dict[str, int]] = {}
        self._load()

    @property
    def today(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def used(self, provider: str) -> int:
        return self._data.get(self.today, {}).get(provider, 0)

    def state(self, provider: str, cap: int | None) -> QuotaState:
        return QuotaState(used_today=self.used(provider), daily_cap=cap)

    def check(self, provider: str, cap: int | None) -> None:
        if cap is not None and self.used(provider) >= cap:
            raise QuotaExhausted(f"{provider}: daily cap {cap} reached")

    def record(self, provider: str) -> None:
        day = self._data.setdefault(self.today, {})
        day[provider] = day.get(provider, 0) + 1
        self._save()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
        except OSError:  # pragma: no cover
            log.debug("quota_persist_failed path=%s", self.path)


# ─────────────────────────────────────────────────────────────────────
#  Decorators
# ─────────────────────────────────────────────────────────────────────
class ProviderDecorator(LlmProvider):
    """Base decorator — delegates everything it doesn't change."""

    def __init__(self, inner: LlmProvider) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.supports_tools = getattr(inner, "supports_tools", False)

    def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]:  # pragma: no cover - overridden
        return self.inner.stream(request)

    async def health(self) -> HealthReport:
        return await self.inner.health()

    def quota_state(self) -> QuotaState:
        return self.inner.quota_state()


class QuotaGuardedProvider(ProviderDecorator):
    """Refuses to call a provider whose free daily budget is spent."""

    def __init__(
        self, inner: LlmProvider, tracker: QuotaTracker, cap: int | None
    ) -> None:
        super().__init__(inner)
        self.tracker = tracker
        self.cap = cap

    async def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]:
        self.tracker.check(self.name, self.cap)
        try:
            async for event in self.inner.stream(request):
                yield event
        finally:
            self.tracker.record(self.name)

    def quota_state(self) -> QuotaState:
        return self.tracker.state(self.name, self.cap)


class TimedProvider(ProviderDecorator):
    """Measures time-to-first-token and total, publishes them as events."""

    def __init__(
        self,
        inner: LlmProvider,
        *,
        events: EventBus | None = None,
        on_timing: Callable[[str, str, float], None] | None = None,
    ) -> None:
        super().__init__(inner)
        self.events = events
        self.on_timing = on_timing
        self.last_ttft_ms: float | None = None
        self.last_total_ms: float | None = None

    async def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]:
        started = time.perf_counter()
        first: float | None = None
        async for event in self.inner.stream(request):
            if first is None and isinstance(event, TextDelta) and event.text.strip():
                first = (time.perf_counter() - started) * 1000
                self.last_ttft_ms = first
                await self._emit("llm.ttft", first)
            yield event
        self.last_total_ms = (time.perf_counter() - started) * 1000
        await self._emit("llm.total", self.last_total_ms)

    async def _emit(self, stage: str, ms: float) -> None:
        if self.on_timing:
            self.on_timing(stage, self.name, ms)
        if self.events:
            await self.events.publish(TimingRecorded(stage=stage, ms=ms, provider=self.name))


class RetryingProvider(ProviderDecorator):
    """Retries *before the first token*. After that, switching providers is pointless."""

    def __init__(
        self, inner: LlmProvider, *, retries: int = 1, backoff_s: float = 0.4
    ) -> None:
        super().__init__(inner)
        self.retries = retries
        self.backoff_s = backoff_s

    async def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]:
        import asyncio

        attempt = 0
        while True:
            emitted = False
            try:
                async for event in self.inner.stream(request):
                    if isinstance(event, TextDelta):
                        emitted = True
                    yield event
                return
            except (RateLimited, ProviderError) as exc:
                if emitted or attempt >= self.retries:
                    raise
                attempt += 1
                delay = self.backoff_s * attempt
                if isinstance(exc, RateLimited) and exc.retry_after_s:
                    delay = min(exc.retry_after_s, 2.0)
                log.info("provider_retry name=%s attempt=%d delay=%.1fs", self.name, attempt, delay)
                await asyncio.sleep(delay)


def build_chain(
    provider: LlmProvider,
    *,
    tracker: QuotaTracker | None = None,
    quota_cap: int | None = None,
    retries: int = 1,
    events: EventBus | None = None,
    on_timing: Callable[[str, str, float], None] | None = None,
) -> LlmProvider:
    """Wrap one provider in the standard decorator stack (order matters, see docs)."""
    chained: LlmProvider = provider
    if tracker is not None:
        chained = QuotaGuardedProvider(chained, tracker, quota_cap)
    chained = RetryingProvider(chained, retries=retries)
    return TimedProvider(chained, events=events, on_timing=on_timing)


# ─────────────────────────────────────────────────────────────────────
#  Canned fallback
# ─────────────────────────────────────────────────────────────────────
class CannedReplyProvider(LlmProvider):
    """Last resort: an honest sentence instead of silence."""

    name = "canned"
    model = "none"

    def __init__(self, language: str = "ar-MA") -> None:
        self.language = language

    def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]:
        return self._gen()

    async def _gen(self) -> AsyncIterator[LlmEvent]:
        yield TextDelta(text=OFFLINE_LINES.get(self.language, OFFLINE_LINES["unknown"]))
        yield StreamEnd(reason="fallback")

    async def health(self) -> HealthReport:
        return HealthReport(ok=True, detail="always available")


# ─────────────────────────────────────────────────────────────────────
#  Router
# ─────────────────────────────────────────────────────────────────────
class ProviderRouter:
    """Picks a provider per turn and reports honestly who answered."""

    def __init__(
        self,
        providers: list[LlmProvider],
        *,
        tracker: QuotaTracker | None = None,
        events: EventBus | None = None,
        retries: int = 1,
        fallback_language: str = "ar-MA",
    ) -> None:
        self.tracker = tracker
        self.events = events
        self.fallback_language = fallback_language
        self.providers = [
            build_chain(
                provider,
                tracker=tracker,
                quota_cap=_cap_of(provider),
                retries=retries,
                events=events,
            )
            for provider in providers
        ]
        self.last_provider: str = ""

    # ── introspection ────────────────────────────────────────────────
    @property
    def names(self) -> list[str]:
        return [provider.name for provider in self.providers]

    def describe(self) -> str:
        if not self.providers:
            return "no providers configured (add a key to .env)"
        return " → ".join(self.names)

    # ── the turn ─────────────────────────────────────────────────────
    async def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]:
        if not self.providers:
            fallback = CannedReplyProvider(self.fallback_language)
            async for event in fallback.stream(request):
                yield event
            self.last_provider = fallback.name
            return

        for provider in self.providers:
            emitted = False
            try:
                async for event in provider.stream(request):
                    if isinstance(event, TextDelta) and event.text.strip():
                        emitted = True
                    yield event
                self.last_provider = provider.name
                return
            except ProviderError as exc:
                log.warning("provider_failed name=%s error=%s", provider.name, exc)
                if emitted:
                    # Cannot un-speak a sentence: end honestly and let the caller explain.
                    yield StreamEnd(reason="truncated")
                    return
                continue

        fallback = CannedReplyProvider(self.fallback_language)
        async for event in fallback.stream(request):
            yield event
        self.last_provider = fallback.name

    # ── diagnostics ──────────────────────────────────────────────────
    async def health(self) -> dict[str, HealthReport]:
        reports: dict[str, HealthReport] = {}
        for provider in self.providers:
            try:
                reports[provider.name] = await provider.health()
            except Exception as exc:
                reports[provider.name] = HealthReport(ok=False, detail=str(exc)[:120])
        return reports

    def quotas(self) -> dict[str, QuotaState]:
        return {provider.name: provider.quota_state() for provider in self.providers}


def _cap_of(provider: LlmProvider) -> int | None:
    """Read the configured daily cap from the wrapped provider (if any)."""
    inner = getattr(provider, "config", None)
    return getattr(inner, "daily_request_cap", None)


__all__ = [
    "OFFLINE_LINES",
    "CannedReplyProvider",
    "ProviderDecorator",
    "ProviderRouter",
    "QuotaGuardedProvider",
    "QuotaTracker",
    "RetryingProvider",
    "TimedProvider",
    "build_chain",
]
