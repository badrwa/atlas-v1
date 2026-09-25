"""Router behaviour: fallback, retry, quota, honesty."""

from __future__ import annotations

import pytest

from atlas_core.contracts import LlmRequest, StreamEnd, TextDelta
from atlas_core.errors import AuthFailed, QuotaExhausted
from atlas_core.events import EventBus, TimingRecorded
from atlas_core.fakes import FakeProvider
from atlas_mind.router import (
    CannedReplyProvider,
    ProviderRouter,
    QuotaGuardedProvider,
    QuotaTracker,
    RetryingProvider,
    build_chain,
)


def _request() -> LlmRequest:
    return LlmRequest()


async def _text(router: ProviderRouter) -> str:
    events = [event async for event in router.stream(_request())]
    return "".join(event.text for event in events if isinstance(event, TextDelta))


# ── fallback ─────────────────────────────────────────────────────────
async def test_uses_first_provider_when_healthy() -> None:
    primary = FakeProvider(name="primary", script=[[{"text": "salam"}]])
    secondary = FakeProvider(name="secondary", script=[[{"text": "nope"}]])
    router = ProviderRouter([primary, secondary])
    assert await _text(router) == "salam"
    assert router.last_provider == "primary"
    assert secondary.calls == []


async def test_falls_back_when_primary_fails_before_any_text() -> None:
    primary = FakeProvider(name="primary", script=[[{"error": "rate_limited"}]])
    secondary = FakeProvider(name="secondary", script=[[{"text": "safi"}]])
    router = ProviderRouter([primary, secondary], retries=0)
    assert await _text(router) == "safi"
    assert router.last_provider == "secondary"


async def test_never_switches_provider_after_text_was_emitted() -> None:
    """You cannot un-speak a sentence — the stream ends honestly instead."""

    class HalfThenFail(FakeProvider):
        async def stream(self, request):  # type: ignore[override]
            yield _text_delta("salam ")
            raise AuthFailed("upstream died mid-sentence")

    from atlas_core.contracts import TextDelta

    def _text_delta(text: str) -> TextDelta:
        return TextDelta(text=text)

    primary = HalfThenFail(name="primary")
    secondary = FakeProvider(name="secondary", script=[[{"text": "should never be spoken"}]])
    router = ProviderRouter([primary, secondary], retries=0)

    events = [event async for event in router.stream(_request())]
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "salam "
    last = events[-1]
    assert isinstance(last, StreamEnd) and last.reason == "truncated"
    assert secondary.calls == []


async def test_canned_reply_when_everything_is_down() -> None:
    router = ProviderRouter(
        [FakeProvider(name="a", script=[[{"error": "unavailable"}]])],
        retries=0,
        fallback_language="en-GB",
    )
    text = await _text(router)
    assert "connection" in text.lower()
    assert router.last_provider == "canned"


async def test_no_providers_configured_still_speaks() -> None:
    router = ProviderRouter([], fallback_language="ar-MA")
    text = await _text(router)
    assert text
    assert "سمح" in text


# ── retry ────────────────────────────────────────────────────────────
async def test_retry_recovers_a_single_rate_limit() -> None:
    provider = RetryingProvider(FakeProvider(name="flaky", fail_times=1), retries=1, backoff_s=0.0)
    events = [event async for event in provider.stream(_request())]
    assert events[-1].kind == "end"


async def test_retry_gives_up_after_the_configured_attempts() -> None:
    from atlas_core.errors import RateLimited

    provider = RetryingProvider(FakeProvider(name="dead", fail_times=5), retries=1, backoff_s=0.0)
    with pytest.raises(RateLimited):
        [event async for event in provider.stream(_request())]


# ── quota ────────────────────────────────────────────────────────────
def test_quota_tracker_counts_and_persists(tmp_path) -> None:
    path = tmp_path / "quota.json"
    tracker = QuotaTracker(path)
    tracker.record("groq")
    tracker.record("groq")
    assert tracker.used("groq") == 2

    reloaded = QuotaTracker(path)
    assert reloaded.used("groq") == 2, "quota must survive a restart"


def test_quota_check_raises_when_cap_reached(tmp_path) -> None:
    tracker = QuotaTracker(tmp_path / "quota.json")
    tracker.record("groq")
    with pytest.raises(QuotaExhausted):
        tracker.check("groq", cap=1)
    tracker.check("groq", cap=2)  # under the cap → fine


async def test_quota_guard_blocks_and_counts(tmp_path) -> None:
    tracker = QuotaTracker(tmp_path / "quota.json")
    inner = FakeProvider(name="groq", script=[[{"text": "salam"}]])
    guarded = QuotaGuardedProvider(inner, tracker, cap=1)

    [event async for event in guarded.stream(_request())]
    assert tracker.used("groq") == 1
    with pytest.raises(QuotaExhausted):
        [event async for event in guarded.stream(_request())]


def test_build_chain_wraps_in_the_documented_order(tmp_path) -> None:
    provider = build_chain(
        FakeProvider(name="groq"), tracker=QuotaTracker(tmp_path / "q.json"), quota_cap=5
    )
    # outermost is the timing layer, innermost is quota-guarded retry
    assert type(provider).__name__ == "TimedProvider"
    assert type(provider.inner).__name__ == "RetryingProvider"  # type: ignore[attr-defined]


# ── timing + events ──────────────────────────────────────────────────
async def test_timing_decorator_publishes_ttft_and_total() -> None:
    bus = EventBus()
    queue = bus.subscribe("timing.recorded")
    provider = build_chain(FakeProvider(name="groq", script=[[{"text": "salam"}]]), events=bus)
    [event async for event in provider.stream(_request())]

    stages: list[str] = []
    while not queue.empty():
        event = await queue.get()
        if isinstance(event, TimingRecorded):
            stages.append(event.stage)
    assert "llm.ttft" in stages
    assert "llm.total" in stages


async def test_canned_provider_speaks_the_right_language() -> None:
    provider = CannedReplyProvider("en-GB")
    events = [e async for e in provider.stream(_request())]
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "connection" in text.lower()
