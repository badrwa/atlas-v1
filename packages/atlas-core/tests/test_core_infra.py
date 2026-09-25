"""DI, EventBus, config and the resource lease."""

from __future__ import annotations

import asyncio
import textwrap

import pytest

from atlas_core.config import AppConfig, load_config
from atlas_core.contracts import ResourceCost
from atlas_core.di import Container
from atlas_core.errors import ConfigError
from atlas_core.events import Event, EventBus, StateChanged, TokenDelta
from atlas_core.fakes import FakeClock, FakeProvider, FakeRecognizer, FakeSynthesizer
from atlas_core.resources import LeaseDenied, ResourceGovernor, ResourceLease

# ── DI ───────────────────────────────────────────────────────────────


def test_container_singleton_and_transient() -> None:
    container = Container()
    container.register("single", lambda _c: object())
    container.register("multi", lambda _c: object(), singleton=False)
    assert container.resolve("single") is container.resolve("single")
    assert container.resolve("multi") is not container.resolve("multi")


def test_container_override_wins() -> None:
    container = Container().register("thing", lambda _c: "real")
    container.override("thing", "fake")
    assert container.resolve("thing") == "fake"


def test_container_unknown_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        Container().resolve("nope")


# ── events ───────────────────────────────────────────────────────────
async def test_event_bus_delivers_to_queue_and_handler() -> None:
    bus = EventBus()
    queue = bus.subscribe("llm.token")
    seen: list[str] = []

    def record(ev: Event) -> None:
        if isinstance(ev, StateChanged):
            seen.append(ev.state)

    bus.on("state.changed", record)

    await bus.publish(TokenDelta(text="salam"))
    await bus.publish(StateChanged(state="listening", previous="waking"))

    first = await asyncio.wait_for(queue.get(), timeout=1)
    assert isinstance(first, TokenDelta)
    assert first.text == "salam"
    assert seen == ["listening"]


async def test_bad_subscriber_cannot_break_the_publisher() -> None:
    bus = EventBus()

    def exploding(_ev: Event) -> None:
        raise RuntimeError("boom")

    bus.on("state.changed", exploding)
    await bus.publish(StateChanged(state="dormant"))  # must not raise
    assert bus.stats["published"] == 1


async def test_slow_subscriber_drops_oldest_not_the_turn() -> None:
    bus = EventBus(queue_size=2)
    queue = bus.subscribe("llm.token")
    for i in range(5):
        await bus.publish(TokenDelta(text=str(i)))
    assert bus.stats["dropped"] == 3
    assert queue.qsize() == 2


def test_event_as_dict_is_serialisable() -> None:
    payload = StateChanged(state="thinking", previous="listening").as_dict()
    assert payload["topic"] == "state.changed"
    assert payload["state"] == "thinking"
    assert isinstance(payload["ts"], str)


# ── config ───────────────────────────────────────────────────────────
def test_config_defaults_and_env_override(tmp_path) -> None:
    config = load_config(path=tmp_path / "missing.toml", env_file=None, env={"ATLAS_VAULT_PATH": "/tmp/vault"})
    assert config.app.language == "ar-MA"
    assert config.obsidian.vault_path == "/tmp/vault"
    assert config.app.profile == "lean"


def test_config_reads_toml_and_provider_order(tmp_path) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text(
        textwrap.dedent(
            """
            [app]
            language = "ar-MA"

            [mind]
            provider_order = ["groq", "gemini"]

            [[providers]]
            name = "gemini"
            kind = "gemini"
            model = "gemini-2.5-flash-lite"
            base_url = "https://example.invalid/v1beta"
            api_key_env = "GEMINI_API_KEY"

            [[providers]]
            name = "groq"
            kind = "openai_compatible"
            model = "llama-3.3-70b-versatile"
            base_url = "https://example.invalid/v1"
            api_key_env = "GROQ_API_KEY"
            """
        ),
        encoding="utf-8",
    )
    env = {"GROQ_API_KEY": "k", "GEMINI_API_KEY": "k"}
    config = load_config(path=toml, env_file=None, env=env)
    assert [p.name for p in config.ordered_providers(env)] == ["groq", "gemini"]


def test_env_var_without_key_is_not_usable() -> None:
    config = AppConfig.model_validate(
        {
            "providers": [
                {
                    "name": "groq",
                    "model": "m",
                    "base_url": "https://example.invalid/v1",
                    "api_key_env": "SOME_MISSING_KEY_FOR_TESTS",
                }
            ]
        }
    )
    assert config.ordered_providers() == []


def test_local_provider_needs_no_key() -> None:
    config = AppConfig.model_validate(
        {
            "providers": [
                {
                    "name": "llamacpp",
                    "model": "local-gguf",
                    "base_url": "http://127.0.0.1:8123/v1",
                }
            ]
        }
    )
    assert len(config.ordered_providers()) == 1


def test_invalid_profile_is_rejected() -> None:
    with pytest.raises(ConfigError):
        AppConfig().with_env({"ATLAS_PROFILE": "turbo"})


def test_invalid_toml_is_a_config_error(tmp_path) -> None:
    bad = tmp_path / "config.toml"
    bad.write_text("[app\nbroken", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path=bad, env_file=None, env={})


# ── leases ───────────────────────────────────────────────────────────
async def test_lease_loads_then_evicts_on_ttl() -> None:
    clock = FakeClock()
    lease = ResourceLease(free_ram_mb=lambda: 4000, clock=clock, ttl_s=120.0)
    engine = FakeRecognizer(cost_mb=300)

    await lease.acquire(engine)
    assert engine.is_loaded() and lease.is_loaded(engine)
    assert lease.resident_mb() == 300

    clock.advance(200)
    evicted = await lease.async_evict_idle()
    assert evicted == [engine.name]
    assert not engine.is_loaded()


async def test_lease_touch_keeps_engine_alive() -> None:
    clock = FakeClock()
    lease = ResourceLease(free_ram_mb=lambda: 4000, clock=clock, ttl_s=120.0)
    engine = FakeRecognizer()
    await lease.acquire(engine)
    clock.advance(100)
    lease.touch(engine)
    clock.advance(100)  # 200 s total, but only 100 s since the last use
    assert await lease.async_evict_idle() == []
    assert engine.is_loaded()


async def test_lease_denies_when_ram_is_tight() -> None:
    lease = ResourceLease(free_ram_mb=lambda: 500, ram_floor_mb=400)
    with pytest.raises(LeaseDenied):
        await lease.acquire(FakeRecognizer(cost_mb=300))
    assert not lease.entries


async def test_lease_evicts_idle_before_denying() -> None:
    # Free RAM mirrors what a real host would report: base minus what is leased.
    clock = FakeClock()
    lease = ResourceLease(free_ram_mb=lambda: 1000, clock=clock, ram_floor_mb=400, ttl_s=60.0)
    lease.free_ram_mb = lambda: 1000 - lease.resident_mb()

    first = FakeRecognizer(text="a", cost_mb=500)
    await lease.acquire(first)
    clock.advance(90)  # first lease is now idle and may be evicted

    second = FakeRecognizer(text="b", cost_mb=400)
    await lease.acquire(second)
    assert not first.is_loaded(), "the idle lease should have been evicted to make room"
    assert second.is_loaded()
    assert lease.resident_mb() == 400


async def test_acquire_twice_is_a_noop() -> None:
    lease = ResourceLease(free_ram_mb=lambda: 4000)
    engine = FakeSynthesizer()
    await lease.acquire(engine)
    await lease.acquire(engine)
    assert len(lease.load_events) == 1


def test_governor_flags_degradation_once() -> None:
    free = {"mb": 2000}
    lease = ResourceLease(free_ram_mb=lambda: free["mb"])
    governor = ResourceGovernor(lease=lease, free_ram_mb=lambda: free["mb"], degrade_below_mb=500)

    assert governor.sample() is False
    free["mb"] = 300
    assert governor.sample() is True
    assert governor.degraded and governor.mode == "degraded"
    assert governor.sample() is False  # no flapping on subsequent ticks


def test_governor_can_afford() -> None:
    lease = ResourceLease(free_ram_mb=lambda: 1000, ram_floor_mb=400)
    governor = ResourceGovernor(lease=lease, free_ram_mb=lambda: 1000)
    assert governor.can_afford(ResourceCost(ram_mb=500)) is True
    assert governor.can_afford(ResourceCost(ram_mb=700)) is False


# ── fakes behave ─────────────────────────────────────────────────────
async def test_fake_provider_streams_text_then_end() -> None:
    provider = FakeProvider(script=[[{"text": "salam "}, {"text": "safi"}]])
    kinds = [ev.kind async for ev in provider.stream(_request())]
    assert kinds == ["text", "text", "end"]


async def test_fake_provider_can_fail_then_recover() -> None:
    from atlas_core.errors import RateLimited

    provider = FakeProvider(fail_times=1)
    with pytest.raises(RateLimited):
        [ev async for ev in provider.stream(_request())]
    kinds = [ev.kind async for ev in provider.stream(_request())]
    assert kinds[-1] == "end"


def _request():  # local helper (keeps imports tidy)
    from atlas_core.contracts import LlmRequest, Message, Role

    return LlmRequest(messages=[Message(role=Role.USER, content="salam")])
