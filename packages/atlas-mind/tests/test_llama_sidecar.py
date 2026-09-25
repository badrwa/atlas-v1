"""The local llama.cpp sidecar: available only when it honestly is.

Nothing here needs a 1 GB model or a real llama-server binary: the process is
injected, so the *decisions* are tested — which files are missing, what command
would run, what happens when the server never becomes healthy, and how the
provider degrades so a conversation never dies because a local model is absent.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from atlas_core.config import ProviderConfig
from atlas_core.contracts import LlmRequest
from atlas_core.resources import ResourceLease
from atlas_mind.providers.llama_cpp import (
    DEFAULT_PORT,
    DEFAULT_THREADS,
    LlamaCppProvider,
    LlamaServerEngine,
    SidecarSpec,
    find_default_paths,
)


class FakeProcess:
    """A Popen stand-in: alive until terminated, like a real server."""

    def __init__(self, argv: list[str], *, dies_immediately: bool = False, **kwargs) -> None:
        self.argv = argv
        self.kwargs = kwargs
        self._alive = not dies_immediately
        self.terminated = False

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self._alive = False
        self.terminated = True

    def kill(self) -> None:
        self._alive = False


def make_spec(tmp_path: Path, *, binary: bool = True, model: bool = True) -> SidecarSpec:
    binary_path = tmp_path / "llama-server"
    model_path = tmp_path / "qwen3-1.7b-q4_k_m.gguf"
    if binary:
        binary_path.write_text("#!/bin/sh\n", encoding="utf-8")
        binary_path.chmod(0o755)
    if model:
        model_path.write_text("not really a model", encoding="utf-8")
    return SidecarSpec(
        binary=binary_path if binary else None,
        model_path=model_path if model else None,
    )


# ── availability ─────────────────────────────────────────────────────
def test_missing_bits_are_named(tmp_path: Path) -> None:
    spec = make_spec(tmp_path, binary=False, model=False)
    assert spec.is_available is False
    assert spec.missing == ["llama-server binary", "GGUF model"]


def test_a_complete_spec_is_available_offline(tmp_path: Path) -> None:
    assert make_spec(tmp_path).is_available is True


def test_hint_tells_the_user_what_to_download(tmp_path: Path) -> None:
    hint = make_spec(tmp_path, binary=False).hint()
    assert "llama-server" in hint
    assert "gguf" in hint.lower()


def test_command_is_the_documented_one(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    argv = spec.command()
    assert argv[0] == str(spec.binary)
    assert argv[1] == "-m"
    assert argv[2] == str(spec.model_path)
    assert "--port" in argv and str(DEFAULT_PORT) in argv
    assert "-t" in argv and str(DEFAULT_THREADS) in argv


def test_threads_match_a_two_core_laptop(tmp_path: Path) -> None:
    """More threads than cores makes a Skylake slower, not faster."""
    assert DEFAULT_THREADS == 2


def test_find_default_paths_reports_absence_honestly(tmp_path: Path) -> None:
    spec = find_default_paths(tmp_path)
    assert spec.is_available is False, "an empty workspace has no local brain"


def test_find_default_paths_finds_what_the_plan_says_to_install(tmp_path: Path) -> None:
    (tmp_path / "vendor" / "local-llm").mkdir(parents=True)
    (tmp_path / "vendor" / "local-llm" / "llama-server").write_text("x", encoding="utf-8")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "qwen3-1.7b-q4_k_m.gguf").write_text("x", encoding="utf-8")

    spec = find_default_paths(tmp_path)
    assert spec.is_available is True
    assert spec.model_path is not None
    assert spec.model_path.name == "qwen3-1.7b-q4_k_m.gguf"


# ── the engine (an Engine, so the lease can own it) ──────────────────
def test_engine_is_unloaded_until_started(tmp_path: Path) -> None:
    engine = LlamaServerEngine(make_spec(tmp_path))
    assert engine.is_loaded() is False
    assert engine.cost_hint().ram_mb > 0, "a model costs RAM; say how much"


def test_load_is_a_no_op_when_already_running(tmp_path: Path) -> None:
    spawned: list[FakeProcess] = []

    def spawn(argv, **kwargs):
        process = FakeProcess(argv, **kwargs)
        spawned.append(process)
        return process

    async def run() -> None:
        engine = LlamaServerEngine(make_spec(tmp_path), popen=spawn, waiter=_always_healthy)
        await engine.load()
        await engine.load()
        assert len(spawned) == 1, "the contract says load() twice is a no-op"

    asyncio.run(run())


def test_unload_when_not_loaded_is_a_no_op(tmp_path: Path) -> None:
    async def run() -> None:
        engine = LlamaServerEngine(make_spec(tmp_path))
        await engine.unload()
        assert engine.is_loaded() is False

    asyncio.run(run())


def test_a_server_that_never_answers_health_is_stopped_and_reported(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []

    def spawn(argv, **kwargs):
        process = FakeProcess(argv, **kwargs)
        processes.append(process)
        return process

    async def run() -> None:
        engine = LlamaServerEngine(
            make_spec(tmp_path), popen=spawn, health_timeout_s=0.2, poll_interval_s=0.05
        )
        with pytest.raises(Exception) as excinfo:
            await engine.load()
        assert "healthy" in str(excinfo.value)
        assert engine.is_loaded() is False, "a dead sidecar must not be left running"

    asyncio.run(run())


def test_load_without_files_refuses_with_the_hint(tmp_path: Path) -> None:
    async def run() -> None:
        engine = LlamaServerEngine(make_spec(tmp_path, binary=False, model=False))
        with pytest.raises(Exception) as excinfo:
            await engine.load()
        assert "llama-server" in str(excinfo.value)

    asyncio.run(run())


async def _always_healthy() -> bool:
    return True


# ── the provider ─────────────────────────────────────────────────────
def provider_for(tmp_path: Path, **kwargs) -> tuple[LlamaCppProvider, SidecarSpec]:
    spec = kwargs.pop("spec", make_spec(tmp_path))
    config = ProviderConfig(name="llamacpp", kind="llama_cpp", model="qwen3-1.7b-q4_k_m")
    provider = LlamaCppProvider(config, spec=spec, **kwargs)
    return provider, spec


def test_provider_points_at_the_local_port(tmp_path: Path) -> None:
    provider, spec = provider_for(tmp_path)
    assert provider.config.base_url == spec.base_url
    assert "127.0.0.1" in provider.config.base_url, "the local brain never leaves the machine"


def test_provider_reports_why_it_cannot_run(tmp_path: Path) -> None:
    provider, _ = provider_for(tmp_path, spec=make_spec(tmp_path, binary=False))
    assert provider.is_available is False
    assert "llama-server" in provider.unavailable_reason()


def test_ensure_ready_returns_false_instead_of_raising(tmp_path: Path) -> None:
    """A missing local model must not break a turn — the cloud answers instead."""
    provider, _ = provider_for(tmp_path, spec=make_spec(tmp_path, model=False))

    async def run() -> None:
        assert await provider.ensure_ready() is False

    asyncio.run(run())


async def test_ensure_ready_starts_the_sidecar_through_the_lease(tmp_path: Path) -> None:
    processes: list[FakeProcess] = []

    def spawn(argv, **kwargs):
        process = FakeProcess(argv, **kwargs)
        processes.append(process)
        return process

    engine = LlamaServerEngine(make_spec(tmp_path), popen=spawn, waiter=_always_healthy)
    lease = ResourceLease(free_ram_mb=lambda: 8192)
    provider, _ = provider_for(tmp_path, engine=engine, lease=lease)

    assert await provider.ensure_ready() is True
    assert engine.is_loaded() is True
    assert lease.loaded_names() == ["llama-server"], "the process is leased, not leaked"

    await provider.aclose()
    assert engine.is_loaded() is False
    assert lease.loaded_names() == [], "closing the provider releases the RAM"


async def test_a_tight_machine_refuses_instead_of_swapping(tmp_path: Path) -> None:
    """8 GB total, ~2 GB free: the lease is what keeps Atlas from thrashing."""
    engine = LlamaServerEngine(make_spec(tmp_path), waiter=_always_healthy)
    lease = ResourceLease(free_ram_mb=lambda: 700, ram_floor_mb=400)
    provider, _ = provider_for(tmp_path, engine=engine, lease=lease)

    assert await provider.ensure_ready() is False, "no room → say so, use the cloud"
    assert engine.is_loaded() is False


async def test_health_says_the_sidecar_is_not_running(tmp_path: Path) -> None:
    provider, _ = provider_for(tmp_path)
    report = await provider.health()
    assert report.ok is False
    assert "sidecar" in report.detail


def test_provider_speaks_the_same_protocol_as_the_cloud_ones(tmp_path: Path) -> None:
    """Nothing above this file may need to know which brain answered."""
    from atlas_core.contracts import LlmProvider

    provider, _ = provider_for(tmp_path)
    assert isinstance(provider, LlmProvider)
    assert isinstance(LlmRequest(), LlmRequest)
