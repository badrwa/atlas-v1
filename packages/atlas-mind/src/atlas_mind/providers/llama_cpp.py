"""The local llama.cpp sidecar — Atlas's brain when the internet is gone.

Bunker mode is not a second-class path: it is what makes the assistant *yours*
rather than a website with a voice.  But a 1.7B model on a 2-core Skylake is a
different animal from a cloud endpoint, so this file is deliberately explicit
about three things:

* **It is leased, not resident.** `ResourceLease` decides whether the ~1.2 GB
  fits in RAM next to whatever else is running; the model loads on demand and is
  unloaded when idle (L10 tunes the timing).
* **It never pretends.** No binary or no model file → the provider reports
  exactly which file is missing and the exact command to fetch it, and the
  router moves on to the next brain.
* **It answers the same protocol.** Same `LlmEvent` stream as Gemini and Groq,
  so nothing above this file knows or cares which one answered.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from atlas_core.config import ProviderConfig
from atlas_core.contracts import Engine, HealthReport, ResourceCost
from atlas_core.errors import Unsupported
from atlas_core.resources import ResourceLease
from atlas_mind.providers.openai_compatible import OpenAiCompatibleProvider

log = logging.getLogger(__name__)

DEFAULT_PORT = 8123
DEFAULT_THREADS = 2  # i5-6200U has 2 cores / 4 threads; more threads than cores hurts


@dataclass(slots=True)
class SidecarSpec:
    """Where the binary and the model live, and how to start them."""

    binary: Path | None = None
    model_path: Path | None = None
    port: int = DEFAULT_PORT
    threads: int = DEFAULT_THREADS
    host: str = "127.0.0.1"
    extra_args: list[str] = field(default_factory=list)
    ram_mb: int = 1200

    @property
    def missing(self) -> list[str]:
        """Which pieces are absent (empty list means runnable)."""
        gaps: list[str] = []
        if self.binary is None or not Path(self.binary).exists():
            gaps.append("llama-server binary")
        if self.model_path is None or not Path(self.model_path).exists():
            gaps.append("GGUF model")
        return gaps

    @property
    def is_available(self) -> bool:
        return not self.missing

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def command(self) -> list[str]:
        """The exact argv used to start the server (also printed when it fails)."""
        if self.binary is None or self.model_path is None:
            raise Unsupported(self.hint())
        return [
            str(self.binary),
            "-m",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-t",
            str(self.threads),
            *self.extra_args,
        ]

    def hint(self) -> str:
        """What the user must do — specific, not 'install the model'."""
        gaps = self.missing
        if not gaps:
            return self.base_url
        lines = [f"local brain unavailable: missing {', '.join(gaps)}."]
        lines.append("  get them (once, ~1 GB):")
        lines.append("    mkdir -p vendor/local-llm models")
        lines.append(
            "    # windows binary: https://github.com/ggml-org/llama.cpp/releases "
            "(llama-server, cpu build)"
        )
        lines.append("    # model: hf.co/Qwen/Qwen3-1.7B-GGUF → models/qwen3-1.7b-q4_k_m.gguf")
        lines.append(f"  then run: {self.binary or 'llama-server'} -m {self.model_path or 'models/x.gguf'}")
        return "\n".join(lines)


def find_default_paths(root: str | Path = ".") -> SidecarSpec:
    """Look where the plan says things live; return a spec either way."""
    base = Path(root)
    binary_names = ["llama-server.exe", "llama-server"] if sys.platform == "win32" else ["llama-server"]
    binary: Path | None = None
    for name in binary_names:
        for candidate in (base / "vendor" / "local-llm" / name, base / "vendor" / name):
            if candidate.exists():
                binary = candidate
                break
        if binary:
            break
    if binary is None:
        found = shutil.which("llama-server")
        binary = Path(found) if found else None

    models = sorted((base / "models").glob("*.gguf")) if (base / "models").is_dir() else []
    return SidecarSpec(binary=binary, model_path=models[0] if models else None)


class LlamaServerEngine(Engine):
    """The sidecar process as an `Engine`, so the lease governor can own it.

    Implements the same contract as every other leased engine (load twice is a
    no-op, unload when unloaded is a no-op, `is_loaded` tells the truth), which
    is what lets `ResourceLease` manage a subprocess and an ONNX session with
    the same code.
    """

    name = "llama-server"

    def __init__(
        self,
        spec: SidecarSpec,
        *,
        popen: Any = subprocess.Popen,
        health_timeout_s: float = 90.0,
        poll_interval_s: float = 0.5,
        waiter: Callable[[], Awaitable[bool]] | None = None,
    ) -> None:
        self.spec = spec
        self._popen = popen
        self.health_timeout_s = health_timeout_s
        self.poll_interval_s = poll_interval_s
        # The health wait is injectable so tests can exercise load/unload without
        # a 1 GB model or a real server process.
        self._wait = waiter
        self._waiter = waiter
        self._process: Any = None
        self.last_error = ""

    def is_loaded(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def cost_hint(self) -> ResourceCost:
        return ResourceCost(
            ram_mb=self.spec.ram_mb,
            cpu_threads=self.spec.threads,
            disk_mb=int(self.spec.model_path.stat().st_size / 1e6) if self.spec.model_path else 0,
            cold_start_s=12.0,
        )

    async def load(self) -> None:
        """Start the server and wait until it answers /health."""
        if self.is_loaded():
            return
        if not self.spec.is_available:
            raise Unsupported(self.spec.hint())

        command = self.spec.command()
        log.info("llama_server_start cmd=%s", " ".join(command))
        creation = 0
        if sys.platform == "win32":  # pragma: no cover - Windows only
            creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = self._popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation,
        )
        healthy = await self._waiter() if self._waiter else await self._wait_healthy()
        if healthy:
            return

        self.last_error = (
            f"llama-server did not become healthy in {self.health_timeout_s:.0f}s "
            f"({self.spec.base_url}/health)"
        )
        await self.unload()
        raise Unsupported(self.last_error)

    async def unload(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        with contextlib.suppress(Exception):
            process.terminate()
            for _ in range(20):
                if process.poll() is not None:
                    return
                await asyncio.sleep(0.1)
            process.kill()
        log.info("llama_server_stopped")

    async def _wait_healthy(self) -> bool:
        deadline = asyncio.get_running_loop().time() + self.health_timeout_s
        url = f"{self.spec.base_url.replace('/v1', '')}/health"
        async with httpx.AsyncClient(timeout=2.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                if not self.is_loaded():
                    self.last_error = "llama-server exited on startup"
                    return False
                try:
                    response = await client.get(url)
                    if response.status_code < 400:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(self.poll_interval_s)
        return False


class LlamaCppProvider(OpenAiCompatibleProvider):
    """OpenAI-compatible provider pointed at a local llama-server sidecar."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        api_key: str = "",
        client: httpx.AsyncClient | None = None,
        spec: SidecarSpec | None = None,
        engine: LlamaServerEngine | None = None,
        lease: ResourceLease | None = None,
    ) -> None:
        self.spec = spec or find_default_paths()
        if config.base_url:
            self.spec.host, port = _split_host_port(config.base_url) or (self.spec.host, self.spec.port)
            self.spec.port = port
        config = config.model_copy(update={"base_url": self.spec.base_url})
        super().__init__(config, api_key=api_key or "local", client=client)
        self.engine = engine or LlamaServerEngine(self.spec)
        self.lease = lease
        self._started_by_us = False

    # ── availability ─────────────────────────────────────────────────
    @property
    def is_available(self) -> bool:
        return self.spec.is_available

    def unavailable_reason(self) -> str:
        return self.spec.hint()

    async def ensure_ready(self) -> bool:
        """Make sure the sidecar is up before a turn. False (never raises) if not.

        A missing local model must degrade to "the cloud answers instead", not to
        an exception in the middle of someone's sentence.
        """
        if self.engine.is_loaded():
            self._touch()
            return True
        if not self.spec.is_available:
            log.info("llama_sidecar_unavailable reason=%s", ", ".join(self.spec.missing))
            return False
        try:
            if self.lease is not None:
                await self.lease.acquire(self.engine)
            else:
                await self.engine.load()
        except Exception as exc:
            log.info("llama_sidecar_load_failed error=%s", exc)
            return False
        self._started_by_us = True
        return True

    async def aclose(self) -> None:
        """Stop the sidecar only if this session started it."""
        if self._started_by_us:
            if self.lease is not None:
                await self.lease.release(self.engine)
            else:
                await self.engine.unload()
            self._started_by_us = False
        await super().aclose()

    # ── health ───────────────────────────────────────────────────────
    async def health(self) -> HealthReport:
        if not self.spec.is_available:
            return HealthReport(ok=False, detail="; ".join(self.spec.missing) + " missing")
        base = self.spec.base_url.replace("/v1", "")
        try:
            response = await self.client.get(f"{base}/health", timeout=3.0)
        except httpx.HTTPError:
            return HealthReport(ok=False, detail="sidecar not running (starts on demand)")
        return HealthReport(ok=response.status_code < 400, detail=f"http {response.status_code}")

    # ── plumbing ─────────────────────────────────────────────────────
    def _touch(self) -> None:
        if self.lease is not None:
            self.lease.touch(self.engine)


def _split_host_port(base_url: str) -> tuple[str, int] | None:
    """Read host/port out of a configured base_url, if it has them."""
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    if not parsed.hostname:
        return None
    port = parsed.port or (443 if parsed.scheme == "https" else DEFAULT_PORT)
    return parsed.hostname, port


__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_THREADS",
    "LlamaCppProvider",
    "LlamaServerEngine",
    "SidecarSpec",
    "find_default_paths",
]
