"""Resource leases: how Atlas survives on 8 GB.

Nothing heavy is resident.  An engine asks the lease manager for permission to
load, gets it (or a veto), loads, serves, and is evicted after an idle TTL or
when the governor sees free RAM dropping.

Diagram: architecture doc D15.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from atlas_core.contracts import Engine, ResourceCost
from atlas_core.errors import AtlasError

log = logging.getLogger(__name__)


class LeaseDenied(AtlasError):
    """The governor refused to load the engine right now (not enough RAM)."""


@dataclass
class LeaseEntry:
    engine: Engine
    loaded_at: float
    last_used: float
    cost: ResourceCost

    def idle_s(self, now: float | None = None) -> float:
        return (now if now is not None else time.monotonic()) - self.last_used


@dataclass
class ResourceLease:
    """Owns load/unload decisions for every engine.

    ``free_ram_mb`` is injected (the process that knows about psutil is the
    governor's host, not the kernel), which keeps this testable without psutil.
    """

    free_ram_mb: Callable[[], int] = lambda: 8192
    clock: Callable[[], float] = time.monotonic
    ram_floor_mb: int = 400
    ttl_s: float = 120.0
    entries: dict[str, LeaseEntry] = field(default_factory=dict)
    load_events: list[tuple[str, float, int]] = field(default_factory=list)
    evictions: list[tuple[str, float]] = field(default_factory=list)

    # ── queries ──────────────────────────────────────────────────────
    def is_loaded(self, engine: Engine) -> bool:
        return engine.name in self.entries

    def loaded_names(self) -> list[str]:
        return sorted(self.entries)

    def resident_mb(self) -> int:
        return sum(entry.cost.ram_mb for entry in self.entries.values())

    # ── loading ──────────────────────────────────────────────────────
    async def acquire(self, engine: Engine, *, force: bool = False) -> None:
        """Load `engine` if allowed. Raises LeaseDenied when RAM is too tight."""
        if engine.is_loaded():
            self.touch(engine)
            return

        cost = engine.cost_hint()
        if not force and not self._affordable(cost):
            # Evicting expired leases is free and usually enough.
            await self.async_evict_idle()
            if not self._affordable(cost):
                raise LeaseDenied(
                    f"{engine.name} needs {cost.ram_mb} MB, "
                    f"free={self.free_ram_mb()} MB, floor={self.ram_floor_mb} MB"
                )

        started = self.clock()
        await engine.load()
        load_ms = (self.clock() - started) * 1000
        self.entries[engine.name] = LeaseEntry(engine, started, started, cost)
        self.load_events.append((engine.name, load_ms, cost.ram_mb))
        log.info("lease_load engine=%s ram_mb=%d ms=%.0f", engine.name, cost.ram_mb, load_ms)

    def _affordable(self, cost: ResourceCost) -> bool:
        return self.free_ram_mb() - cost.ram_mb >= self.ram_floor_mb

    async def release(self, engine: Engine) -> None:
        if engine.name not in self.entries:
            return
        del self.entries[engine.name]
        await engine.unload()
        self.evictions.append((engine.name, self.clock()))
        log.info("lease_release engine=%s", engine.name)

    # ── lifetime ─────────────────────────────────────────────────────
    def touch(self, engine: Engine) -> None:
        if entry := self.entries.get(engine.name):
            entry.last_used = self.clock()

    def expired(self, *, now: float | None = None) -> list[str]:
        now = now if now is not None else self.clock()
        return [name for name, entry in self.entries.items() if entry.idle_s(now) >= self.ttl_s]

    async def async_evict_idle(self, *, now: float | None = None) -> list[str]:
        """Unload every lease past its idle TTL. Returns the evicted names."""
        evicted: list[str] = []
        for name in self.expired(now=now):
            entry = self.entries.pop(name)
            await entry.engine.unload()
            self.evictions.append((name, now if now is not None else self.clock()))
            evicted.append(name)
            log.info("lease_evict engine=%s", name)
        return evicted

    async def release_all(self) -> None:
        for name in list(self.entries):
            engine = self.entries[name].engine
            await self.release(engine)

    def snapshot(self) -> dict[str, object]:
        return {
            "loaded": self.loaded_names(),
            "resident_mb": self.resident_mb(),
            "free_ram_mb": self.free_ram_mb(),
            "loads": len(self.load_events),
            "evictions": len(self.evictions),
        }


@dataclass
class ResourceGovernor:
    """Decides the operating mode (lean/bunker) and enforces the RAM floor.

    Kept separate from the lease so policy (when to degrade) and mechanism
    (load/unload) never get tangled.
    """

    lease: ResourceLease
    free_ram_mb: Callable[[], int]
    degrade_below_mb: int = 500
    mode: str = "lean"
    degraded: bool = False
    reasons: list[str] = field(default_factory=list)

    def sample(self) -> bool:
        """One governance tick. Returns True when the degraded flag changed."""
        free = self.free_ram_mb()
        should_degrade = free < self.degrade_below_mb
        if should_degrade == self.degraded:
            return False
        self.degraded = should_degrade
        self.mode = "degraded" if should_degrade else "lean"
        self.reasons.append(f"free_ram_mb={free}")
        log.warning("governor_mode=%s free_mb=%d", self.mode, free)
        return True

    def can_afford(self, cost: ResourceCost) -> bool:
        return self.free_ram_mb() - cost.ram_mb >= self.lease.ram_floor_mb

    def snapshot(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "degraded": self.degraded,
            "free_ram_mb": self.free_ram_mb(),
            "reasons": self.reasons[-3:],
            "lease": self.lease.snapshot(),
        }


__all__ = ["LeaseDenied", "LeaseEntry", "ResourceGovernor", "ResourceLease"]
