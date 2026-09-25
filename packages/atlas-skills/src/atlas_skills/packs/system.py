"""System skills — the ones that make Atlas feel like it lives in your PC.

Deliberately small and honest, because this is an 8 GB machine: knowing how much
RAM is actually free is a feature, not a toy.
"""

from __future__ import annotations

import logging
import shutil
import webbrowser
from typing import Any
from urllib.parse import urlparse

from atlas_core.contracts import Permission, Skill, SkillContext, SkillResult, ToolSpec

log = logging.getLogger(__name__)

ALLOWED_SCHEMES = ("http", "https")


class SystemStatsSkill(Skill):
    """RAM / disk / battery — the numbers this laptop actually cares about."""

    name = "system_stats"
    permission = Permission.SAFE

    def __init__(self, ram_probe=None, disk_probe=None, battery_probe=None) -> None:
        # Probes are injected so the skill is testable without psutil.
        self._ram = ram_probe or _psutil_ram
        self._disk = disk_probe or _shutil_disk
        self._battery = battery_probe or _psutil_battery

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Report free RAM, disk space and battery level of this PC.",
            description_darija="عطيني شحال باقي ميموار، ديسك، و لاباطري.",
            parameters={
                "type": "object",
                "properties": {
                    "what": {
                        "type": "string",
                        "enum": ["all", "ram", "disk", "battery"],
                        "description": "which metric to report",
                    }
                },
            },
            permission=self.permission,
        )

    def invoke(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        what = str(args.get("what", "all"))
        facts: dict[str, float] = {}
        parts: list[str] = []

        if what in ("all", "ram"):
            total, free = self._ram()
            facts.update(ram_total_mb=total, ram_free_mb=free)
            parts.append(f"RAM: {free} MB free of {total} MB")
        if what in ("all", "disk"):
            total, free = self._disk()
            facts.update(disk_total_gb=total, disk_free_gb=free)
            parts.append(f"disk: {free} GB free of {total} GB")
        if what in ("all", "battery"):
            percent, plugged = self._battery()
            facts.update(battery_percent=percent, plugged=int(plugged))
            parts.append(f"battery: {percent}%{' (charging)' if plugged else ''}")

        if not parts:
            return SkillResult(ok=False, spoken=f"ma3reftch chno {what}", data=facts)
        return SkillResult(ok=True, spoken=" · ".join(parts), data=facts)


class OpenUrlSkill(Skill):
    """Open a link in the default browser. Schema-constrained, so no surprises."""

    name = "open_url"
    permission = Permission.SAFE

    def __init__(self, opener=None) -> None:
        self._open = opener or webbrowser.open
        self.opened: list[str] = []

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Open an http/https URL in the default browser.",
            description_darija="حل لينك ف لبراوزر.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "http or https URL"}},
                "required": ["url"],
            },
            permission=self.permission,
        )

    def invoke(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        url = str(args.get("url", "")).strip()
        scheme = urlparse(url).scheme.lower()
        if scheme not in ALLOWED_SCHEMES:
            return SkillResult(ok=False, spoken="ghir http wla https", data={"url": url})
        if ctx.dry_run:
            return SkillResult(ok=True, spoken=f"[dry-run] + {url}", data={"url": url, "dry_run": True})
        opened = bool(self._open(url))
        self.opened.append(url)
        return SkillResult(
            ok=opened,
            spoken=f"hlit {url}" if opened else "ma9dertch nhalha",
            data={"url": url},
        )


class RememberSkill(Skill):
    """Write a fact to the Obsidian vault (via the vault adapter, L6)."""

    name = "remember"
    permission = Permission.SAFE

    def __init__(self, vault) -> None:
        self.vault = vault

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Save a durable fact about the owner into their notes.",
            description_darija="سجل معلومة عليا ف النوطات.",
            parameters={
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "the fact, one sentence"}},
                "required": ["fact"],
            },
            permission=self.permission,
        )

    def invoke(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        fact = str(args.get("fact", "")).strip()
        if len(fact) < 3:
            return SkillResult(ok=False, spoken="ma sme3tch chi haja", data={})
        if ctx.dry_run:
            return SkillResult(ok=True, spoken=f"[dry-run] nsejel: {fact}", data={"fact": fact})
        path = self.vault.remember(fact)
        return SkillResult(
            ok=True, spoken="safi, dert note", data={"fact": fact, "path": str(path)}
        )


class ShutdownSkill(Skill):
    """The canonical CONFIRM skill — destructive, owner-only, never silent."""

    name = "shutdown_pc"
    permission = Permission.CONFIRM
    owner_only = True

    def __init__(self, action=None) -> None:
        self._action = action or _noop_action

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description="Shut the PC down (asks for spoken confirmation first).",
            description_darija="طفي البيسي (كيطلب تأكيد قبل).",
            parameters={"type": "object", "properties": {}},
            permission=self.permission,
        )

    def invoke(self, args: dict[str, Any], ctx: SkillContext) -> SkillResult:
        if ctx.dry_run:
            return SkillResult(ok=True, spoken="[dry-run] kont ghadi ntfi", data={"dry_run": True})
        done = self._action()
        return SkillResult(ok=bool(done), spoken="safi, kaytfi daba", data={})


# ── default probes (imported lazily so the package works without psutil) ──
def _psutil_ram() -> tuple[int, int]:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is optional
        return (0, 0)
    memory = psutil.virtual_memory()
    return (int(memory.total / 1024 / 1024), int(memory.available / 1024 / 1024))


def _shutil_disk() -> tuple[int, int]:
    usage = shutil.disk_usage("/")
    gb = 1024**3
    return (int(usage.total / gb), int(usage.free / gb))


def _psutil_battery() -> tuple[int, bool]:
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return (0, False)
    battery = psutil.sensors_battery()
    if battery is None:
        return (0, True)
    return (int(battery.percent), bool(battery.power_plugged))


def _noop_action() -> bool:  # pragma: no cover - replaced in production wiring
    log.warning("shutdown_action_not_wired")
    return False


__all__ = ["OpenUrlSkill", "RememberSkill", "ShutdownSkill", "SystemStatsSkill"]
