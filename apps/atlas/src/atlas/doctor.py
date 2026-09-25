"""`atlas doctor` — the honest state of this machine.

The rule: a check may only report what it measured.  No microphone installed is
a *warning* with the level where it gets fixed, not a failure; a missing
provider key is a warning; a broken config is a failure.  Doctor must stay under
three seconds and never require a network call, or nobody runs it.
"""

from __future__ import annotations

import os
import platform
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from atlas.console import GREEN, RED, YELLOW, Console
from atlas_core.config import AppConfig, ConfigError, load_config

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    hint: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, hint: str = "") -> None:
        self.checks.append(Check(name, status, detail, hint))

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == FAIL]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == WARN]

    def counts(self) -> tuple[int, int, int]:
        return (
            sum(1 for c in self.checks if c.status == OK),
            len(self.warnings),
            len(self.failures),
        )


def _free_ram_mb() -> tuple[int, int] | None:
    """Free and total RAM in MB, measured — psutil if present, else the OS."""
    try:
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.available / 1024 / 1024), int(memory.total / 1024 / 1024)
    except ImportError:
        pass
    if os.name == "nt":  # pragma: no cover - Windows only
        return None
    try:
        fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, rest = line.partition(":")
            fields[key.strip()] = int(rest.strip().split()[0])
        total = fields["MemTotal"] // 1024
        free = (fields.get("MemAvailable", fields.get("MemFree", 0))) // 1024
        return free, total
    except (OSError, ValueError, KeyError):
        return None


def check_python(report: Report) -> None:
    version = platform.python_version()
    bits = platform.architecture()[0]
    if bits != "64bit":
        report.add("Python", FAIL, f"{version} ({bits})", "install 64-bit Python 3.11+")
        return
    report.add("Python", OK, f"{version} ({bits}) on {platform.system()}")


def check_hardware(report: Report) -> None:
    ram = _free_ram_mb()
    if ram is None:
        report.add("RAM", WARN, "not measurable", "install psutil for an exact number")
    else:
        free, total = ram
        detail = f"{free / 1024:.1f} GB free of {total / 1024:.1f} GB"
        if free < 1500:
            report.add("RAM", FAIL, detail, "close a browser tab — Atlas needs ~1.5 GB free")
        elif free < 2048:
            report.add("RAM", WARN, detail, "lean profile: cloud brain, small local models only")
        else:
            report.add("RAM", OK, detail)

    try:
        usage = shutil.disk_usage(Path.home())
        free_gb = usage.free / 1024**3
        if free_gb < 5:
            report.add("disk", WARN, f"{free_gb:.1f} GB free", "models need ~2 GB (L2/L3)")
        else:
            report.add("disk", OK, f"{free_gb:.1f} GB free")
    except OSError:
        report.add("disk", WARN, "not measurable")


def check_gpu(report: Report) -> None:
    """This laptop has no usable AI accelerator. Say so, once, clearly."""
    if platform.system() != "Windows":
        report.add("GPU", OK, "not Windows — no GPU assumptions made")
        return
    report.add(
        "GPU",
        OK,
        "CPU-only by design (HD 520 has no AI accelerator)",
        "cloud brain + ONNX INT8 local models",
    )


def check_config(report: Report, config: AppConfig | None, error: str = "") -> None:
    if config is None:
        report.add("config.toml", FAIL, error or "not found", "copy config.toml from the repo root")
        return
    report.add("config.toml", OK, f"profile={config.app.profile} language={config.app.language}")

    configured = [p for p in config.providers if p.is_configured()]
    if not configured:
        report.add(
            "providers",
            WARN,
            f"0 of {len(config.providers)} have keys",
            "chat falls back to a canned reply until a key is in .env",
        )
    else:
        names = ", ".join(p.name for p in configured)
        report.add(
            "providers",
            OK,
            f"{len(configured)} of {len(config.providers)} have keys: {names}",
        )

    # A key that is present but the wrong shape fails later, confusingly.
    for provider in config.providers:
        key = provider.api_key()
        if provider.kind == "gemini" and key and not key.startswith("AIza"):
            report.add(
                f"{provider.name} key",
                WARN,
                f"starts with {key[:3]!r} — that is not an AI Studio key",
                "get one at aistudio.google.com/apikey (starts with AIza)",
            )


def check_vault(report: Report, config: AppConfig | None) -> None:
    if config is None or not config.obsidian.vault_path:
        report.add("vault", WARN, "not configured", "atlas vault init ~/AtlasVault")
        return
    path = Path(config.obsidian.vault_path).expanduser()
    if not path.is_dir():
        report.add("vault", WARN, f"missing: {path}", "atlas vault init <path>")
        return
    if not os.access(path, os.W_OK):
        report.add("vault", FAIL, f"not writable: {path}")
        return
    notes = sum(1 for _ in path.rglob("*.md"))
    git = "git" if (path / ".git").exists() else "no git"
    report.add("vault", OK, f"{notes} notes, {git}")


def check_audio(report: Report) -> None:
    try:
        from atlas_audio.devices import list_devices
    except ImportError:
        report.add("audio", WARN, "atlas-audio not installed")
        return
    microphones, speakers, host_apis = list_devices()
    if not speakers and not microphones:
        report.add(
            "audio",
            WARN,
            "no devices visible",
            "voice lands in L2 — pip install 'atlas-audio[audio]' to try it",
        )
        return
    apis = f" · {', '.join(host_apis)}" if host_apis else ""
    report.add("audio", OK, f"{len(microphones)} in, {len(speakers)} out{apis}")


def check_workspace(report: Report, config: AppConfig | None) -> None:
    root = Path.cwd()
    for name in ("data", "models", "logs"):
        target = root / name
        try:
            target.mkdir(exist_ok=True)
        except OSError as exc:
            report.add(f"{name}/", FAIL, str(exc))
            continue
        report.add(f"{name}/", OK, str(target.relative_to(root)) if target.is_relative_to(root) else str(target))

    if config is not None:
        keywords = ", ".join(config.wake.keywords) or "—"
        state = "enabled" if config.wake.enabled else "disabled"
        report.add("wake word", OK, f"{keywords} ({state}, listens from L2)")


def check_environment(report: Report) -> None:
    """Secrets belong in .env; the repo must never contain them."""
    env_file = Path(".env")
    if not env_file.exists():
        report.add(".env", WARN, "missing", "cp .env.example .env and add your keys")
        return
    keys = 0
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            _, _, value = stripped.partition("=")
            if value.strip():
                keys += 1
    report.add(".env", OK, f"{keys} values set (never printed)")


def run_checks(*, config_path: str | None = None) -> Report:
    report = Report()
    check_python(report)
    check_hardware(report)
    check_gpu(report)

    config: AppConfig | None = None
    error = ""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        error = str(exc)

    check_config(report, config, error)
    check_vault(report, config)
    check_audio(report)
    check_workspace(report, config)
    check_environment(report)
    return report


def render(report: Report, console: Console, *, config_path: str | None = None) -> None:
    console.title("ATLAS doctor")
    console.write(console.paint(f"config: {config_path or 'config.toml'}", "\033[2m"))
    console.write()
    for check in report.checks:
        if check.status == OK:
            console.ok(check.name, check.detail)
        elif check.status == WARN:
            console.warn(check.name, check.detail)
            if check.hint:
                console.write(f"  {console.paint('→ ' + check.hint, YELLOW)}")
        else:
            console.fail(check.name, check.detail)
            if check.hint:
                console.write(f"  {console.paint('→ ' + check.hint, RED)}")

    ok, warn, fail = report.counts()
    console.write()
    summary = f"{ok} ok · {warn} warnings · {fail} failures"
    if fail:
        console.write(console.paint(f"✗ {summary}", RED))
    elif warn:
        console.write(console.paint(f"✓ {summary}", YELLOW))
    else:
        console.write(console.paint(f"✓ {summary}", GREEN))

    if fail:
        console.write("Fix the ✗ rows first; Atlas will not start otherwise.")
    elif warn:
        console.write("Warnings are survivable — Atlas runs, with fewer capabilities.")
    else:
        console.write("Everything Atlas needs right now is in place.")
