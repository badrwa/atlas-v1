"""`atlas doctor` — the first command you run, and the one you re-run forever.

Every check answers three questions: does it work, what exactly is missing, and
what do I type to fix it.  Exit code is 0 unless something is genuinely broken.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from atlas import console
from atlas_core.config import AppConfig

OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"

MIN_RAM_FREE_MB = 2000
MIN_DISK_FREE_GB = 20


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    hint: str = ""


@dataclass
class DoctorReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", hint: str = "") -> None:
        self.results.append(CheckResult(name, status, detail, hint))

    @property
    def failures(self) -> int:
        return sum(1 for result in self.results if result.status == FAIL)

    @property
    def warnings(self) -> int:
        return sum(1 for result in self.results if result.status == WARN)

    def render(self) -> str:
        lines = [console.title("ATLAS doctor"), ""]
        lines += [
            console.status_line(result.name, result.status, result.detail, result.hint)
            for result in self.results
        ]
        lines.append("")
        summary = f"{len(self.results) - self.failures - self.warnings} ok · {self.warnings} warnings · {self.failures} failures"
        lines.append(console.ok(summary) if not self.failures else console.fail(summary))
        if self.failures:
            lines.append(console.info("fix the failures above, then run `python -m atlas doctor` again"))
        return "\n".join(lines)


# ── individual checks ────────────────────────────────────────────────
def check_python(report: DoctorReport) -> None:
    version = sys.version.split()[0]
    arch = platform.machine()
    bits = platform.architecture()[0]
    if bits != "64bit":
        report.add("Python", FAIL, f"{version} ({bits})", "install 64-bit Python 3.11/3.12 from python.org")
        return
    report.add("Python", OK, f"{version} · {arch} · {bits}")
    if not os.environ.get("VIRTUAL_ENV") and ".venv" not in sys.executable:
        report.add(
            "virtualenv",
            WARN,
            "not running inside .venv",
            "activate it first: .\\.venv\\Scripts\\Activate.ps1",
        )
    else:
        report.add("virtualenv", OK, Path(sys.executable).parent.name)


def check_hardware(report: DoctorReport) -> None:
    try:
        import psutil
    except ImportError:
        report.add("RAM", SKIP, "psutil not installed", "pip install psutil")
        return

    memory = psutil.virtual_memory()
    free_mb = int(memory.available / 1024 / 1024)
    total_gb = memory.total / 1024**3
    if free_mb < MIN_RAM_FREE_MB:
        report.add(
            "RAM",
            FAIL,
            f"{free_mb} MB free of {total_gb:.1f} GB",
            "close heavy apps (Chrome, Teams) — Atlas needs ≥ 2 GB free to run local engines",
        )
    else:
        report.add("RAM", OK, f"{free_mb} MB free of {total_gb:.1f} GB")

    disk = shutil.disk_usage(Path.home())
    free_gb = disk.free / 1024**3
    status = OK if free_gb >= MIN_DISK_FREE_GB else WARN
    report.add(
        "disk (C:)",
        status,
        f"{free_gb:.0f} GB free",
        "Atlas needs ~3 GB for a venv + models" if status != OK else "",
    )

    battery = psutil.sensors_battery()
    if battery is not None:
        state = "plugged in" if battery.power_plugged else "on battery"
        status = OK if battery.power_plugged else WARN
        report.add(
            "power",
            status,
            f"{int(battery.percent)}% · {state}",
            "always-on Atlas should stay plugged in (15 W CPU throttles on battery)",
        )


def check_audio(report: DoctorReport) -> None:
    try:
        from atlas_audio import check_audio_stack
    except ImportError:
        report.add("audio", SKIP, "atlas-audio not installed")
        return

    stack = check_audio_stack()
    if not stack.sounddevice_available:
        report.add(
            "microphone",
            SKIP,
            "sounddevice not installed",
            "L2 task: pip install 'atlas-audio[audio]'",
        )
        return
    status = OK if stack.ok else WARN
    report.add("microphone", status, stack.summary())
    inputs = ", ".join(device.name for device in stack.inputs[:2])
    if inputs:
        report.add("audio devices", OK, inputs)
    for problem in stack.problems:
        report.add("audio problem", WARN, problem)


def check_config(report: DoctorReport, config: AppConfig | None) -> None:
    if config is None:
        report.add("config.toml", FAIL, "not loaded", "run from the repo root or set ATLAS_CONFIG")
        return
    report.add(
        "config.toml",
        OK,
        f"profile={config.app.profile} · language={config.app.language} · secondary={config.app.secondary_language}",
    )


def check_providers(report: DoctorReport, config: AppConfig | None) -> None:
    if config is None:
        return
    usable = config.ordered_providers()
    enabled = [provider for provider in config.providers if provider.enabled]

    if not usable:
        report.add(
            "LLM providers",
            FAIL,
            "none usable",
            "put a key in .env (see .env.example) — free tiers are enough",
        )
    else:
        report.add(
            "LLM providers",
            OK,
            " → ".join(provider.name for provider in usable),
        )

    for provider in enabled:
        if provider.is_configured() or provider.is_local:
            continue
        report.add(
            f"key: {provider.name}",
            WARN,
            f"{provider.api_key_env} missing",
            f"add {provider.api_key_env}=... to .env (or disable the provider)",
        )

    # Specific, actionable hint for the Gemini key format.
    gemini = next((provider for provider in config.providers if provider.kind == "gemini"), None)
    if gemini is not None:
        key = gemini.api_key()
        if key and not key.startswith("AIza"):
            report.add(
                "gemini key format",
                WARN,
                f"starts with {key[:3]!r}, expected 'AIza'",
                "Google AI Studio API keys start with AIza… — an 'AQ.' value is an OAuth/CLI token "
                "and will be rejected by the Gemini API. Get one at aistudio.google.com/apikey",
            )


def check_vault(report: DoctorReport, config: AppConfig | None) -> None:
    if config is None:
        return
    path = config.obsidian.vault_path
    if not path:
        report.add(
            "Obsidian vault",
            WARN,
            "not configured (L6)",
            "create one: python -m atlas vault init D:\\AtlasVault",
        )
        return

    vault = Path(path)
    if not vault.is_dir():
        report.add("Obsidian vault", FAIL, f"{path} does not exist", "check ATLAS_VAULT_PATH or run vault init")
        return

    writable = os.access(vault, os.W_OK)
    git_repo = (vault / ".git").exists()
    detail = f"{path} · {'writable' if writable else 'READ-ONLY'} · git={'yes' if git_repo else 'no'}"
    status = OK if (writable and git_repo) else WARN
    hint = ""
    if not writable:
        hint = "Atlas cannot write memories here — pick a folder you own"
    elif not git_repo:
        hint = "no git repo: 'undo that' will not work — run git init in the vault"
    report.add("Obsidian vault", status, detail, hint)


def check_paths(report: DoctorReport) -> None:
    for name, path in (("models dir", Path(os.environ.get("ATLAS_MODELS_DIR", "models"))),):
        if path.exists():
            report.add(name, OK, str(path))
        else:
            report.add(name, SKIP, f"{path} (created in L2)")


def check_webview(report: DoctorReport) -> None:
    """WebView2 ships with Windows 10/11 — verify before planning on it (L5)."""
    if os.name != "nt":
        report.add("WebView2", SKIP, "not Windows")
        return
    candidates = [
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Microsoft/EdgeWebView/Application",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/EdgeWebView/Application",
    ]
    if any(candidate.exists() for candidate in candidates):
        report.add("WebView2", OK, "installed (orb UI can use it in L5)")
    else:
        report.add(
            "WebView2",
            WARN,
            "not found",
            "install the Evergreen WebView2 runtime — needed for the orb UI (L5)",
        )


def check_import_rules(report: DoctorReport) -> None:
    """The architecture's import rules, checked as a fact rather than a promise."""
    try:
        sys.path.insert(0, str(Path("scripts").resolve()))
        from check_import_rules import find_violations  # type: ignore[import-not-found]

        violations = find_violations(Path("packages"))
        if violations:
            report.add(
                "import rules",
                FAIL,
                f"{len(violations)} violation(s)",
                violations[0],
            )
        else:
            report.add("import rules", OK, "no layer violations")
    except Exception:
        report.add("import rules", SKIP, "scripts/check_import_rules.py not found")


CHECKS: list[Callable[..., None]] = [
    check_python,
    check_hardware,
    check_audio,
]

CONFIG_CHECKS: list[Callable[..., None]] = [
    check_config,
    check_providers,
    check_vault,
]


def run_checks(config: AppConfig | None = None) -> DoctorReport:
    report = DoctorReport()
    for check in CHECKS:
        check(report)
    for check in CONFIG_CHECKS:
        check(report, config)
    check_paths(report)
    check_webview(report)
    check_import_rules(report)
    return report


__all__ = ["MIN_DISK_FREE_GB", "MIN_RAM_FREE_MB", "DoctorReport", "run_checks"]
