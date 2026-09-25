#!/usr/bin/env python3
"""Keep PyTorch out of the core environment (plan anti-pattern #1).

An i5-6200U with 8 GB of RAM and an Intel HD 520 gains nothing from torch, and
loses 2.5–4 GB of disk plus ~1 GB of RSS.  This script fails the build if torch
(or a package that drags it in) appears in the core dependency set.

The one sanctioned exception is the isolated DarijaTTS sidecar venv
(`vendor/darija-tts/.venv`), which is never imported by Atlas.
"""

from __future__ import annotations

from pathlib import Path

FORBIDDEN = {"torch", "torchaudio", "torchvision", "tensorflow", "jax", "nvidia-cublas-cu12"}
ALLOWED_PATHS = ("vendor/", "packages/atlas-audio/")  # sidecars & optional extras may mention them


def find_violations(root: Path = Path()) -> list[str]:
    violations: list[str] = []
    for pyproject in sorted(root.glob("**/pyproject.toml")):
        relative = pyproject.as_posix()
        if any(allowed in relative for allowed in ALLOWED_PATHS):
            continue
        if ".venv" in relative or "node_modules" in relative:
            continue
        text = pyproject.read_text(encoding="utf-8").lower()
        for package in FORBIDDEN:
            if f'"{package}' in text or f"'{package}" in text:
                violations.append(f"{relative}: declares forbidden dependency {package!r}")
    return violations


def installed_violations() -> list[str]:
    """Also check the *running* environment — the real proof."""
    import importlib.util

    return [f"runtime: {name} is importable in the core venv" for name in FORBIDDEN
            if importlib.util.find_spec(name.replace("-", "_")) is not None]


def main() -> int:
    violations = find_violations() + installed_violations()
    if violations:
        print("Torch-free policy violated:")
        for violation in violations:
            print(f"  ✗ {violation}")
        return 1
    print("✓ core environment is torch-free")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
