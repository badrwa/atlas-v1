#!/usr/bin/env python3
"""Static checks for the orb's JavaScript and CSS (no Node in this project).

    python scripts/check_ui_assets.py

Exits 1 and prints what is wrong, so CI fails on a truncated file or a drifted
protocol instead of shipping a blank window.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("atlas-core", "atlas-ui"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))

from atlas_ui.assets import check_assets  # noqa: E402


def main() -> int:
    report = check_assets()
    for line in report.as_lines():
        print(line)
    for problem in report.problems:
        print(f"  · {problem}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
