#!/usr/bin/env python3
"""Enforce the architecture's import rules as a build failure, not a promise.

    atlas_core      → stdlib + third-party only (it is the kernel)
    atlas_audio     → atlas_core
    atlas_obsidian  → atlas_core
    atlas_ui        → atlas_core
    atlas_skills    → atlas_core, atlas_obsidian
    atlas_mind      → atlas_core, atlas_audio, atlas_obsidian, atlas_skills
    atlas (app)     → anything (it is the composition root)

Usage: python scripts/check_import_rules.py [packages_dir]
Exit code 1 when a violation is found (CI + `atlas doctor` both use this).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

RULES: dict[str, set[str]] = {
    "atlas_core": set(),
    "atlas_audio": {"atlas_core"},
    "atlas_obsidian": {"atlas_core"},
    "atlas_ui": {"atlas_core"},
    "atlas_skills": {"atlas_core", "atlas_obsidian"},
    "atlas_mind": {"atlas_core", "atlas_audio", "atlas_obsidian", "atlas_skills"},
    "atlas": {
        "atlas_core", "atlas_mind", "atlas_audio", "atlas_obsidian", "atlas_skills", "atlas_ui"
    },
}


def _imports_in(path: Path) -> set[str]:
    """Top-level module names imported by one file (stdlib + third-party + atlas)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def _package_name(path: Path) -> str | None:
    for part in path.parts:
        if part.startswith("atlas"):
            return part
    return None


def find_violations(root: Path = Path("packages")) -> list[str]:
    violations: list[str] = []
    sources = list(root.glob("*/src/**/*.py")) + list(Path("apps").glob("*/src/**/*.py"))

    for file in sources:
        owner = _package_name(file)
        if owner is None or owner not in RULES:
            continue
        allowed = RULES[owner]
        for imported in _imports_in(file):
            if not imported.startswith("atlas"):
                continue
            if imported not in allowed and imported != owner:
                violations.append(
                    f"{file}: {owner} must not import {imported} "
                    f"(allowed: {', '.join(sorted(allowed)) or 'stdlib + third-party only'})"
                )
    return sorted(set(violations))


def main() -> int:
    violations = find_violations(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("packages"))
    if violations:
        print("Import-rule violations:")
        for violation in violations:
            print(f"  ✗ {violation}")
        return 1
    print("✓ import rules hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
