"""Tiny ANSI console helpers — no `rich`, no colour library, no dependency debt.

Windows 10's console understands ANSI once virtual-terminal processing is on;
we enable it when possible and degrade to plain text otherwise.
"""

from __future__ import annotations

import os
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name != "nt":
        return True
    try:  # Windows 10+: turn on virtual terminal sequences
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


USE_COLOR = _supports_color()


def paint(text: str, *codes: str) -> str:
    if not USE_COLOR or not codes:
        return text
    return "".join(codes) + text + RESET


def ok(text: str) -> str:
    return paint(f"✓ {text}", GREEN)


def warn(text: str) -> str:
    return paint(f"! {text}", YELLOW)


def fail(text: str) -> str:
    return paint(f"✗ {text}", RED)


def info(text: str) -> str:
    return paint(text, DIM)


def title(text: str) -> str:
    return paint(text, BOLD, CYAN)


def say(text: str) -> str:
    return paint(text, MAGENTA)


ICONS = {"ok": ok(""), "warn": warn(""), "fail": fail(""), "skip": info("· "), "info": info("· ")}


def status_line(name: str, status: str, detail: str = "", hint: str = "") -> str:
    """One doctor row: icon, check name, detail, and how to fix it."""
    icon = ICONS.get(status, ICONS["info"])
    line = f"{icon}{name:<28}{detail}"
    if hint and status in {"warn", "fail"}:
        line += "\n" + paint(f"    → {hint}", DIM)
    return line


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    head = "  ".join(paint(str(h).ljust(widths[i]), BOLD) for i, h in enumerate(headers))
    body = [
        "  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row))
        for row in rows
    ]
    return "\n".join([head, *body])


def write(text: str = "") -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def stream(text: str) -> None:
    """Print a streamed token without newlines (used by chat)."""
    sys.stdout.write(text)
    sys.stdout.flush()


__all__ = [
    "BOLD",
    "CYAN",
    "DIM",
    "GREEN",
    "MAGENTA",
    "RED",
    "RESET",
    "USE_COLOR",
    "YELLOW",
    "fail",
    "info",
    "ok",
    "paint",
    "say",
    "status_line",
    "stream",
    "table",
    "title",
    "warn",
    "write",
]
