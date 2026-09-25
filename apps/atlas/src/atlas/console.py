"""Terminal output: colour, tables, and one place where Windows is handled.

Atlas runs in `cmd.exe` on Windows 10 as often as in a modern terminal, so
colour is opt-in, detected, and never required.  Written once here because
hand-rolled `print(f"\\033[...")` calls scattered through the CLI are how a
project ends up printing escape codes into a log file.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Sequence
from typing import TextIO

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"


def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    try:  # pragma: no cover - Windows only
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass


def supports_color(stream: TextIO | None = None) -> bool:
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("ATLAS_FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


class Console:
    """Minimal colour-aware printer. No dependency, no global state."""

    def __init__(self, *, color: bool | None = None, stream: TextIO | None = None) -> None:
        self.stream: TextIO = stream or sys.stdout
        if color is None:
            _enable_windows_ansi()
            color = supports_color(self.stream)
        self.color = color

    def paint(self, text: str, *codes: str) -> str:
        if not self.color or not codes:
            return text
        return "".join(codes) + text + RESET

    def write(self, text: str = "", *, end: str = "\n") -> None:
        print(text, end=end, file=self.stream, flush=True)

    def ok(self, label: str, detail: str = "") -> None:
        self.write(f"{self.paint('✓', GREEN)} {label:<24} {self.paint(detail, DIM)}")

    def warn(self, label: str, detail: str = "") -> None:
        self.write(f"{self.paint('!', YELLOW)} {label:<24} {detail}")

    def fail(self, label: str, detail: str = "") -> None:
        self.write(f"{self.paint('✗', RED)} {label:<24} {detail}")

    def info(self, label: str, detail: str = "") -> None:
        self.write(f"{self.paint('·', BLUE)} {label:<24} {detail}")

    def title(self, text: str) -> None:
        self.write(self.paint(text, BOLD, CYAN))

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
        rows = [[str(cell) for cell in row] for row in rows]
        widths = [len(h) for h in headers]
        for row in rows:
            for index, cell in enumerate(row):
                if index < len(widths):
                    widths[index] = max(widths[index], len(cell))
        self.write("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
        for row in rows:
            self.write("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))

    def rule(self, width: int = 60) -> None:
        self.write(self.paint("─" * width, DIM))

    def ask(self, prompt: str) -> str:
        return input(self.paint(prompt, CYAN))
