"""Logging setup — readable in a terminal, greppable in a file.

No third-party logging library: the plan's dependency rule wins.  Structured
enough to debug a turn, cheap enough to leave on all day.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import ClassVar

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


class _ColorFormatter(logging.Formatter):
    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\033[38;5;244m",
        logging.INFO: "\033[0m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self, *, color: bool) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if not self.color:
            return text
        return f"{self.COLORS.get(record.levelno, '')}{text}{self.RESET}"


def setup_logging(
    level: str = "INFO",
    *,
    log_dir: str | Path | None = None,
    color: bool | None = None,
    quiet_third_party: bool = True,
) -> logging.Logger:
    """Configure the root logger. Idempotent — safe to call twice."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    root.setLevel(_LEVELS.get(level.upper(), logging.INFO))
    use_color = sys.stderr.isatty() if color is None else color

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(_ColorFormatter(color=use_color))
    root.addHandler(console)

    if log_dir:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(directory / "atlas.log", encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(name)s %(message)s")
        )
        root.addHandler(file_handler)

    if quiet_third_party:
        for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "watchdog"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return root


__all__ = ["setup_logging"]
