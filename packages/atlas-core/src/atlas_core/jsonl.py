"""Append-only JSON Lines logs, and the one rule that keeps them safe.

`timings.jsonl` (L1) and `data/wake_log.jsonl` (L2) are the same object with
different record types: append one JSON object per line, keep the session in
memory for summaries, and never let a disk problem interrupt a conversation.
Written twice, those rules drift — so they live here once, and each log only
supplies its record type.

Loading is opt-in because the two logs want opposite things: a wake log must
remember previous sessions (the false-wake rate is meaningless otherwise), while
a timing summary is about *this* run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Generic, TypeVar

log = logging.getLogger(__name__)

R = TypeVar("R")


class JsonlLog(Generic[R]):
    """Base class for the append-only JSON Lines logs."""

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = True,
        load: bool = False,
    ) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.records: list[R] = []
        #: True once a record has actually reached the disk this session.
        self.persisted = False
        if enabled and load:
            self.load()

    # ── subclass hooks ───────────────────────────────────────────────
    def _from_dict(self, data: dict[str, Any]) -> R:
        """Rebuild a record from one JSON line. Must raise for junk."""
        raise NotImplementedError

    def _as_dict(self, record: R) -> dict[str, Any]:
        return record.as_dict()  # type: ignore[attr-defined]

    # ── writing ──────────────────────────────────────────────────────
    def add(self, record: R) -> None:
        """Remember a record, then persist it — never at the cost of the turn."""
        self.records.append(record)
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self._as_dict(record), ensure_ascii=False) + "\n")
            self.persisted = True
        except OSError:  # pragma: no cover - disk full / permission denied
            log.warning("jsonl_unwritable path=%s — continuing in memory", self.path)
            self.enabled = False

    # ── reading ──────────────────────────────────────────────────────
    def load(self) -> list[R]:
        """Read the file back. A torn line from a crash is skipped, not fatal."""
        if not self.path.exists():
            return self.records
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    self.records.append(self._from_dict(json.loads(line)))
                except (TypeError, ValueError):
                    continue
        except OSError:  # pragma: no cover - unreadable file
            self.enabled = False
        return self.records

    def tail(self, limit: int = 20) -> list[R]:
        return self.records[-limit:]

    def last(self) -> R | None:
        return self.records[-1] if self.records else None

    def clear(self) -> None:
        self.records.clear()


__all__ = ["JsonlLog"]
