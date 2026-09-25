"""Latency instrumentation.

One line of JSON per turn/stage in ``timings.jsonl``.  The plan's rule
(anti-pattern #13): latency gets fixed with data, not with vibes.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TimingRecord:
    turn: int = 0
    wake_ms: float | None = None
    asr_ms: float | None = None
    ttft_ms: float | None = None
    tts_first_ms: float | None = None
    total_ms: float | None = None
    provider: str = ""
    mode: str = "lean"
    language: str = ""
    asr_model: str = ""
    tts_engine: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None or k == "turn"}


class TimingRecorder:
    """Append-only recorder + in-memory summary for the CLI/doctor.

    Writes are line-buffered and failure-tolerant: losing a timing line must
    never break a conversation.
    """

    def __init__(self, path: str | Path = "timings.jsonl", *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.records: list[TimingRecord] = []

    def add(self, record: TimingRecord) -> None:
        self.records.append(record)
        if not self.enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
        except OSError:  # pragma: no cover - disk full / permission denied
            self.enabled = False

    def last(self) -> TimingRecord | None:
        return self.records[-1] if self.records else None

    def summary(self) -> dict[str, dict[str, float]]:
        """p50/p95 per stage — the numbers you actually tune."""
        stages = ("wake_ms", "asr_ms", "ttft_ms", "tts_first_ms", "total_ms")
        out: dict[str, dict[str, float]] = {}
        for stage in stages:
            values = [
                float(value)
                for record in self.records
                if (value := getattr(record, stage)) is not None
            ]
            if values:
                out[stage] = {
                    "n": float(len(values)),
                    "p50": round(statistics.median(values), 1),
                    "p95": round(_percentile(values, 95), 1),
                    "max": round(max(values), 1),
                }
        return out

    def format_summary(self) -> str:
        summary = self.summary()
        if not summary:
            return "no timings recorded yet"
        lines = [f"{'stage':<14}{'n':>4}{'p50':>9}{'p95':>9}{'max':>9}"]
        for stage, stats in summary.items():
            lines.append(
                f"{stage:<14}{int(stats['n']):>4}{stats['p50']:>9.0f}{stats['p95']:>9.0f}{stats['max']:>9.0f}"
            )
        return "\n".join(lines)


def _percentile(values: list[float], pct: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["TimingRecord", "TimingRecorder", "now_iso"]
