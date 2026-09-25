"""Latency instrumentation.

One line of JSON per turn/stage in ``timings.jsonl``.  The plan's rule
(anti-pattern #13): latency gets fixed with data, not with vibes.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from atlas_core.jsonl import JsonlLog


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


class TimingRecorder(JsonlLog[TimingRecord]):
    """Append-only recorder + in-memory summary for the CLI/doctor.

    Writes are line-buffered and failure-tolerant: losing a timing line must
    never break a conversation.  `load=False` by default — a summary is about
    *this* run, unlike the wake log, which needs its history.
    """

    def __init__(self, path: str | Path = "timings.jsonl", *, enabled: bool = True) -> None:
        super().__init__(path, enabled=enabled)

    def _from_dict(self, data: dict[str, Any]) -> TimingRecord:
        return TimingRecord(**data)

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
                    "p95": round(percentile(values, 95), 1),
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


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile.

    Public because the ASR bench needs the same number the timing summary shows:
    two definitions of p95 would let the notes and the CLI disagree.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (pct / 100) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


#: Kept so an old import keeps working; new code should use `percentile`.
_percentile = percentile


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["TimingRecord", "TimingRecorder", "now_iso"]
