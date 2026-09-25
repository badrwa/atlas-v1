"""The TTS cache — because the same lines come back, and they should be instant.

Atlas repeats itself more than it seems: greetings, "safi", "daba", "3ndek zwe9?",
the time, "mafhemtshch, 3awed", the continuation prompt.  Those lines are the
*difference* between a cached reply (play a WAV, < 200 ms) and a synthesis round
(≈ half a second of CPU on this laptop).  The plan's gate is exactly that number,
so the cache is not an optimisation — it is part of the latency budget.

Design choices worth stating:

* **SQLite, one row per clip.** No index file to corrupt, atomic writes, and the
  LRU is a single `DELETE ... ORDER BY used_at LIMIT` query.  A directory of WAVs
  with a JSON index would need its own crash story; SQLite already has one.
* **Key = sha256(normalised text + voice + rate + engine).** Rate is in the key
  on purpose: a sentence spoken softly at 0.92× is *different audio* and must not
  be served as the calm version.
* **A connection per operation.** Synthesis happens on a worker thread, and
  sqlite3 connections are not shareable across threads by default.  Opening one
  per call costs microseconds and removes a whole class of "database is locked"
  bugs; the WAL journal keeps concurrent readers happy.
* **LRU to a byte cap**, not a row cap: a 3-minute answer and a "safi" are not
  the same size.  `prune()` runs after writes, cheaply (it is one query, and only
  when the total says it might be needed).
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    key        TEXT PRIMARY KEY,
    engine     TEXT NOT NULL,
    voice      TEXT NOT NULL,
    text       TEXT NOT NULL,
    sample_rate INTEGER NOT NULL,
    pcm        BLOB NOT NULL,
    bytes      INTEGER NOT NULL,
    created_at REAL NOT NULL,
    used_at    REAL NOT NULL,
    uses       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS clips_used ON clips (used_at);
"""


def cache_key(text: str, *, voice: str = "", language: str = "", engine: str = "", rate: float = 1.0) -> str:
    """Stable across processes and runs — that is the whole requirement.

    Normalisation is deliberately light (case, surrounding space and repeated
    whitespace): the *words* are what matter, and a punctuation-only difference
    legitimately changes how a sentence is read.
    """
    payload = "\x1f".join(
        (" ".join(text.split()).lower(), voice, language, engine, f"{rate:.3f}")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CachedClip:
    pcm: bytes
    sample_rate: int
    key: str


@dataclass(slots=True)
class CacheStats:
    entries: int = 0
    bytes: int = 0
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "entries": self.entries,
            "mb": round(self.bytes / 1e6, 2),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 3),
        }


class TtsCache:
    """A small, honest LRU over synthesised speech."""

    #: An in-memory database only exists while a connection is open, and a
    #: connection per operation would therefore see an empty database every time.
    #: The shared-cache URI plus one keeper connection is what makes `:memory:`
    #: behave like the real thing — and the name is unique per instance, because
    #: a *shared* in-memory database is shared by the whole process: two caches
    #: in one test run would otherwise read each other's clips.
    MEMORY_TEMPLATE = "file:{name}?mode=memory&cache=shared"

    def __init__(self, path: str | Path = "data/tts-cache.sqlite3", *, max_mb: float = 300.0) -> None:
        self.path = Path(path)
        self.memory = str(path) == ":memory:"
        self.max_bytes = int(max_mb * 1_000_000)
        self.hits = 0
        self.misses = 0
        self.bytes = 0
        self._keeper: sqlite3.Connection | None = None
        self._uri = self.MEMORY_TEMPLATE.format(name=f"atlas-tts-{uuid4().hex}")
        if not self.memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._keeper = sqlite3.connect(self._uri, uri=True, check_same_thread=False)
        self._init()

    # ── plumbing ─────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        """A fresh connection per operation: synthesis runs on a worker thread,
        and sqlite3 connections are not shareable across threads by default."""
        if self.memory:
            return sqlite3.connect(self._uri, uri=True, check_same_thread=False)
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
        self.bytes = self._total_bytes()

    def _total_bytes(self) -> int:
        try:
            with self._connect() as connection:
                row = connection.execute("SELECT COALESCE(SUM(bytes), 0) FROM clips").fetchone()
        except sqlite3.DatabaseError as exc:  # pragma: no cover - disk failure
            log.warning("tts_cache_unavailable error=%s", exc)
            return 0
        return int(row[0]) if row else 0

    # ── reads and writes ─────────────────────────────────────────────
    def get(self, key: str) -> CachedClip | None:
        """Return the clip and mark it used (that *is* the LRU)."""
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT pcm, sample_rate FROM clips WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    self.misses += 1
                    return None
                connection.execute(
                    "UPDATE clips SET used_at = ?, uses = uses + 1 WHERE key = ?",
                    (time.time(), key),
                )
        except sqlite3.DatabaseError as exc:  # pragma: no cover - disk failure
            log.warning("tts_cache_read_failed error=%s", exc)
            self.misses += 1
            return None
        self.hits += 1
        return CachedClip(pcm=bytes(row[0]), sample_rate=int(row[1]), key=key)

    def has(self, key: str) -> bool:
        """Existence check that does **not** count as a hit or a miss.

        Warming the cache at boot asks "what is missing?" — that is not a user
        listening, and counting it would make the hit-rate lie.
        """
        try:
            with self._connect() as connection:
                return connection.execute("SELECT 1 FROM clips WHERE key = ?", (key,)).fetchone() is not None
        except sqlite3.DatabaseError:  # pragma: no cover - disk failure
            return False

    def put(self, key: str, pcm: bytes, *, sample_rate: int, engine: str = "", voice: str = "", text: str = "") -> None:
        now = time.time()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO clips (key, engine, voice, text, sample_rate, pcm, bytes, created_at, used_at, uses)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(key) DO UPDATE SET
                        pcm = excluded.pcm,
                        sample_rate = excluded.sample_rate,
                        bytes = excluded.bytes,
                        used_at = excluded.used_at
                    """,
                    (key, engine, voice, " ".join(text.split())[:200], sample_rate, pcm, len(pcm), now, now),
                )
        except sqlite3.DatabaseError as exc:  # pragma: no cover - disk failure
            log.warning("tts_cache_write_failed error=%s", exc)
            return
        # Recompute rather than add: a re-synthesis of the same key overwrites
        # the old clip, and adding blindly would inflate the total until the
        # LRU started evicting clips that were never there.
        self.bytes = self._total_bytes()
        if self.bytes > self.max_bytes:
            self.prune()

    def prune(self) -> int:
        """Drop least-recently-used clips until the cap holds.  Returns rows gone."""
        removed = 0
        try:
            with self._connect() as connection:
                while self.bytes > self.max_bytes:
                    rows = connection.execute(
                        "SELECT key, bytes FROM clips ORDER BY used_at ASC LIMIT 50"
                    ).fetchall()
                    if not rows:
                        break
                    for key, size in rows:
                        # Re-check per clip: a batch delete of fifty clips to
                        # reclaim one is how a cache becomes a mystery.
                        if self.bytes <= self.max_bytes:
                            break
                        connection.execute("DELETE FROM clips WHERE key = ?", (key,))
                        self.bytes -= int(size)
                        removed += 1
                connection.commit()
        except sqlite3.DatabaseError as exc:  # pragma: no cover - disk failure
            log.warning("tts_cache_prune_failed error=%s", exc)
            return removed
        if removed:
            log.info("tts_cache_pruned clips=%s remaining_mb=%.1f", removed, self.bytes / 1e6)
        return removed

    def clear(self) -> int:
        with self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM clips").fetchone()[0])
            connection.execute("DELETE FROM clips")
            connection.commit()
        self.bytes = 0
        return count

    def stats(self) -> CacheStats:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM clips"
                ).fetchone()
            entries, size = int(row[0]), int(row[1])
        except sqlite3.DatabaseError:  # pragma: no cover - disk failure
            entries, size = 0, self.bytes
        self.bytes = size
        return CacheStats(entries=entries, bytes=size, hits=self.hits, misses=self.misses)

    # ── warming ──────────────────────────────────────────────────────
    def missing(self, keys: Iterable[str]) -> list[str]:
        """Which of these keys are not cached yet — the warming work list.

        Keys, not texts: only the caller knows the voice, the engine and the
        prosody a line will actually be spoken with, and a second guess at that
        key here is how a "warm" cache somehow still misses (it did — the
        continuation question is a *question*, so it is cached at 0.95×).
        """
        return [key for key in keys if not self.has(key)]


__all__ = ["CacheStats", "CachedClip", "TtsCache", "cache_key"]
