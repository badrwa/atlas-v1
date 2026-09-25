"""Identity — who is speaking, and therefore what Atlas may do.

L4 is the level where Atlas stops being a microphone and starts being *someone's*
assistant.  It has exactly one security decision (`ContextGuard`), one place that
stores voice profiles (`ProfileRepository`), and one log that makes the decision
auditable (`SpeakerLog`).  Everything else in the project asks those three.

**What this is not.**  A voice profile is a *gate that reduces accidental access*,
not a cryptographic identity.  It stops the television and a guest from reading
your notes; it does not stop someone who records ten minutes of your voice.  Every
destructive action still needs a confirmation (L7), and the READMEs say this in
plain words rather than promising "voiceprint security" (plan pitfall #2).

**Capabilities, not permissions per word.**  Verification happens once per
utterance, and its result is a set of capabilities:

===========================  ==========  ============  ==================
capability                   owner       known other   stranger
===========================  ==========  ============  ==================
general chat / search        yes         yes           yes
read personal memory         yes         yes           no
write the vault              yes         own note only no
PC control                   yes         SAFE only     no
destructive / irreversible   yes+confirm no            no
enrol / wipe profiles        yes         no            no
===========================  ==========  ============  ==================

The table is data (`CAPABILITY_SETS`), the decision is one function, and the test
that checks every cell is exhaustive on purpose — that test is the safety belt.

**Embeddings are biometric data.**  They live in SQLite, which the `.gitignore`
excludes, and never in the vault, never in a log line, never in a provider
payload.  `SpeakerEvent` carries the *score*, not the vector; the vault gets a
human-readable note; the test suite asserts all three.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_core.contracts import Capability, LanguageTag, SpeakerMatch
from atlas_core.jsonl import JsonlLog

log = logging.getLogger(__name__)

#: How many embeddings one person keeps.  A rolling window on purpose: voices
#: drift with the room, the microphone and the season, and a profile welded to
#: three recordings from last winter is a profile that starts rejecting you.
DEFAULT_WINDOW = 6


OWNER_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.GENERAL,
        Capability.READ_MEMORY,
        Capability.WRITE_VAULT,
        Capability.OWN_NOTES,
        Capability.PC_CONTROL,
        Capability.DESTRUCTIVE,
        Capability.ENROL,
    }
)
#: An enrolled other person: their own note and the inbox, nothing of the owner's.
KNOWN_CAPABILITIES: frozenset[Capability] = frozenset(
    {Capability.GENERAL, Capability.OWN_NOTES}
)
#: An unrecognised voice: general conversation, and that is all.
RESTRICTED_CAPABILITIES: frozenset[Capability] = frozenset({Capability.GENERAL})


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two embeddings — the one definition in the project.

    Returns 0.0 for a zero vector or a length mismatch instead of raising: a
    half-loaded model must not crash a conversation, and "no similarity" is the
    safe answer when the numbers cannot be compared at all.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for left, right in zip(a, b, strict=True):
        dot += left * right
        norm_a += left * left
        norm_b += right * right
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def centroid(vectors: list[list[float]]) -> list[float]:
    """The average embedding, renormalised — a profile's centre of mass."""
    if not vectors:
        return []
    length = len(vectors[0])
    total = [0.0] * length
    for vector in vectors:
        if len(vector) != length:
            continue
        for index, value in enumerate(vector):
            total[index] += value
    count = float(len(vectors))
    average = [value / count for value in total]
    magnitude = math.sqrt(sum(value * value for value in average))
    return [value / magnitude for value in average] if magnitude else average


def best_match(embedding: list[float], profile: SpeakerProfile) -> float:
    """Highest similarity against every stored sample, not just the centroid.

    The centroid alone is smoother but forgets the day you had a cold; matching
    against the window's individual samples is what makes "tired voice" still
    count as you, while a stranger still has to beat all of them.
    """
    scores = [cosine(embedding, sample) for sample in profile.embeddings]
    if profile.centroid:
        scores.append(cosine(embedding, profile.centroid))
    return max(scores) if scores else 0.0


# ── permissions ──────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Permissions:
    """One utterance's allowance.  Immutable: nothing edits it after the guard."""

    capabilities: frozenset[Capability] = RESTRICTED_CAPABILITIES
    speaker: str = ""
    owner: bool = False
    known: bool = False
    accepted: bool = False
    score: float = 0.0
    subject: str = ""
    reason: str = ""

    # ── factories: the only ways to build one ────────────────────────
    @classmethod
    def owner_of(
        cls, name: str = "", *, score: float = 0.0, reason: str = "verified_owner",
        capabilities: frozenset[Capability] | None = None,
    ) -> Permissions:
        return cls(
            capabilities=capabilities or OWNER_CAPABILITIES,
            speaker=name,
            owner=True,
            known=True,
            accepted=True,
            score=score,
            reason=reason,
        )

    @classmethod
    def guest_of(cls, name: str, *, score: float = 0.0, reason: str = "verified_guest") -> Permissions:
        return cls(
            capabilities=KNOWN_CAPABILITIES,
            speaker=name,
            known=True,
            accepted=True,
            score=score,
            subject=name,
            reason=reason,
        )

    @classmethod
    def stranger(
        cls, *, score: float = 0.0, reason: str = "unknown_voice", speaker: str = ""
    ) -> Permissions:
        return cls(
            capabilities=RESTRICTED_CAPABILITIES,
            speaker=speaker,
            accepted=False,
            score=score,
            reason=reason,
        )

    # ── the rule ─────────────────────────────────────────────────────
    def allows(self, capability: Capability | str) -> bool:
        """May this utterance do that?

        One of the answers has a footnote, and it lives here so it stays one
        answer: writing to the vault is allowed for a *known other* person when
        they have `OWN_NOTES` **and** a subject — a guest may write their own
        `30_People/<name>.md`, never the owner's memory.  The skill that writes
        reads `Permissions.subject` to know which file it means.
        """
        capability = Capability(capability)
        if capability in self.capabilities:
            return True
        if capability is Capability.WRITE_VAULT:
            return Capability.OWN_NOTES in self.capabilities and bool(self.subject)
        return False

    @property
    def restricted(self) -> bool:
        """True when this is a stranger: general conversation only."""
        return not self.accepted

    def spoken_refusal(self, language: LanguageTag | str = "ar-MA") -> str:
        """What Atlas says when a private request arrives from a stranger.

        Polite, brief, and it does not hint at what exists — the plan's rule, in
        the two languages the user actually talks in.
        """
        if str(language).startswith("en"):
            return "Sorry, that one is personal — only for my owner."
        return "سمح ليا، هادشي خاص بصاحبي."

    def as_dict(self) -> dict[str, object]:
        return {
            "speaker": self.speaker,
            "owner": self.owner,
            "known": self.known,
            "accepted": self.accepted,
            "score": round(self.score, 3),
            "capabilities": sorted(str(capability) for capability in self.capabilities),
            "subject": self.subject,
            "reason": self.reason,
        }


@dataclass(slots=True)
class Audience:
    """What the *brain* is allowed to know about who is talking (L1 persona).

    Deliberately smaller than `Permissions`: the persona needs a name, whether it
    is the owner, and whether it must refuse private questions.  It never needs
    the capability list, so it never gets it — prompts are the last place to leak
    an implementation detail.
    """

    name: str = ""
    owner: bool = True
    known: bool = True
    restricted: bool = False
    subject: str = ""
    language: LanguageTag = "ar-MA"

    @classmethod
    def from_permissions(cls, permissions: Permissions, *, language: LanguageTag = "ar-MA") -> Audience:
        return cls(
            name=permissions.speaker or ("owner" if permissions.owner else ""),
            owner=permissions.owner,
            known=permissions.known,
            restricted=permissions.restricted,
            subject=permissions.subject,
            language=language,
        )

    @property
    def display(self) -> str:
        if self.owner:
            return self.name or "the owner"
        if self.name:
            return f"{self.name} (not the owner)"
        return "an unrecognised voice"

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "owner": self.owner,
            "known": self.known,
            "restricted": self.restricted,
            "subject": self.subject,
        }


# ── profiles ─────────────────────────────────────────────────────────
@dataclass(slots=True)
class SpeakerProfile:
    """One person's voice: their embeddings, and how good they were.

    The embeddings stay in SQLite.  This object is what travels inside the
    process — never into a prompt, a vault file or a log line.
    """

    name: str
    embeddings: list[list[float]] = field(default_factory=list)
    quality: float = 0.0
    samples: int = 0
    owner: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    note: str = ""

    @property
    def centroid(self) -> list[float]:
        return centroid(self.embeddings)

    @property
    def dimension(self) -> int:
        return len(self.embeddings[0]) if self.embeddings else 0

    def add(self, embedding: list[float], *, window: int = DEFAULT_WINDOW) -> None:
        """Append a sample and keep the window — the rolling-window drift rule."""
        if not embedding:
            return
        self.embeddings.append(list(embedding))
        if len(self.embeddings) > window:
            del self.embeddings[: len(self.embeddings) - window]
        self.samples += 1
        self.updated_at = time.time()

    def match(self, embedding: list[float], *, threshold: float = 0.65) -> SpeakerMatch:
        """Compare one utterance to this profile (best sample, not the mean)."""
        score = best_match(embedding, self)
        return SpeakerMatch(name=self.name, score=score, owner=self.owner and score >= threshold)

    def as_dict(self, *, include_vectors: bool = False) -> dict[str, object]:
        """The safe projection by default: no vectors leave unless asked for.

        `include_vectors=True` exists for the repository and for tests that assert
        the vault and the logs do *not* contain them.
        """
        payload: dict[str, object] = {
            "name": self.name,
            "quality": round(self.quality, 3),
            "samples": self.samples,
            "owner": self.owner,
            "window": len(self.embeddings),
            "dimension": self.dimension,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "note": self.note,
        }
        if include_vectors:
            payload["embeddings"] = [list(vector) for vector in self.embeddings]
            payload["centroid"] = self.centroid
        return payload


@dataclass(slots=True)
class SpeakerEvent:
    """One verification, logged.  The score is the point; the vector is absent."""

    speaker: str = ""
    score: float = 0.0
    threshold: float = 0.65
    accepted: bool = False
    owner: bool = False
    known: bool = False
    reason: str = ""
    utterance_ms: float = 0.0
    mood: str = ""
    ms: float = 0.0
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "accepted": self.accepted,
            "owner": self.owner,
            "known": self.known,
            "reason": self.reason,
            "utterance_ms": round(self.utterance_ms, 1),
            "mood": self.mood,
            "ms": round(self.ms, 2),
            "at": self.at,
        }


class SpeakerLog(JsonlLog[SpeakerEvent]):
    """`data/speaker_log.jsonl` — per-utterance scores, for honest tuning.

    The FAR/FRR sweep in `scripts/bench_speaker.py` starts here: a threshold
    picked without data is a guess wearing a number.
    """

    def __init__(self, path: str | Path = "data/speaker_log.jsonl", *, enabled: bool = True, load: bool = True) -> None:
        super().__init__(path, enabled=enabled, load=load)

    def _from_dict(self, data: dict[str, Any]) -> SpeakerEvent:
        return SpeakerEvent(
            speaker=str(data.get("speaker", "")),
            score=float(data.get("score", 0.0)),
            threshold=float(data.get("threshold", 0.65)),
            accepted=bool(data.get("accepted", False)),
            owner=bool(data.get("owner", False)),
            known=bool(data.get("known", False)),
            reason=str(data.get("reason", "")),
            utterance_ms=float(data.get("utterance_ms", 0.0)),
            mood=str(data.get("mood", "")),
            ms=float(data.get("ms", 0.0)),
            at=float(data.get("at", 0.0)),
        )

    # ── summaries ────────────────────────────────────────────────────
    @property
    def events(self) -> list[SpeakerEvent]:
        return self.records

    def summary(self) -> dict[str, object]:
        if not self.records:
            return {"events": 0, "accepted": 0, "accept_rate": 0.0, "speakers": {}}
        per_speaker: dict[str, list[float]] = {}
        for event in self.records:
            per_speaker.setdefault(event.speaker or "(unknown)", []).append(event.score)
        accepted = sum(1 for event in self.records if event.accepted)
        return {
            "events": len(self.records),
            "accepted": accepted,
            "accept_rate": round(accepted / len(self.records), 3),
            "speakers": {
                name: {
                    "n": len(scores),
                    "p50": round(sorted(scores)[len(scores) // 2], 3),
                    "max": round(max(scores), 3),
                }
                for name, scores in sorted(per_speaker.items())
            },
        }

    def scores(self, name: str = "") -> list[float]:
        """Every score for one speaker (or all of them), for the sweep."""
        return [event.score for event in self.records if not name or event.speaker == name]

    def tail(self, limit: int = 10) -> list[SpeakerEvent]:
        return self.records[-limit:]


class ProfileRepository:
    """Voice profiles in SQLite — local, gitignored, never in the vault.

    Two tables, because embeddings are a rolling window and a person is not:
    `speaker_profile` is who they are, `speaker_embedding` is what their voice
    sounded like on the days Atlas heard it.  Vectors are stored as float32
    blobs, which is both compact and exact enough for a cosine.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS speaker_profile (
        name       TEXT PRIMARY KEY,
        owner      INTEGER NOT NULL DEFAULT 0,
        quality    REAL NOT NULL DEFAULT 0,
        samples    INTEGER NOT NULL DEFAULT 0,
        note       TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS speaker_embedding (
        name       TEXT NOT NULL,
        idx        INTEGER NOT NULL,
        dim        INTEGER NOT NULL,
        vector     BLOB NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (name, idx)
    );
    CREATE INDEX IF NOT EXISTS speaker_embedding_name ON speaker_embedding (name);
    """

    def __init__(self, path: str | Path = "data/speakers.sqlite3", *, window: int = DEFAULT_WINDOW) -> None:
        self.path = Path(path)
        self.window = window
        # `sqlite3.connect(":memory:")` gives every *connection* its own empty
        # database, so the schema would vanish between two calls.  A shared-cache
        # URI plus one keeper connection makes it behave like a real store — this
        # is what lets the guard be tested without touching a disk.
        self._uri = (
            f"file:atlas-speakers-{id(self):x}?mode=memory&cache=shared"
            if str(path) == ":memory:"
            else None
        )
        self._keeper: sqlite3.Connection | None = (
            sqlite3.connect(self._uri, uri=True) if self._uri else None
        )
        if self._uri is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # A path check in one place: the vault must never be the home of biometrics.
        if "Obsidian" in self.path.as_posix() or "vault" in self.path.as_posix().lower():
            raise ValueError("voice profiles do not belong in the vault — use data/")
        self._init()

    # ── plumbing ─────────────────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        connection = (
            sqlite3.connect(self._uri, uri=True, timeout=5.0)
            if self._uri
            else sqlite3.connect(str(self.path), timeout=5.0)
        )
        if not self._uri:
            connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def close(self) -> None:
        """Drop the keeper connection (only meaningful for an in-memory store)."""
        if self._keeper is not None:
            self._keeper.close()
            self._keeper = None

    def _init(self) -> None:
        with self._connect() as connection:
            connection.executescript(self.SCHEMA)

    # ── reading ──────────────────────────────────────────────────────
    def names(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT name FROM speaker_profile ORDER BY name").fetchall()
        return [row[0] for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM speaker_profile").fetchone()[0])

    def owner_name(self) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT name FROM speaker_profile WHERE owner = 1 ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else ""

    def load(self, name: str) -> SpeakerProfile | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT owner, quality, samples, note, created_at, updated_at FROM speaker_profile WHERE name = ?",
                (name,),
            ).fetchone()
            if row is None:
                return None
            vectors = [
                _from_blob(blob, dim)
                for blob, dim in connection.execute(
                    "SELECT vector, dim FROM speaker_embedding WHERE name = ? ORDER BY idx", (name,)
                ).fetchall()
            ]
        return SpeakerProfile(
            name=name,
            embeddings=[vector for vector in vectors if vector],
            quality=float(row[1]),
            samples=int(row[2]),
            owner=bool(row[0]),
            note=str(row[3]),
            created_at=float(row[4]),
            updated_at=float(row[5]),
        )

    def load_all(self) -> dict[str, SpeakerProfile]:
        return {name: profile for name in self.names() if (profile := self.load(name)) is not None}

    # ── writing ──────────────────────────────────────────────────────
    def save(self, profile: SpeakerProfile, *, owner: bool | None = None) -> SpeakerProfile:
        """Store the profile and its rolling window of embeddings."""
        if owner is not None:
            profile.owner = owner
        if profile.owner:
            # Exactly one owner: enrolling a new one demotes the previous.
            with self._connect() as connection:
                connection.execute("UPDATE speaker_profile SET owner = 0 WHERE owner = 1")
        now = time.time()
        profile.updated_at = now
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO speaker_profile (name, owner, quality, samples, note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    owner = excluded.owner, quality = excluded.quality, samples = excluded.samples,
                    note = excluded.note, updated_at = excluded.updated_at
                """,
                (
                    profile.name,
                    int(profile.owner),
                    profile.quality,
                    profile.samples,
                    profile.note,
                    profile.created_at,
                    now,
                ),
            )
            connection.execute("DELETE FROM speaker_embedding WHERE name = ?", (profile.name,))
            window = profile.embeddings[-self.window :]
            for index, vector in enumerate(window):
                connection.execute(
                    "INSERT INTO speaker_embedding (name, idx, dim, vector, created_at) VALUES (?, ?, ?, ?, ?)",
                    (profile.name, index, len(vector), _to_blob(vector), now),
                )
        profile.embeddings = [list(vector) for vector in window]
        return profile

    def add_embedding(self, name: str, embedding: list[float]) -> SpeakerProfile | None:
        """Grow an existing profile by one sample (the drift fix, step 8)."""
        profile = self.load(name)
        if profile is None or not embedding:
            return None
        profile.add(embedding, window=self.window)
        return self.save(profile)

    def delete(self, name: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM speaker_profile WHERE name = ?", (name,))
            connection.execute("DELETE FROM speaker_embedding WHERE name = ?", (name,))
        return bool(cursor.rowcount)

    def wipe(self) -> int:
        """The voice-print eraser: every profile, every embedding, gone."""
        removed = self.count()
        with self._connect() as connection:
            connection.execute("DELETE FROM speaker_profile")
            connection.execute("DELETE FROM speaker_embedding")
        return removed

    def stats(self) -> dict[str, object]:
        profiles = self.load_all()
        return {
            "path": str(self.path),
            "people": len(profiles),
            "owner": self.owner_name(),
            "samples": sum(len(profile.embeddings) for profile in profiles.values()),
            "dimension": next((profile.dimension for profile in profiles.values() if profile.dimension), 0),
        }


def _to_blob(vector: list[float]) -> bytes:
    return array("f", vector).tobytes()


def _from_blob(blob: bytes, dim: int) -> list[float]:
    values = array("f")
    values.frombytes(blob)
    return [float(value) for value in values[:dim]]


# ── configuration and the guard ──────────────────────────────────────
@dataclass(slots=True)
class IdentityConfig:
    """`[identity]`, read without importing the config model (duck-typed).

    Defaults are the *safe* ones: identity on, restricted for strangers, and a
    threshold from the plan.  A user who wants the old single-user behaviour sets
    `enabled = false` and gets exactly L3 back.
    """

    enabled: bool = True
    owner_name: str = ""
    threshold: float = 0.65
    #: An utterance shorter than this cannot carry an owner-only decision: the
    #: television says one word too.  Short matches keep general capabilities.
    trust_min_ms: float = 1000.0
    #: Guests get SAFE PC control only when the owner opts in.
    guest_pc_control: bool = False
    #: The speaker-embedding ONNX model.  Kept here (and not in the audio package)
    #: so `[identity] model_path` has exactly one reader.
    model_path: str = "models/speaker/3dspeaker_speech_eres2net_base.onnx"
    profiles_path: str = "data/speakers.sqlite3"
    log_path: str = "data/speaker_log.jsonl"
    window: int = DEFAULT_WINDOW
    enrol_samples: int = 3
    enrol_seconds: float = 10.0
    enrol_min_speech_ms: float = 2500.0
    quality_floor: float = 0.55
    drift_rejections: int = 2
    greet_once_per_day: bool = True

    @classmethod
    def from_config(cls, config: Any = None) -> IdentityConfig:
        section = getattr(config, "identity", None)
        if section is None:
            return cls()
        return cls(
            enabled=bool(getattr(section, "enabled", True)),
            owner_name=str(getattr(section, "owner_name", "") or ""),
            threshold=float(getattr(section, "threshold", 0.65) or 0.65),
            trust_min_ms=float(getattr(section, "trust_min_ms", 1000.0) or 1000.0),
            guest_pc_control=bool(getattr(section, "guest_pc_control", False)),
            model_path=str(
                getattr(section, "model_path", "models/speaker/3dspeaker_speech_eres2net_base.onnx")
            ),
            profiles_path=str(getattr(section, "profiles_path", "data/speakers.sqlite3")),
            log_path=str(getattr(section, "log_path", "data/speaker_log.jsonl")),
            window=int(getattr(section, "window", DEFAULT_WINDOW) or DEFAULT_WINDOW),
            enrol_samples=int(getattr(section, "enrol_samples", 3) or 3),
            enrol_seconds=float(getattr(section, "enrol_seconds", 10.0) or 10.0),
            enrol_min_speech_ms=float(getattr(section, "enrol_min_speech_ms", 2500.0) or 2500.0),
            quality_floor=float(getattr(section, "quality_floor", 0.55) or 0.55),
            drift_rejections=int(getattr(section, "drift_rejections", 2) or 2),
            greet_once_per_day=bool(getattr(section, "greet_once_per_day", True)),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "owner": self.owner_name,
            "threshold": self.threshold,
            "trust_min_ms": self.trust_min_ms,
            "guest_pc_control": self.guest_pc_control,
        }


class ContextGuard:
    """The single place that decides what a speaker may do.

    `ToolLoop`, the memory service, the vault writer and the PC skills all ask
    *this* object — none of them re-derives the rule from a name (R5).  The
    signature is the contract: a match, a confidence, an utterance length, and
    the current mood.  Mood is recorded and *never* widens access: an assistant
    whose permissions depend on how it feels about you is not an assistant.
    """

    def __init__(
        self,
        config: IdentityConfig | None = None,
        *,
        repository: ProfileRepository | None = None,
        log: SpeakerLog | None = None,
        profiles: dict[str, SpeakerProfile] | None = None,
    ) -> None:
        self.config = config or IdentityConfig()
        self.repository = repository
        self.log = log
        self._profiles = profiles or {}
        #: Consecutive rejections per name in this session — the drift signal.
        self.rejections: dict[str, int] = {}

    # ── profiles ─────────────────────────────────────────────────────
    @property
    def profiles(self) -> dict[str, SpeakerProfile]:
        """Injected profiles win; otherwise the repository is the source."""
        if self._profiles:
            return self._profiles
        if self.repository is not None:
            return self.repository.load_all()
        return {}

    def enroll(self, profile: SpeakerProfile, *, owner: bool | None = None) -> SpeakerProfile:
        """Save a profile, optionally as the owner.

        `owner=None` means "the first person enrolled owns this Atlas" — a
        single-user assistant must not need a flag to exist.
        """
        others = [p for name, p in self.profiles.items() if name != profile.name]
        if owner is None:
            owner = profile.owner or not any(other.owner for other in others)
        profile.owner = bool(owner)
        self._profiles[profile.name] = profile
        if self.repository is not None:
            self.repository.save(profile, owner=profile.owner)
            if profile.owner:
                # Mirror the demotion into the in-memory view: exactly one owner.
                for name, existing in self._profiles.items():
                    if name != profile.name:
                        existing.owner = False
        return profile

    @property
    def owner_name(self) -> str:
        if self.config.owner_name:
            return self.config.owner_name
        for profile in self.profiles.values():
            if profile.owner:
                return profile.name
        return ""

    @property
    def known_names(self) -> list[str]:
        return sorted(self.profiles)

    # ── the decision ─────────────────────────────────────────────────
    def verify(
        self,
        embedding: list[float],
        *,
        utterance_ms: float = 0.0,
        mood: Any = None,
        elapsed_ms: float = 0.0,
    ) -> Permissions:
        """Score an utterance against every profile and return the allowance."""
        if not self.config.enabled:
            return Permissions.owner_of(
                self.owner_name or "owner", reason="identity_disabled"
            )

        best: SpeakerMatch | None = None
        for profile in self.profiles.values():
            candidate = profile.match(embedding, threshold=self.config.threshold)
            if best is None or candidate.score > best.score:
                best = candidate

        if best is None:
            return self._record(
                Permissions.stranger(reason="no_profiles"),
                threshold=self.config.threshold,
                utterance_ms=utterance_ms,
                mood=mood,
                elapsed_ms=elapsed_ms,
            )

        if best.score < self.config.threshold:
            self.rejections[best.name] = self.rejections.get(best.name, 0) + 1
            return self._record(
                Permissions.stranger(score=best.score, reason="below_threshold", speaker=best.name),
                threshold=self.config.threshold,
                utterance_ms=utterance_ms,
                mood=mood,
                elapsed_ms=elapsed_ms,
            )

        self.rejections.pop(best.name, None)
        short = utterance_ms and utterance_ms < self.config.trust_min_ms
        if best.owner:
            if short:
                # A match is a match; the *trust* is what a short utterance cannot
                # carry.  General capabilities, so an owner saying "safi" is never
                # refused — they just cannot delete a file with two syllables.
                return self._record(
                    Permissions.owner_of(
                        best.name,
                        score=best.score,
                        reason="utterance_too_short",
                        capabilities=frozenset({Capability.GENERAL}),
                    ),
                    threshold=self.config.threshold,
                    utterance_ms=utterance_ms,
                    mood=mood,
                    elapsed_ms=elapsed_ms,
                )
            return self._record(
                Permissions.owner_of(best.name, score=best.score),
                threshold=self.config.threshold,
                utterance_ms=utterance_ms,
                mood=mood,
                elapsed_ms=elapsed_ms,
            )

        capabilities = set(KNOWN_CAPABILITIES)
        if self.config.guest_pc_control and not short:
            # SAFE-only, and never confirmation-free: the DESTRUCTIVE capability
            # is not in this set, so the CONFIRM gate in SkillRegistry still bites.
            capabilities.add(Capability.PC_CONTROL)
        if not short:
            capabilities.add(Capability.READ_MEMORY)
        return self._record(
            Permissions(
                capabilities=frozenset(capabilities),
                speaker=best.name,
                known=True,
                accepted=True,
                score=best.score,
                subject=best.name,
                reason="verified_guest",
            ),
            threshold=self.config.threshold,
            utterance_ms=utterance_ms,
            mood=mood,
            elapsed_ms=elapsed_ms,
        )

    def effective_permissions(
        self,
        match: SpeakerMatch,
        *,
        confidence: float = 0.0,
        mood: Any = None,
        utterance_ms: float = 0.0,
    ) -> Permissions:
        """The spec's signature, for callers that already have a `SpeakerMatch`.

        `verify()` is the path the voice loop uses (it has an embedding); this is
        the same decision for anything holding a match — the vault writer, the
        tool loop, a test.  `confidence` is the ASR confidence: it is logged, and
        it never buys a capability either.
        """
        if not self.config.enabled:
            return self._record(
                Permissions.owner_of(self.owner_name or "owner", reason="identity_disabled"),
                threshold=self.config.threshold,
                utterance_ms=utterance_ms,
                mood=mood,
            )
        if not match.name or match.score < self.config.threshold:
            return self._record(
                Permissions.stranger(score=match.score, reason="below_threshold", speaker=match.name),
                threshold=self.config.threshold,
                utterance_ms=utterance_ms,
                mood=mood,
                confidence=confidence,
            )
        if match.owner:
            return self._record(
                Permissions.owner_of(match.name, score=match.score),
                threshold=self.config.threshold,
                utterance_ms=utterance_ms,
                mood=mood,
                confidence=confidence,
            )
        return self._record(
            Permissions.guest_of(match.name, score=match.score),
            threshold=self.config.threshold,
            utterance_ms=utterance_ms,
            mood=mood,
            confidence=confidence,
        )

    # ── drift ────────────────────────────────────────────────────────
    def should_reoffer_enrolment(self, name: str) -> bool:
        """Twice rejected in one session is a changed voice, not an impostor."""
        return self.rejections.get(name, 0) >= self.config.drift_rejections

    def reoffer_line(self, language: LanguageTag | str = "ar-MA") -> str:
        if str(language).startswith("en"):
            return "Your voice sounds a bit different — shall I re-enrol it? Five seconds is enough."
        return "Sowtek tbdel chwiya, 3awd nsejlo? Khams tawani kafyin."

    # ── greetings ────────────────────────────────────────────────────
    def greeting(self, name: str, *, language: LanguageTag | str = "ar-MA", extra: str = "") -> str:
        """A first-wake-of-the-day greeting, with one optional personal line."""
        if not name:
            return ""
        line = f"Morning, {name}." if str(language).startswith("en") else f"Sbah lkhir {name}."
        return f"{line} {extra}".strip()

    # ── what happened ────────────────────────────────────────────────
    @property
    def events(self) -> list[SpeakerEvent]:
        """Every verdict this guard has made (empty when no log is attached)."""
        return list(self.log.records) if self.log is not None else []

    @property
    def last_event(self) -> SpeakerEvent | None:
        """The verdict just made — what `atlas identity log` and the tests read."""
        return self.events[-1] if self.events else None

    # ── the gate cannot run ──────────────────────────────────────────
    def unavailable(self, *, reason: str, utterance_ms: float = 0.0, mood: Any = None) -> Permissions:
        """The verifier could not run: fall back to single-user, and *log it*.

        Two very different situations share this path — a machine with no model
        installed (identity is effectively off, and the user is the only person
        in the room) and an utterance too short to identify.  The reason string
        is what tells them apart in `data/speaker_log.jsonl`, and the falling
        back to the owner is deliberate: a missing verifier must not lock the
        owner out of their own notes.
        """
        return self._record(
            Permissions.owner_of(self.owner_name or "owner", reason=reason),
            threshold=self.config.threshold,
            utterance_ms=utterance_ms,
            mood=mood,
        )

    # ── logging ──────────────────────────────────────────────────────
    def _record(
        self,
        permissions: Permissions,
        *,
        threshold: float,
        utterance_ms: float = 0.0,
        mood: Any = None,
        elapsed_ms: float = 0.0,
        confidence: float = 0.0,
    ) -> Permissions:
        if self.log is not None:
            self.log.add(
                SpeakerEvent(
                    speaker=permissions.speaker,
                    score=permissions.score,
                    threshold=threshold,
                    accepted=permissions.accepted,
                    owner=permissions.owner,
                    known=permissions.known,
                    reason=permissions.reason,
                    utterance_ms=utterance_ms,
                    mood=str(getattr(mood, "label", "") or ""),
                    ms=elapsed_ms,
                )
            )
        log.debug(
            "speaker_decision speaker=%r score=%.3f accepted=%s reason=%s confidence=%.2f",
            permissions.speaker,
            permissions.score,
            permissions.accepted,
            permissions.reason,
            confidence,
        )
        return permissions


class DailyGreeter:
    """Greet by name once a day, and then shut up about it.

    A separate tiny object rather than a flag on the guard: it is UI state, it
    must survive a restart, and it must not be able to influence access.  Stored
    as one line of text (`data/last_greeting.txt`), because a database for a date
    is how a project gets heavy.
    """

    def __init__(self, path: str | Path = "data/last_greeting.txt") -> None:
        self.path = Path(path)

    def due(self, name: str, *, now: float | None = None) -> bool:
        if not name:
            return False
        stamp = time.strftime("%Y-%m-%d", time.localtime(now if now is not None else time.time()))
        return self._read() != f"{name}|{stamp}"

    def mark(self, name: str, *, now: float | None = None) -> None:
        stamp = time.strftime("%Y-%m-%d", time.localtime(now if now is not None else time.time()))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(f"{name}|{stamp}", encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk trouble must not block a greeting
            log.warning("greeting_state_unwritable error=%s", exc)

    def _read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""


def fingerprint(profile: SpeakerProfile) -> str:
    """A short, non-reversible id for a profile — for logs and audit lines.

    Never the vector: a hash of the centroid is enough to tell two enrolments
    apart in a log, and it cannot be turned back into a voice.
    """
    payload = json.dumps(
        {"name": profile.name, "centroid": [round(value, 4) for value in profile.centroid]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


__all__ = [
    "DEFAULT_WINDOW",
    "KNOWN_CAPABILITIES",
    "OWNER_CAPABILITIES",
    "RESTRICTED_CAPABILITIES",
    "Audience",
    "Capability",
    "ContextGuard",
    "DailyGreeter",
    "IdentityConfig",
    "Permissions",
    "ProfileRepository",
    "SpeakerEvent",
    "SpeakerLog",
    "SpeakerProfile",
    "best_match",
    "centroid",
    "cosine",
    "fingerprint",
]
