"""Git journal — Atlas's memory is reversible.

Every batch of vault writes becomes one commit.  So "Atlas, undo that" is a
`git revert`, not an apology, and every change to your second brain is
auditable with `git log`.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from atlas_core.errors import VaultError

log = logging.getLogger(__name__)

REVERT_BODY = re.compile(r"This reverts commit ([0-9a-f]{7,40})")


@dataclass
class JournalEntry:
    sha: str = ""
    message: str = ""
    paths: tuple[str, ...] = ()


@dataclass
class GitJournal:
    """Thin, explicit wrapper around the git CLI (no GitPython dependency)."""

    root: Path
    enabled: bool = True
    author_name: str = "Atlas"
    author_email: str = "atlas@localhost"
    history: list[JournalEntry] = field(default_factory=list)

    # ── availability ─────────────────────────────────────────────────
    @property
    def is_repo(self) -> bool:
        return (self.root / ".git").exists()

    @property
    def _git_available(self) -> bool:
        return shutil.which("git") is not None

    def init(self) -> bool:
        """Create the repo if needed. Returns True when something changed."""
        if self.is_repo or not self._git_available:
            return False
        self._run("init", "-q")
        self._run("config", "user.name", self.author_name)
        self._run("config", "user.email", self.author_email)
        return True

    # ── writing ──────────────────────────────────────────────────────
    def commit(self, message: str, paths: list[str] | None = None) -> JournalEntry:
        """Stage and commit. Returns an empty entry when there was nothing to do."""
        if not self.enabled or not self.is_repo or not self._git_available:
            return JournalEntry(message=message)

        self._run("add", "-A", *(paths or []))
        status = self._run("status", "--porcelain")
        if not status.strip():
            return JournalEntry(message=message)

        self._run("commit", "-q", "-m", message)
        sha = self._run("rev-parse", "HEAD").strip()
        entry = JournalEntry(sha=sha, message=message, paths=tuple(paths or ()))
        self.history.append(entry)
        log.info("vault_commit sha=%s message=%s", sha[:8], message)
        return entry

    def undo_last(self, *, hard: bool = False) -> JournalEntry | None:
        """Revert the most recent *Atlas* write.

        Resolved from git history rather than process memory, so "Atlas, undo
        that" works after a restart — and it skips writes that were already
        reverted, so saying "undo" twice unwinds two writes instead of reverting
        the same one forever.
        """
        target = self._undo_target()
        if target is None:
            return None
        sha, subject = target
        entry = JournalEntry(sha=sha, message=subject)
        self.history = [item for item in self.history if item.sha != sha]

        if not self.is_repo or not self._git_available:
            return entry
        try:
            if hard:
                self._run("checkout", f"{sha}~1", "--", ".")
            else:
                self._run("revert", "--no-edit", sha)
        except VaultError:
            log.warning("vault_undo_failed sha=%s", sha[:8])
            return None
        return entry

    def _undo_target(self) -> tuple[str, str] | None:
        """Newest commit this journal made that has not already been reverted."""
        if not self.is_repo or not self._git_available:
            if self.history:
                last = self.history[-1]
                return (last.sha, last.message)
            return None

        # `git revert` records the original commit sha in the body ("This reverts
        # commit <sha>.") — that is the only reliable marker, because Atlas's write
        # messages are deliberately similar (and sometimes identical).
        try:
            raw = self._run("log", "--pretty=%H%x1f%s%x1f%b%x1e")
        except VaultError:
            return None

        reverted: set[str] = set()
        written: list[tuple[str, str]] = []
        for record in raw.split("\x1e"):
            if not record.strip():
                continue
            sha, _, rest = record.partition("\x1f")
            subject, _, body = rest.partition("\x1f")
            sha, subject = sha.strip(), subject.strip()
            if match := REVERT_BODY.search(body):
                reverted.add(match.group(1))
            elif subject.startswith("atlas:"):
                written.append((sha, subject))

        for sha, subject in written:  # newest first
            if sha not in reverted:
                return (sha, subject)
        return None

    def log(self, limit: int = 20) -> list[str]:
        if not self.is_repo or not self._git_available:
            return []
        try:
            raw = self._run("log", f"-{limit}", "--pretty=%h %s")
        except VaultError:
            return []
        return [line for line in raw.splitlines() if line.strip()]

    def is_dirty(self) -> bool:
        if not self.is_repo or not self._git_available:
            return False
        return bool(self._run("status", "--porcelain").strip())

    # ── plumbing ─────────────────────────────────────────────────────
    def _run(self, *args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise VaultError(f"git {args[0]} failed: {exc}") from exc
        if result.returncode != 0 and args[0] not in {"status", "rev-parse"}:
            raise VaultError(f"git {args[0]}: {result.stderr.strip()[:200]}")
        return result.stdout


__all__ = ["GitJournal", "JournalEntry"]
