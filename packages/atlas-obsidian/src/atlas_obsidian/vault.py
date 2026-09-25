"""VaultAdapter — the single writer for Atlas's second brain.

Everything Atlas remembers lives here as Markdown.  The adapter enforces the
folder contract, appends rather than rewrites, archives rather than deletes, and
hands every change to the git journal so nothing is ever lost.

Obsidian stays a *viewer*: Atlas never needs the app to be running.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from atlas_core.config import ObsidianSection
from atlas_core.errors import VaultError
from atlas_obsidian.journal import GitJournal

log = logging.getLogger(__name__)

FRONTMATTER = "---\n{body}---\n"


@dataclass
class NoteRef:
    """A vault file Atlas can point at (only relative paths are ever exposed)."""

    path: str
    title: str
    kind: str = "note"

    @property
    def wiki_link(self) -> str:
        return f"[[{self.title}]]"


class VaultAdapter:
    """Read/write access to one Obsidian vault, under the folder contract."""

    def __init__(self, config: ObsidianSection, *, journal: GitJournal | None = None) -> None:
        self.config = config
        self.root = Path(config.vault_path) if config.vault_path else Path()
        self.journal = journal or GitJournal(Path(self.root), enabled=config.git_commit_writes)

    # ── lifecycle ────────────────────────────────────────────────────
    @property
    def exists(self) -> bool:
        return bool(self.config.vault_path) and self.root.is_dir()

    def require(self) -> Path:
        if not self.exists:
            raise VaultError(
                f"no vault configured (got {self.config.vault_path!r}) — run: python -m atlas vault init <path>"
            )
        return self.root

    @classmethod
    def init_from_template(
        cls,
        target: str | Path,
        template: str | Path,
        *,
        git: bool = True,
    ) -> VaultAdapter:
        """Create a new vault from `vault-template/`, then commit the skeleton."""
        target_path = Path(target)
        template_path = Path(template)
        if not template_path.is_dir():
            raise VaultError(f"template not found: {template_path}")

        target_path.mkdir(parents=True, exist_ok=True)
        for item in template_path.iterdir():
            destination = target_path / item.name
            if destination.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)

        section = ObsidianSection(vault_path=str(target_path), git_commit_writes=git)
        journal = GitJournal(Path(target_path), enabled=git)
        if git:
            journal.init()
            journal.commit("vault: initial Atlas structure")
        return cls(section, journal=journal)

    # ── paths ────────────────────────────────────────────────────────
    def folder(self, name: str) -> Path:
        """Resolve a contract folder ('inbox', 'daily', …) to an absolute path."""
        relative = self.config.folder(name)
        return self.require() / relative

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.require())).replace("\\", "/")
        except ValueError as exc:
            raise VaultError(f"{path} is outside the vault — refusing to touch it") from exc

    def _assert_inside_contract(self, path: Path) -> None:
        """Guard against writes outside the folders Atlas is allowed to write."""
        relative = self._relative(path)
        if relative.startswith((".git/", ".obsidian/")):
            raise VaultError(f"refusing to write to {relative}")

    # ── reading ──────────────────────────────────────────────────────
    def read(self, relative_path: str) -> str:
        path = self.require() / relative_path
        if not path.is_file():
            raise VaultError(f"note not found: {relative_path}")
        return path.read_text(encoding="utf-8")

    def read_optional(self, relative_path: str) -> str:
        try:
            return self.read(relative_path)
        except VaultError:
            return ""

    def list_notes(self, folder: str | None = None) -> list[NoteRef]:
        base = self.folder(folder) if folder else self.require()
        if not base.exists():
            return []
        refs: list[NoteRef] = []
        for path in sorted(base.rglob("*.md")):
            # Folder docs and Obsidian templates are not notes.
            if path.name.startswith("_") or path.name == "README.md":
                continue
            refs.append(NoteRef(path=self._relative(path), title=path.stem))
        return refs

    def recent_notes(self, limit: int = 5, folder: str | None = None) -> list[NoteRef]:
        notes = self.list_notes(folder)
        notes.sort(key=lambda ref: (self.require() / ref.path).stat().st_mtime, reverse=True)
        return notes[:limit]

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        """Naive substring search for now; FTS5 lands in L6 (`index.py`)."""
        needle = query.lower().strip()
        if not needle:
            return []
        hits: list[tuple[str, str]] = []
        for path in self.require().rglob("*.md"):
            if any(part in {".git", ".obsidian"} for part in path.parts):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if needle in text.lower():
                snippet = self._snippet(text, needle)
                hits.append((self._relative(path), snippet))
                if len(hits) >= limit:
                    break
        return hits

    # ── writing ──────────────────────────────────────────────────────
    def append_daily(self, text: str, *, when: datetime | None = None, speaker: str = "") -> Path:
        """Append one line to today's daily note (never rewrites history)."""
        moment = when or datetime.now()
        path = self.folder("daily") / f"{moment:%Y-%m-%d}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(self._daily_skeleton(moment), encoding="utf-8")

        stamp = moment.strftime("%H:%M")
        who = f" (speaker: {speaker})" if speaker else ""
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {stamp}{who} — {text.strip()}\n")
        self._commit(f"atlas: daily log {moment:%Y-%m-%d} +1 line")
        return path

    def create_note(
        self,
        title: str,
        body: str,
        *,
        folder: str = "notes",
        tags: list[str] | None = None,
        links: list[str] | None = None,
    ) -> Path:
        """Create a permanent note with frontmatter and enforced link hygiene."""
        slug = _slugify(title)
        path = self.folder(folder) / f"{slug}.md"
        if path.exists():
            path = self.folder(folder) / f"{slug}-{datetime.now():%H%M%S}.md"

        link_line = ""
        if links:
            link_line = "\n" + " ".join(f"[[{link}]]" for link in links) + "\n"
        elif folder == "notes":
            log.debug("note_without_links slug=%s", slug)

        frontmatter = FRONTMATTER.format(
            body=(
                f"type: note\n"
                f"title: {title}\n"
                f"created: {datetime.now():%Y-%m-%d %H:%M}\n"
                f"tags: [{', '.join(tags or [])}]\n"
                f"atlas: managed\n"
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{frontmatter}\n# {title}\n\n{body.strip()}\n{link_line}", encoding="utf-8")
        self._commit(f"atlas: note '{title}'")
        return path

    def remember(self, fact: str, *, directory: str = "atlas") -> Path:
        """Append a durable fact to `50_Atlas/memory.md` (person-attributed later)."""
        path = self.folder("atlas") if directory == "atlas" else self.folder(directory)
        memory = path / "memory.md"
        path.mkdir(parents=True, exist_ok=True)
        if not memory.exists():
            memory.write_text(
                FRONTMATTER.format(body="type: atlas-memory\ntags: [atlas, memory]\n")
                + "\n# Durable memory\n",
                encoding="utf-8",
            )
        line = f"- ({datetime.now():%Y-%m-%d}) {fact.strip()}\n"
        with memory.open("a", encoding="utf-8") as handle:
            handle.write(line)
        self._commit(f"atlas: remember '{fact[:40]}'")
        return memory

    def read_memory(self) -> str:
        return self.read_optional(f"{self.config.folder_atlas}/memory.md")

    def archive(self, relative_path: str, *, reason: str = "") -> Path:
        """Move a note to 90_Archive with a date prefix. Never deletes."""
        source = self.require() / relative_path
        if not source.is_file():
            raise VaultError(f"nothing to archive at {relative_path}")

        archive_dir = self.folder("archive")
        archive_dir.mkdir(parents=True, exist_ok=True)
        target = archive_dir / f"{datetime.now():%Y-%m-%d}_{source.name}"
        shutil.move(str(source), str(target))
        if reason:
            header = f"\n\n<!-- archived {datetime.now():%Y-%m-%d}: {reason} -->\n"
            with target.open("a", encoding="utf-8") as handle:
                handle.write(header)
        self._commit(f"atlas: archive {relative_path}")
        return target

    def undo_last_write(self) -> bool:
        return self.journal.undo_last() is not None

    # ── helpers ──────────────────────────────────────────────────────
    def _commit(self, message: str) -> None:
        self.journal.commit(message)

    def _daily_skeleton(self, moment: datetime) -> str:
        return (
            FRONTMATTER.format(
                body=f"type: daily\ndate: {moment:%Y-%m-%d}\ntags: [daily]\natlas: managed\n"
            )
            + f"\n# {moment:%A %d %B %Y}\n\n## Log\n\n## Tasks\n\n## Atlas summary\n"
        )

    @staticmethod
    def _snippet(text: str, needle: str, width: int = 120) -> str:
        index = text.lower().find(needle)
        start = max(0, index - width // 3)
        return text[start : start + width].replace("\n", " ").strip()

    def stats(self) -> dict[str, int | str]:
        if not self.exists:
            return {"vault": str(self.root), "notes": 0, "daily": 0, "configured": "no"}
        return {
            "vault": str(self.root),
            "configured": "yes",
            "notes": len(self.list_notes("notes")),
            "daily": len(self.list_notes("daily")),
            "inbox": len(self.list_notes("inbox")),
            "people": len(self.list_notes("people")),
            "journal_commits": len(self.journal.log(limit=999)),
            "dirty": "yes" if self.journal.is_dirty() else "no",
        }


def _slugify(title: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^\w\s\u0600-\u06ff-]", "", title, flags=re.UNICODE).strip()
    slug = re.sub(r"\s+", "-", slug)
    return (slug[:max_length] or f"note-{datetime.now():%Y%m%d-%H%M%S}").lower()


__all__ = ["NoteRef", "VaultAdapter"]
