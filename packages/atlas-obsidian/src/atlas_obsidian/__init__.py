"""Atlas second brain — one writer, git-committed, never deletes."""

from atlas_obsidian.journal import GitJournal, JournalEntry
from atlas_obsidian.vault import NoteRef, VaultAdapter

__all__ = ["GitJournal", "JournalEntry", "NoteRef", "VaultAdapter"]
