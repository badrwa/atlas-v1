"""Vault contract: append-only logs, archived not deleted, git-committed, in-scope."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from atlas_core.config import ObsidianSection
from atlas_core.errors import VaultError
from atlas_obsidian.journal import GitJournal
from atlas_obsidian.vault import VaultAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = REPO_ROOT / "vault-template"


@pytest.fixture
def vault(tmp_path: Path) -> VaultAdapter:
    return VaultAdapter.init_from_template(tmp_path / "AtlasVault", TEMPLATE, git=True)


def test_init_creates_the_contract_folders(vault: VaultAdapter) -> None:
    for folder in ("inbox", "daily", "notes", "people", "projects", "atlas", "archive"):
        assert vault.folder(folder).is_dir(), folder
    assert (vault.folder("atlas") / "preferences.md").is_file()
    assert (vault.folder("atlas") / "jokes.md").is_file()


def test_init_is_idempotent(tmp_path: Path) -> None:
    first = VaultAdapter.init_from_template(tmp_path / "v", TEMPLATE, git=True)
    first.append_daily("salam")
    second = VaultAdapter.init_from_template(tmp_path / "v", TEMPLATE, git=True)
    notes = second.list_notes("daily")
    assert len(notes) == 1
    assert "salam" in second.read(notes[0].path)


def test_daily_note_is_append_only(vault: VaultAdapter) -> None:
    first = vault.append_daily("kafé m3a l'équipe", speaker="badr")
    second = vault.append_daily("zidt note f l'idée dyal l'app")
    assert first == second, "both lines belong to today's note"
    text = first.read_text(encoding="utf-8")
    assert "kafé m3a l'équipe" in text and "zidt note" in text
    assert text.index("kafé") < text.index("zidt"), "new lines land at the bottom"
    assert "speaker: badr" in text


def test_daily_note_has_frontmatter_once(vault: VaultAdapter) -> None:
    vault.append_daily("one")
    vault.append_daily("two")
    note = vault.list_notes("daily")[0]
    text = (vault.root / note.path).read_text(encoding="utf-8")
    assert text.count("type: daily") == 1
    assert text.startswith("---\n")


def test_create_note_writes_frontmatter_and_links(vault: VaultAdapter) -> None:
    path = vault.create_note(
        "Idée dyal l'app l'jadida",
        "Serveur local + darija voice.",
        tags=["idea", "project"],
        links=["Atlas"],
    )
    text = path.read_text(encoding="utf-8")
    assert "type: note" in text and "atlas: managed" in text
    assert "[[Atlas]]" in text
    assert path.parent == vault.folder("notes")


def test_duplicate_titles_do_not_overwrite(vault: VaultAdapter) -> None:
    first = vault.create_note("Same title", "first")
    second = vault.create_note("Same title", "second")
    assert first != second
    assert "first" in first.read_text(encoding="utf-8")
    assert "second" in second.read_text(encoding="utf-8")


def test_remember_appends_to_memory_file(vault: VaultAdapter) -> None:
    vault.remember("l'idée dyal l'app: serveur local")
    vault.remember("tea: atay b nana")
    memory = vault.read_memory()
    assert "serveur local" in memory and "atay b nana" in memory
    assert memory.count("# Durable memory") == 1


def test_archive_moves_and_never_deletes(vault: VaultAdapter) -> None:
    note = vault.create_note("Temporary thought", "not needed")
    relative = str(note.relative_to(vault.root)).replace("\\", "/")

    archived = vault.archive(relative, reason="user asked")
    assert not note.exists(), "the original location is cleared"
    assert archived.exists() and archived.parent == vault.folder("archive")
    assert "user asked" in archived.read_text(encoding="utf-8")


def test_writes_are_committed_to_git(vault: VaultAdapter) -> None:
    vault.append_daily("committed line")
    log = vault.journal.log()
    assert any("initial Atlas structure" in line for line in log)
    assert any("daily log" in line for line in log)


def test_undo_reverts_the_last_write_and_keeps_earlier_history(vault: VaultAdapter) -> None:
    vault.append_daily("first line stays")
    vault.append_daily("second line gets undone")
    daily = vault.list_notes("daily")[0]
    assert "second line gets undone" in vault.read(daily.path)

    assert vault.undo_last_write() is True
    text = vault.read(daily.path)
    assert "second line gets undone" not in text
    assert "first line stays" in text, "undo must not touch earlier history"


def test_undoing_the_creation_of_a_note_removes_it(tmp_path: Path, vault: VaultAdapter) -> None:
    """Undo means 'as if it never happened' — including the file it created."""
    vault.append_daily("only line ever")
    daily = vault.list_notes("daily")[0]
    assert vault.undo_last_write() is True
    assert not (vault.root / daily.path).exists()


def test_undo_works_from_a_fresh_process(vault: VaultAdapter) -> None:
    """The CLI is a new process every time: undo must come from git, not memory."""
    vault.append_daily("first write")
    vault.append_daily("second write")
    daily = vault.list_notes("daily")[0]

    from atlas_core.config import ObsidianSection

    fresh = VaultAdapter(ObsidianSection(vault_path=str(vault.root)))
    assert fresh.journal.history == [], "a fresh process starts with empty in-memory history"

    assert fresh.undo_last_write() is True
    text = fresh.read(daily.path)
    assert "second write" not in text and "first write" in text

    assert fresh.undo_last_write() is True, "a second undo unwinds the next write, not the same one"
    assert fresh.undo_last_write() is False, "and then there is nothing left to undo"


def test_undo_twice_unwinds_two_writes(vault: VaultAdapter) -> None:
    vault.append_daily("first")
    vault.append_daily("second")
    daily = vault.list_notes("daily")[0]

    assert vault.undo_last_write() is True
    assert "second" not in vault.read(daily.path)
    assert "first" in vault.read(daily.path)


def test_writes_outside_the_vault_are_refused(vault: VaultAdapter) -> None:
    with pytest.raises(VaultError):
        vault.archive("../../etc/passwd")


def test_git_internals_are_protected(vault: VaultAdapter) -> None:
    with pytest.raises(VaultError):
        vault._assert_inside_contract(vault.root / ".git" / "config")


def test_search_finds_text_and_returns_snippet(vault: VaultAdapter) -> None:
    vault.create_note("Chicken tajine recipe", "sbarma w zitoun b bezzaf")
    hits = vault.search("zitoun")
    assert hits and "zitoun" in hits[0][1]


def test_list_notes_skips_templates_and_readmes(vault: VaultAdapter) -> None:
    titles = {note.title for note in vault.list_notes("daily")}
    assert "_TEMPLATE" not in titles
    assert all(not note.title.startswith("_") for note in vault.list_notes())


def test_search_ignores_git_directory(vault: VaultAdapter) -> None:
    vault.append_daily("unique-token-xyz")
    paths = [path for path, _ in vault.search("unique-token-xyz")]
    assert paths and all(".git" not in path for path in paths)


def test_unconfigured_vault_raises_a_helpful_error() -> None:
    adapter = VaultAdapter(ObsidianSection())
    with pytest.raises(VaultError) as excinfo:
        adapter.require()
    assert "vault init" in str(excinfo.value)


def test_stats_summarise_the_vault(vault: VaultAdapter) -> None:
    vault.append_daily("hi")
    vault.create_note("A note", "body")
    stats = vault.stats()
    assert stats["configured"] == "yes"
    assert stats["daily"] == 1 and stats["notes"] == 1


def test_journal_disabled_writes_without_git(tmp_path: Path) -> None:
    adapter = VaultAdapter.init_from_template(tmp_path / "plain", TEMPLATE, git=False)
    adapter.append_daily("no git here")
    assert adapter.journal.log() == []
    assert len(adapter.list_notes("daily")) == 1


def test_journal_handles_missing_git_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    journal = GitJournal(tmp_path, enabled=True)
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert journal.init() is False
    assert journal.commit("nothing").sha == ""


def test_git_log_is_readable(vault: VaultAdapter) -> None:
    vault.append_daily("x")
    result = subprocess.run(
        ["git", "log", "--oneline"], cwd=vault.root, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "atlas" in result.stdout.lower()


# ── L4: person notes ─────────────────────────────────────────────────
def test_a_person_note_is_created_once_and_readable(vault: VaultAdapter) -> None:
    path = vault.ensure_person("Said El Amrani", relationship="friend")
    assert path.exists()
    assert path.parent.name == "30_People"
    assert path.name == "said-el-amrani.md"  # a filesystem-safe slug

    text = vault.read_person("Said El Amrani")
    assert "name: Said El Amrani" in text
    assert "relationship: friend" in text
    assert "## Facts Atlas should remember" in text

    assert vault.ensure_person("Said El Amrani") == path, "second call is a no-op"
    assert len(vault.people()) == 1


def test_facts_are_appended_with_a_date_and_never_overwrite(vault: VaultAdapter) -> None:
    vault.append_person_fact("said", "kayt9an f atay b nan3na3")
    vault.append_person_fact("said", "kaykhdem f Casablanca")
    text = vault.read_person("said")
    assert "kayt9an f atay b nan3na3" in text
    assert "kaykhdem f Casablanca" in text, "the earlier fact is still there"
    assert text.count("- (") == 2, "one dated bullet per fact"


def test_the_owner_and_a_guest_have_separate_notes(vault: VaultAdapter) -> None:
    vault.remember("project Atlas kayt9an f darija")  # the owner's memory file
    vault.append_person_fact("said", "sahbi dyal badr")
    assert "sahbi" not in vault.read_memory()
    assert "sahbi" in vault.read_person("said")
    assert "darija" not in vault.read_person("said")


def test_a_person_note_never_holds_a_voice_print(vault: VaultAdapter) -> None:
    """The vault is git-journaled and synced; embeddings stay in SQLite."""
    vault.append_person_fact("said", "sowti tsejjel")  # a fact, not a vector
    text = vault.read_person("said")
    assert "embedding" not in text.lower()
    assert "[[0." not in text and "0.5, " not in text


def test_a_person_note_needs_a_name(vault: VaultAdapter) -> None:
    from atlas_obsidian.vault import VaultError

    with pytest.raises(VaultError):
        vault.ensure_person("   ")


def test_person_writes_are_in_the_git_journal(vault: VaultAdapter) -> None:
    vault.append_person_fact("said", "kaysken f Rabat")
    lines = vault.journal.log(limit=5)
    assert any("said" in line for line in lines), lines
