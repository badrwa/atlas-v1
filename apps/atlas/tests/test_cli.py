"""CLI tests — the commands a human actually types, run for real."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from atlas.cli import main

REPO_ROOT = Path(__file__).resolve().parents[3]

SANDBOX_CONFIG = """
[app]
profile = "lean"
language = "ar-MA"

[mind]
provider_order = ["deadlocal"]

[[providers]]
name = "deadlocal"
kind = "openai_compatible"
model = "nothing"
base_url = "http://127.0.0.1:9/v1"
timeout_s = 1.0
"""


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway workspace: its own config.toml, no .env, no real providers."""
    config = tmp_path / "config.toml"
    config.write_text(SANDBOX_CONFIG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_CONFIG", str(config))
    return tmp_path


# ── version / protocol ───────────────────────────────────────────────
def test_version_prints_components(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert "atlas 0.1.0" in out and "python" in out


def test_ui_protocol_emits_generated_typescript(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["ui-protocol"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("// AUTO-GENERATED")
    assert "export interface StateMessage" in out
    assert "export type AtlasMessage" in out


# ── doctor ───────────────────────────────────────────────────────────
def test_doctor_runs_and_reports(capsys: pytest.CaptureFixture[str], sandbox: Path) -> None:
    exit_code = main(["doctor"])
    out = capsys.readouterr().out
    assert "ATLAS doctor" in out
    assert "Python" in out and "config.toml" in out
    assert exit_code in (0, 1), "doctor reports, it does not crash"


def test_doctor_flags_a_missing_provider_key(capsys: pytest.CaptureFixture[str], sandbox: Path) -> None:
    main(["doctor"])
    out = capsys.readouterr().out
    # deadlocal needs no key, so the CLI must still find it usable
    assert "deadlocal" in out


# ── providers ────────────────────────────────────────────────────────
def test_providers_lists_configuration(capsys: pytest.CaptureFixture[str], sandbox: Path) -> None:
    assert main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "deadlocal" in out
    assert "local" in out, "a 127.0.0.1 provider needs no key"


def test_providers_reports_when_nothing_is_usable(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        """
        [[providers]]
        name = "gemini"
        kind = "gemini"
        model = "gemini-2.5-flash-lite"
        base_url = "https://example.invalid"
        api_key_env = "ATLAS_TEST_ABSENT_KEY"
        """,
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_CONFIG", str(config))
    monkeypatch.delenv("ATLAS_TEST_ABSENT_KEY", raising=False)
    assert main(["providers"]) == 1
    assert "no usable provider" in capsys.readouterr().out


# ── chat (offline brain still speaks) ────────────────────────────────
def test_chat_falls_back_honestly_when_every_provider_is_down(
    capsys: pytest.CaptureFixture[str], sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("salam\n/quit\n"))
    assert main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "سمح ليا" in out, "the offline line must be spoken, not silence"
    assert "canned" in out, "and it must say who answered"


def test_chat_reports_timings_and_rejects_unknown_commands(
    capsys: pytest.CaptureFixture[str], sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("/nope\n/timing\n/quit\n"))
    assert main(["chat"]) == 0
    out = capsys.readouterr().out
    assert "unknown command" in out
    assert "no timings recorded yet" in out


# ── vault ────────────────────────────────────────────────────────────
def test_vault_lifecycle_init_status_undo(
    capsys: pytest.CaptureFixture[str], sandbox: Path
) -> None:
    vault = sandbox / "MyVault"
    assert main(["vault", "init", str(vault)]) == 0
    assert (vault / "50_Atlas" / "memory.md").is_file()
    capsys.readouterr()

    assert main(["vault", "status", "--vault", str(vault)]) == 0
    out = capsys.readouterr().out
    assert "MyVault" in out and "journal_commits" in out

    # writing through the adapter is committed, and `undo` reverts it
    from atlas_core.config import ObsidianSection
    from atlas_obsidian import VaultAdapter

    adapter = VaultAdapter(ObsidianSection(vault_path=str(vault)))
    adapter.append_daily("hadchi ghadi ytreverta")

    assert main(["vault", "undo", "--vault", str(vault)]) == 0
    assert "reverted" in capsys.readouterr().out


def test_vault_undo_without_history_is_reported(
    capsys: pytest.CaptureFixture[str], sandbox: Path
) -> None:
    vault = sandbox / "EmptyVault"
    main(["vault", "init", str(vault), "--no-git"])
    capsys.readouterr()
    assert main(["vault", "undo", "--vault", str(vault)]) == 1
    assert "nothing to undo" in capsys.readouterr().out


def test_vault_status_without_a_vault_explains_the_fix(
    capsys: pytest.CaptureFixture[str], sandbox: Path
) -> None:
    assert main(["vault", "status"]) == 1
    assert "vault init" in capsys.readouterr().out


# ── skills ───────────────────────────────────────────────────────────
def test_skills_lists_permissions(capsys: pytest.CaptureFixture[str], sandbox: Path) -> None:
    assert main(["skills"]) == 0
    out = capsys.readouterr().out
    assert "system_stats" in out and "safe" in out
    assert "shutdown_pc" in out and "confirm" in out
    assert "owner only" in out
    assert "طفي البيسي" in out, "descriptions must exist in Darija too"


def test_listen_is_honest_about_the_level(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["listen"]) == 0
    out = capsys.readouterr().out
    assert "L2" in out
    assert "chat" in out, "it must point at what works today"


# ── config failures ──────────────────────────────────────────────────
def test_broken_config_is_reported_not_raised(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[app\nbroken", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_CONFIG", str(config))
    assert main(["providers"]) == 1
    assert "config error" in capsys.readouterr().out


def test_runtime_state_stays_inside_the_workspace(
    capsys: pytest.CaptureFixture[str], sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI run must keep its state in ./data, never inside the installed package."""
    monkeypatch.setattr("sys.stdin", io.StringIO("salam\n/quit\n"))
    main(["chat"])
    capsys.readouterr()
    assert (sandbox / "data").is_dir()
    assert (sandbox / "data" / "quota.json").is_file()
    assert (sandbox / "data" / "timings.jsonl").is_file()
