"""CLI tests — the commands a human actually types, run for real.

No mocks: `main()` is called with a temporary config and a temporary working
directory, and the output is read back.  If `atlas doctor` cannot run in CI, it
will not run on the laptop either.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from atlas.cli import main

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE = REPO_ROOT / "vault-template"

# A provider that fails instantly and locally: no key, no network, no waiting.
OFFLINE_CONFIG = """
[app]
profile = "lean"
language = "ar-MA"

[mind]
provider_order = ["deadlocal"]

[[providers]]
name = "deadlocal"
kind = "openai_compatible"
model = "none"
base_url = "http://127.0.0.1:9/v1"
timeout_s = 1.0
"""


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway working directory with its own config.toml."""
    (tmp_path / "config.toml").write_text(OFFLINE_CONFIG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_CONFIG", str(tmp_path / "config.toml"))
    return tmp_path


@pytest.fixture
def vc(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[Path, Callable[..., tuple[int, str]]]:
    """Run a command and return (exit code, stdout)."""

    def run(*argv: str) -> tuple[int, str]:
        code = main(list(argv))
        return code, capsys.readouterr().out

    return workspace, run


# ── version / protocol ───────────────────────────────────────────────
def test_version_prints_the_pieces(vc) -> None:
    code, out = vc[1]("version")
    assert code == 0
    assert "atlas 0.1.0" in out
    assert "atlas-core" in out and "python" in out


def test_ui_protocol_emits_typescript(vc) -> None:
    code, out = vc[1]("ui-protocol")
    assert code == 0
    assert out.startswith("// AUTO-GENERATED")
    assert "export type AtlasMessage" in out
    assert "powers: Record<string, boolean>" in out, "containers must not degrade to boolean"


def test_ui_protocol_writes_a_file(vc) -> None:
    target = vc[0] / "protocol.ts"
    code, out = vc[1]("ui-protocol", "--out", str(target))
    assert code == 0
    assert "written" in out
    assert target.read_text(encoding="utf-8").startswith("// AUTO-GENERATED")


# ── doctor / health ──────────────────────────────────────────────────
def test_doctor_reports_instead_of_crashing(vc) -> None:
    code, out = vc[1]("doctor")
    assert code in (0, 1), "doctor reports problems; it does not fail on them"
    assert "ATLAS doctor" in out
    assert "ok ·" in out and "failures" in out
    assert "config.toml" in out


def test_doctor_says_which_piece_reaches_which_level(vc) -> None:
    _, out = vc[1]("doctor")
    assert "wake word" in out
    assert "L2" in out, "the voice loop is L2; doctor must not pretend otherwise"


def test_doctor_warns_about_a_key_of_the_wrong_shape(
    vc, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = vc[0]
    (workspace / "config.toml").write_text(
        """
        [[providers]]
        name = "gemini"
        kind = "gemini"
        model = "gemini-2.5-flash-lite"
        api_key_env = "ATLAS_TEST_BAD_GEMINI"
        base_url = "https://example.invalid"
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("ATLAS_TEST_BAD_GEMINI", "AQ.not-a-real-key")
    _, out = vc[1]("doctor")
    assert "not an AI Studio key" in out
    assert "aistudio.google.com" in out
    assert "AQ.not-a-real-key" not in out, "never echo a secret, not even a broken one"


def test_health_is_machine_readable(vc) -> None:
    code, out = vc[1]("health")
    payload = json.loads(out)
    assert code == 0 and payload["ok"] is True
    assert any(check["name"] == "RAM" for check in payload["checks"])


# ── providers ────────────────────────────────────────────────────────
def test_providers_lists_local_ones_as_usable_without_a_key(vc) -> None:
    """Ollama and llama.cpp need no key — that is the whole point of having them."""
    code, out = vc[1]("providers")
    assert code == 0
    assert "deadlocal" in out
    assert "key ✓" in out


def test_providers_explains_when_no_key_is_set(vc) -> None:
    vc[0].joinpath("config.toml").write_text(
        """
        [[providers]]
        name = "gemini"
        kind = "gemini"
        model = "gemini-2.5-flash-lite"
        api_key_env = "ATLAS_TEST_MISSING_KEY"
        base_url = "https://example.invalid"
        """,
        encoding="utf-8",
    )
    code, out = vc[1]("providers")
    assert code == 1
    assert "no key" in out
    assert "no usable provider" in out


def test_providers_marks_local_and_cloud_apart(vc, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = vc[0]
    (workspace / "config.toml").write_text(
        """
        [[providers]]
        name = "groq"
        kind = "openai_compatible"
        model = "llama-3.3-70b-versatile"
        base_url = "https://api.groq.com/openai/v1"
        api_key_env = "ATLAS_TEST_GROQ"
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("ATLAS_TEST_GROQ", "secret-value")
    code, out = vc[1]("providers")
    assert code == 0
    assert "key ✓" in out
    assert "secret-value" not in out, "keys are never printed"


# ── skills ───────────────────────────────────────────────────────────
def test_skills_show_permissions_and_darija(vc) -> None:
    code, out = vc[1]("skills")
    assert code == 0
    assert "system_stats" in out and "shutdown_pc" in out
    assert "confirm" in out and "owner only" in out
    assert "طفي البيسي" in out, "a Darija assistant describes itself in Darija"


def test_skills_audit_starts_empty(vc) -> None:
    code, out = vc[1]("skills", "audit")
    assert code == 0
    assert "nothing called yet" in out


# ── vault ────────────────────────────────────────────────────────────
def test_vault_init_status_and_undo(vc) -> None:
    workspace, run = vc
    vault = workspace / "Vault"
    code, out = run("vault", "init", str(vault), "--template", str(TEMPLATE))
    assert code == 0, out
    assert (vault / "50_Atlas" / "memory.md").is_file()

    code, out = run("vault", "status", "--vault", str(vault))
    assert code == 0
    assert "journal_commits" in out

    from atlas_core.config import ObsidianSection
    from atlas_obsidian import VaultAdapter

    adapter = VaultAdapter(ObsidianSection(vault_path=str(vault)))
    adapter.append_daily("hadchi ghadi ytreverta")

    code, out = run("vault", "undo", "--vault", str(vault))
    assert code == 0
    assert "reverted" in out


def test_vault_init_without_a_path_explains_itself(vc) -> None:
    code, out = vc[1]("vault", "init")
    assert code == 1
    assert "tell me where" in out


def test_vault_undo_with_nothing_to_undo(vc) -> None:
    workspace, run = vc
    vault = workspace / "Empty"
    run("vault", "init", str(vault), "--template", str(TEMPLATE))
    code, out = run("vault", "undo", "--vault", str(vault))
    assert code == 1
    assert "nothing to undo" in out


# ── chat ─────────────────────────────────────────────────────────────
def test_chat_falls_back_with_an_honest_line(vc) -> None:
    code, out = vc[1]("chat", "--once", "salam")
    assert code == 0
    assert "سمح ليا" in out, "the offline line must be spoken, not silence"
    assert "canned" in out, "and it must say who answered"


def test_chat_commands_are_understood(vc, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("/timing\n/lang\n/quit\n"))
    code, out = vc[1]("chat")
    assert code == 0
    assert "no timings recorded yet" in out
    assert "language is ar-MA" in out


def test_unknown_chat_command_is_explained(vc, monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("/nope\n/quit\n"))
    _, out = vc[1]("chat")
    assert "unknown command" in out


# ── argument handling ────────────────────────────────────────────────
def test_missing_command_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
    capsys.readouterr()


def test_startup_keeps_state_in_the_working_directory(vc) -> None:
    workspace = vc[0]
    vc[1]("doctor")
    assert (workspace / "data").is_dir()
    assert (workspace / "logs").is_dir()
