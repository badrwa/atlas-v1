"""The permission truth table — the most important test file in the project.

Every combination of (permission × owner × confirmed × dry-run) is asserted.
If this file goes green by accident, the gate is broken.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_core.config import ObsidianSection
from atlas_core.contracts import Permission, Skill, SkillContext, SkillResult
from atlas_core.errors import SkillError
from atlas_core.fakes import FakeSkill
from atlas_obsidian.vault import VaultAdapter
from atlas_skills.packs.system import (
    OpenUrlSkill,
    RememberSkill,
    ShutdownSkill,
    SystemStatsSkill,
)
from atlas_skills.registry import SkillRegistry

REPO_ROOT = Path(__file__).resolve().parents[3]


def _ctx(
    *,
    speaker: str = "badr",
    owner: bool = True,
    dry_run: bool = False,
    confirmed: bool = False,
) -> SkillContext:
    return SkillContext(speaker=speaker, owner=owner, dry_run=dry_run, confirmed=confirmed)


class _Recorder:
    """Records what a fake action did — and returns True, like a real action would."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def open(self, url: str) -> bool:
        self.calls.append(url)
        return True

    def fire(self) -> bool:
        self.calls.append("shutdown")
        return True


def _recorder() -> _Recorder:
    return _Recorder()


def _registry(*skills: Skill) -> SkillRegistry:
    registry = SkillRegistry()
    for skill in skills:
        registry.register(skill)
    return registry


def _fake(name: str, permission: Permission = Permission.SAFE, **kwargs) -> FakeSkill:
    """FakeSkill with a unique name (the registry refuses duplicates)."""
    skill = FakeSkill(permission=permission, **kwargs)
    skill.name = name
    return skill


class BlockedSkill(FakeSkill):
    name = "blocked_thing"
    permission = Permission.BLOCKED


class OwnerOnlySkill(FakeSkill):
    name = "owner_thing"
    permission = Permission.SAFE
    owner_only = True


class ExplodingSkill(FakeSkill):
    name = "exploding"

    def invoke(self, args, ctx):  # type: ignore[override]
        raise RuntimeError("boom")


# ── registration ─────────────────────────────────────────────────────
def test_duplicate_names_are_refused() -> None:
    registry = _registry(_fake("same_name"))
    with pytest.raises(SkillError):
        registry.register(_fake("same_name"))


def test_unknown_skill_is_reported_not_executed() -> None:
    result = _registry().call("nope", {}, _ctx())
    assert result.ok is False
    assert "unknown skill" in result.spoken


# ── the truth table ──────────────────────────────────────────────────
@pytest.mark.parametrize(
    ("permission", "owner", "confirmed", "expected_ok"),
    [
        (Permission.SAFE, True, False, True),
        (Permission.SAFE, False, False, True),
        (Permission.CONFIRM, True, False, False),   # asks first
        (Permission.CONFIRM, True, True, True),     # owner confirmed
        (Permission.CONFIRM, False, True, False),   # stranger's "yes" is not consent
        (Permission.BLOCKED, True, True, False),    # never, under any circumstances
        (Permission.BLOCKED, True, False, False),
    ],
)
def test_permission_truth_table(permission, owner, confirmed, expected_ok) -> None:
    skill = FakeSkill(permission=permission)
    registry = _registry(skill)
    result = registry.call(skill.name, {}, _ctx(owner=owner, confirmed=confirmed))
    assert result.ok is expected_ok, f"{permission} owner={owner} confirmed={confirmed}"


def test_confirm_skill_returns_the_question_not_an_action() -> None:
    skill = _fake("confirm_thing", Permission.CONFIRM)
    registry = _registry(skill)
    result = registry.call(skill.name, {"path": "Downloads"}, _ctx(confirmed=False))
    assert result.needs_confirmation is True
    assert "yes" in result.spoken.lower()
    assert skill.calls == [], "nothing may execute before confirmation"


def test_confirm_skill_needs_the_owner_even_when_confirmed() -> None:
    skill = _fake("confirm_thing", Permission.CONFIRM)
    registry = _registry(skill)
    result = registry.call("confirm_thing", {}, _ctx(owner=False, confirmed=True))
    assert result.ok is False
    assert skill.calls == [], "a guest must never reach a destructive action"


def test_owner_only_skill_refuses_a_guest() -> None:
    registry = _registry(OwnerOnlySkill())
    assert registry.call("owner_thing", {}, _ctx(owner=False)).ok is False
    assert registry.call("owner_thing", {}, _ctx(owner=True)).ok is True


def test_blocked_skills_are_never_advertised_to_the_model() -> None:
    registry = _registry(_fake("fake_skill"), BlockedSkill(), ShutdownSkill())
    names = [spec.name for spec in registry.specs()]
    assert "blocked_thing" not in names
    assert "shutdown_pc" in names, "CONFIRM skills are visible — they are asked about, not run"
    assert "blocked_thing" not in registry.describe()
    assert "blocked_thing" not in registry.describe("en-GB")


def test_a_crashing_skill_does_not_kill_the_turn() -> None:
    registry = _registry(ExplodingSkill())
    result = registry.call("exploding", {}, _ctx())
    assert result.ok is False
    assert "failed" in result.spoken


# ── audit ────────────────────────────────────────────────────────────
def test_every_call_is_audited_including_refusals() -> None:
    registry = _registry(
        _fake("fake_skill"), _fake("confirm_thing", Permission.CONFIRM), BlockedSkill()
    )
    registry.call("fake_skill", {}, _ctx())
    registry.call("confirm_thing", {}, _ctx(confirmed=False))
    registry.call("blocked_thing", {}, _ctx())
    registry.call("does_not_exist", {}, _ctx())

    tail = registry.audit_tail()
    assert len(tail) == 4
    assert [entry.ok for entry in tail] == [True, False, False, False]
    assert all(entry.ms >= 0 for entry in tail)
    assert tail[0].speaker == "badr" and tail[0].owner is True


# ── real skills ──────────────────────────────────────────────────────
def test_system_stats_reports_real_numbers() -> None:
    skill = SystemStatsSkill(ram_probe=lambda: (8192, 2048), disk_probe=lambda: (256, 40), battery_probe=lambda: (73, False))
    result = skill.invoke({"what": "all"}, _ctx())
    assert result.ok and result.data["ram_free_mb"] == 2048
    assert "RAM" in result.spoken
    assert result.data["battery_percent"] == 73


def test_system_stats_rejects_an_unknown_metric() -> None:
    skill = SystemStatsSkill(ram_probe=lambda: (1, 1), disk_probe=lambda: (1, 1), battery_probe=lambda: (1, True))
    assert skill.invoke({"what": "gpu"}, _ctx()).ok is False


def test_open_url_refuses_non_http_schemes() -> None:
    opened = _recorder()
    skill = OpenUrlSkill(opener=opened.open)
    assert skill.invoke({"url": "file:///C:/Windows"}, _ctx()).ok is False
    assert skill.invoke({"url": "javascript:alert(1)"}, _ctx()).ok is False
    assert opened.calls == []


def test_open_url_opens_http() -> None:
    opened = _recorder()
    skill = OpenUrlSkill(opener=opened.open)
    result = skill.invoke({"url": "https://example.com"}, _ctx())
    assert result.ok is True
    assert opened.calls == ["https://example.com"]


def test_dry_run_mode_never_touches_the_system() -> None:
    opened = _recorder()
    skill = OpenUrlSkill(opener=opened.open)
    result = skill.invoke({"url": "https://example.com"}, _ctx(dry_run=True))
    assert result.ok and "[dry-run]" in result.spoken
    assert opened.calls == []
    assert ShutdownSkill(action=lambda: True).invoke({}, _ctx(dry_run=True)).data["dry_run"] is True


def test_remember_writes_into_the_obsidian_vault(tmp_path: Path) -> None:
    vault = VaultAdapter.init_from_template(tmp_path / "v", REPO_ROOT / "vault-template", git=True)
    skill = RememberSkill(vault)
    result = skill.invoke({"fact": "sahbi kayt9an f atay b nana"}, _ctx())
    assert result.ok is True
    assert "atay b nana" in vault.read_memory()


def test_remember_rejects_empty_input(tmp_path: Path) -> None:
    vault = VaultAdapter(ObsidianSection(vault_path=str(tmp_path)))
    assert RememberSkill(vault).invoke({"fact": "  "}, _ctx()).ok is False


def test_shutdown_requires_confirmation_through_the_registry() -> None:
    calls = _recorder()
    registry = _registry(ShutdownSkill(action=calls.fire))

    first = registry.call("shutdown_pc", {}, _ctx(confirmed=False))
    assert first.needs_confirmation is True
    assert calls.calls == []

    second = registry.call("shutdown_pc", {}, _ctx(confirmed=True))
    assert second.ok is True
    assert calls.calls == ["shutdown"]


def test_guest_cannot_shutdown_even_with_confirmation() -> None:
    calls = _recorder()
    registry = _registry(ShutdownSkill(action=calls.fire))
    result = registry.call("shutdown_pc", {}, _ctx(owner=False, confirmed=True))
    assert result.ok is False
    assert calls.calls == []


def test_skill_result_shape_is_stable() -> None:
    result = SkillResult(ok=True, spoken="safi")
    assert result.data == {} and result.needs_confirmation is False
