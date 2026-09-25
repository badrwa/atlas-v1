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


# ── L4: the capability gate ──────────────────────────────────────────
# `ctx.owner` is a *display* flag.  The gate is `ctx.permissions`, which comes
# from the speaker verifier — so these tests pass an owner flag that disagrees
# with the verdict, on purpose: the verdict must win.
def _gated(
    *,
    capabilities=(),
    owner: bool = False,
    subject: str = "said",
    displayed_owner: bool | None = None,
) -> SkillContext:
    """A context whose *only* identity source is the guard's verdict.

    `displayed_owner` exists so a test can hand the registry a stale flag that
    disagrees with the verdict — the same shape a half-wired caller would have.
    """
    from atlas_core.identity import Permissions

    return SkillContext(
        owner=owner if displayed_owner is None else displayed_owner,
        permissions=Permissions(
            capabilities=frozenset(capabilities),
            speaker=subject or "unknown",
            subject=subject,
            known=bool(subject),
            owner=owner,
            accepted=owner,
        ),
    )


def test_a_restricted_voice_cannot_shutdown_even_with_the_owner_flag() -> None:
    from atlas_core.identity import RESTRICTED_CAPABILITIES

    calls = _recorder()
    registry = _registry(ShutdownSkill(action=calls.fire))
    context = _gated(capabilities=RESTRICTED_CAPABILITIES, owner=False, subject="",
                     displayed_owner=True)
    result = registry.call("shutdown_pc", {}, context)

    assert result.ok is False
    assert calls.calls == []
    # …and the refusal is the Darija one, because the *verdict* is restricted
    # even though the caller's own flag claimed otherwise.
    assert "صاحبي" in result.spoken
    assert context.owner is False, "the verdict overrides the stale flag"


def test_a_restricted_voice_cannot_control_the_pc_or_write_notes() -> None:
    from atlas_core.identity import RESTRICTED_CAPABILITIES

    opened = _recorder()
    registry = _registry(SystemStatsSkill(), OpenUrlSkill(opened.open))
    context = _gated(capabilities=RESTRICTED_CAPABILITIES, subject="")

    for name, args in (
        ("system_stats", {}),
        ("open_url", {"url": "https://example.com"}),
    ):
        result = registry.call(name, args, context)
        assert result.ok is False, name
    assert opened.calls == []


def test_general_conversation_still_works_for_a_restricted_voice() -> None:
    from atlas_core.identity import RESTRICTED_CAPABILITIES

    registry = _registry(FakeSkill())
    # FakeSkill carries the default GENERAL capability, so a stranger may use it.
    stranger = _gated(capabilities=RESTRICTED_CAPABILITIES, subject="")
    assert registry.call("fake_skill", {}, stranger).ok is True


def replace_context(context: SkillContext) -> SkillContext:
    """The same verdict, plus the confirmation the CONFIRM gate demands."""
    from dataclasses import replace

    return replace(context, confirmed=True)


def test_the_owner_keeps_everything() -> None:
    from atlas_core.identity import OWNER_CAPABILITIES

    calls = _recorder()
    registry = _registry(ShutdownSkill(action=calls.fire), SystemStatsSkill())
    context = _gated(capabilities=OWNER_CAPABILITIES, owner=True, subject="badr")
    assert registry.call("system_stats", {}, context).ok is True
    confirmed = replace_context(context)
    assert registry.call("shutdown_pc", {}, confirmed).ok is True


def test_a_denied_call_is_audited_and_never_reaches_the_action() -> None:
    from atlas_core.identity import RESTRICTED_CAPABILITIES

    calls = _recorder()
    registry = _registry(ShutdownSkill(action=calls.fire))
    registry.call(
        "shutdown_pc", {}, _gated(capabilities=RESTRICTED_CAPABILITIES, subject="")
    )
    entry = registry.audit_tail(1)[0]
    assert entry.ok is False
    assert entry.skill == "shutdown_pc"
    assert "صاحبي" in entry.spoken, "the audit records the refusal the person heard"
    assert entry.owner is False, "the audit records the verdict, not a caller's flag"
    assert calls.calls == []


def test_a_guest_writes_facts_about_themselves_and_never_into_memory(
    tmp_path: Path,
) -> None:
    from atlas_core.identity import KNOWN_CAPABILITIES

    vault = VaultAdapter.init_from_template(
        tmp_path / "v", REPO_ROOT / "vault-template", git=False
    )
    registry = _registry(RememberSkill(vault))
    context = _gated(capabilities=KNOWN_CAPABILITIES, subject="said", owner=False)

    result = registry.call("remember", {"fact": "atay b nan3na3"}, context)
    assert result.ok is True
    # `ctx.subject` came from the verdict, so the fact landed in `said`'s note.
    assert context.subject == "said"
    assert "said" in result.spoken
    assert "atay" in vault.read_person("said")
    assert "atay" not in vault.read_memory(), "a guest's fact never enters the owner's memory"


def test_a_context_without_permissions_is_the_pre_l4_world() -> None:
    """Unit tests and internal callers keep working — no permissions, no gate."""
    calls = _recorder()
    registry = _registry(ShutdownSkill(action=calls.fire))
    assert registry.call("shutdown_pc", {}, _ctx(confirmed=True)).ok is True
