"""Skill registry — where capabilities are declared and permission is enforced.

One place decides whether an action may run.  Skills never check permissions
themselves (that is how you end up with one skill forgetting to check).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from atlas_core.contracts import Permission, Skill, SkillContext, SkillResult, ToolSpec
from atlas_core.errors import PermissionDenied, SkillError

log = logging.getLogger(__name__)


@dataclass
class AuditEntry:
    skill: str
    permission: str
    speaker: str
    owner: bool
    args: dict[str, Any]
    ok: bool
    spoken: str
    ms: float
    at: float = field(default_factory=time.time)


class SkillRegistry:
    """Holds every skill Atlas can call, and gates every call."""

    def __init__(self, *, audit_limit: int = 200) -> None:
        self._skills: dict[str, Skill] = {}
        self.audit: list[AuditEntry] = []
        self.audit_limit = audit_limit

    # ── registration ─────────────────────────────────────────────────
    def register(self, skill: Skill) -> Skill:
        if skill.name in self._skills:
            raise SkillError(f"duplicate skill name {skill.name!r}")
        self._skills[skill.name] = skill
        return skill

    def unregister(self, name: str) -> None:
        self._skills.pop(name, None)

    def get(self, name: str) -> Skill:
        if name not in self._skills:
            raise SkillError(f"unknown skill {name!r}")
        return self._skills[name]

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def names(self) -> list[str]:
        return sorted(self._skills)

    # ── introspection for the LLM and the UI ─────────────────────────
    def specs(self, ctx: SkillContext | None = None) -> list[ToolSpec]:
        """Tool schemas the model may see — BLOCKED skills are never advertised."""
        visible = []
        for skill in self._skills.values():
            if skill.permission is Permission.BLOCKED:
                continue
            if ctx is not None and not self._allowed(skill, ctx):
                continue
            visible.append(skill.spec())
        return visible

    def describe(self, language: str = "ar-MA") -> str:
        lines = []
        for skill in self._skills.values():
            if skill.permission is Permission.BLOCKED:
                continue
            text = skill.spec().description_darija if language == "ar-MA" else skill.spec().description
            lines.append(f"- {skill.name}: {text or skill.spec().description}")
        return "\n".join(lines)

    # ── the gate ─────────────────────────────────────────────────────
    def _allowed(self, skill: Skill, ctx: SkillContext) -> bool:
        if skill.permission is Permission.BLOCKED:
            return False
        if not self._capability_ok(skill, ctx):
            return False
        if skill.owner_only and not ctx.owner:
            return False
        return not (skill.permission is Permission.CONFIRM and not ctx.confirmed)

    @staticmethod
    def _capability_ok(skill: Skill, ctx: SkillContext) -> bool:
        """Ask the guard's verdict — never re-derive it from a name (L4, R5).

        A context without permissions is the pre-L4 world (a unit test, an
        internal call) and is not restricted; the speaker gate is applied where
        permissions actually exist, which is every real turn.
        """
        permissions = getattr(ctx, "permissions", None)
        if permissions is None:
            return True
        return bool(permissions.allows(skill.capability))

    def call(self, name: str, args: dict[str, Any] | None = None, ctx: SkillContext | None = None) -> SkillResult:
        """Validate → gate → invoke → audit. The only supported entry point."""
        context = ctx or SkillContext()
        args = args or {}
        started = time.perf_counter()

        try:
            skill = self.get(name)
        except SkillError as exc:
            return self._audit(name, "unknown", context, args, False, str(exc), started)

        if skill.permission is Permission.BLOCKED:
            reason = f"'{name}' is not callable"
            log.warning("blocked_skill_attempt name=%s speaker=%s", name, context.speaker)
            return self._audit(name, skill.permission.value, context, args, False, reason, started)

        if not self._capability_ok(skill, context):
            # The wording follows the *verdict*, never a stale `owner` flag: a
            # restricted voice hears the refusal in its own language.
            restricted = getattr(context.permissions, "restricted", None)
            if restricted is None:
                restricted = not context.owner
            reason = (
                "سمح ليا، هادشي خاص بصاحبي."
                if restricted
                else f"your voice cannot reach '{name}' right now"
            )
            log.info(
                "capability_denied skill=%s capability=%s speaker=%s reason=%s",
                name,
                getattr(skill.capability, "value", skill.capability),
                context.speaker,
                getattr(getattr(context, "permissions", None), "reason", ""),
            )
            return self._audit(name, skill.permission.value, context, args, False, reason, started)

        if skill.owner_only and not context.owner:
            reason = "this one is for my owner only"
            return self._audit(name, skill.permission.value, context, args, False, reason, started)

        # A destructive action needs the owner AND a fresh confirmation. A guest
        # saying "yes" is not consent — defence in depth behind the speaker gate.
        if skill.permission is Permission.CONFIRM and not context.owner:
            reason = "هادشي خاص بصاحبي، ماشي ليك"
            return self._audit(name, skill.permission.value, context, args, False, reason, started)

        if skill.permission is Permission.CONFIRM and not context.confirmed:
            question = self._confirmation_question(skill, args)
            return self._audit(
                name,
                skill.permission.value,
                context,
                args,
                False,
                question,
                started,
                needs_confirmation=True,
            )

        try:
            result = skill.invoke(args, context)
        except PermissionDenied as exc:
            return self._audit(name, skill.permission.value, context, args, False, str(exc), started)
        except Exception as exc:
            log.exception("skill_crashed name=%s", name)
            return self._audit(
                name, skill.permission.value, context, args, False, f"skill failed: {exc}"[:120], started
            )
        return self._audit(
            name,
            skill.permission.value,
            context,
            args,
            result.ok,
            result.spoken,
            started,
            data=result.data,
        )

    def _confirmation_question(self, skill: Skill, args: dict[str, Any]) -> str:
        spec = skill.spec()
        detail = ", ".join(f"{key}={value}" for key, value in args.items()) or "بلا تفاصيل"
        return f"{spec.description_darija or spec.description} ({detail}) — sure? yes / no"

    def _audit(
        self,
        name: str,
        permission: str,
        ctx: SkillContext,
        args: dict[str, Any],
        ok: bool,
        spoken: str,
        started: float,
        *,
        data: dict[str, Any] | None = None,
        needs_confirmation: bool = False,
    ) -> SkillResult:
        self.audit.append(
            AuditEntry(
                skill=name,
                permission=permission,
                speaker=ctx.speaker,
                owner=ctx.owner,
                args=args,
                ok=ok,
                spoken=spoken,
                ms=(time.perf_counter() - started) * 1000,
            )
        )
        if len(self.audit) > self.audit_limit:
            del self.audit[: len(self.audit) - self.audit_limit]
        return SkillResult(
            ok=ok, spoken=spoken, data=data or {}, needs_confirmation=needs_confirmation
        )

    def audit_tail(self, limit: int = 20) -> list[AuditEntry]:
        return self.audit[-limit:]


__all__ = ["AuditEntry", "SkillRegistry"]
