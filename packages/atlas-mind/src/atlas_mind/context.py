"""Context assembly under a token budget.

Atlas's whole latency story is "short prompt, stream early".  So context is
built with a hard budget and a strict priority order:

    system persona  →  recalled memory  →  recent turns (newest kept)  →  user

Anything that does not fit is dropped, oldest first.  Dropping a turn costs a
little continuity; blowing the budget costs the whole experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from atlas_core.contracts import LlmRequest, Message, Role, estimate_tokens


@dataclass
class ContextBuilder:
    """Builds `LlmRequest`s that always fit the budget."""

    budget_tokens: int = 1200
    memory_budget: int = 400
    max_turns: int = 12
    config_temperature: float = 0.7
    max_output_tokens: int = 512
    last_build: dict[str, int] = field(default_factory=dict)

    def build(
        self,
        *,
        system: str,
        user_text: str,
        history: list[Message] | None = None,
        memory: str = "",
        tools: list | None = None,
        json_schema: dict | None = None,
        language: str = "",
    ) -> LlmRequest:
        history = list(history or [])[-self.max_turns :]

        system_cost = estimate_tokens(system)
        user_cost = estimate_tokens(user_text)

        memory = self._fit_memory(memory, system_cost + user_cost)
        memory_cost = estimate_tokens(memory) if memory else 0

        remaining = self.budget_tokens - system_cost - memory_cost - user_cost
        kept: list[Message] = []
        for message in reversed(history):
            cost = estimate_tokens(message.content)
            if cost > remaining:
                break
            kept.append(message)
            remaining -= cost
        kept.reverse()

        if memory:
            kept.insert(0, Message(role=Role.USER, content=f"[remembered context]\n{memory}"))

        self.last_build = {
            "system": system_cost,
            "memory": memory_cost,
            "history": sum(estimate_tokens(m.content) for m in kept),
            "user": user_cost,
            "budget": self.budget_tokens,
            "turns_kept": len(kept),
        }
        return LlmRequest(
            messages=[*kept, Message(role=Role.USER, content=user_text)],
            system=system,
            tools=tools or [],
            temperature=self.config_temperature,
            max_output_tokens=self.max_output_tokens,
            json_schema=json_schema,
            language=language,
        )

    def _fit_memory(self, memory: str, spent: int) -> str:
        """Truncate recalled memory to its own budget, newest lines last."""
        if not memory.strip():
            return ""
        allowance = min(self.memory_budget, max(0, self.budget_tokens - spent - 200))
        if allowance <= 0:
            return ""
        lines = [line for line in memory.splitlines() if line.strip()]
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            cost = estimate_tokens(line)
            if used + cost > allowance:
                break
            kept.append(line)
            used += cost
        return "\n".join(reversed(kept))


__all__ = ["ContextBuilder"]
