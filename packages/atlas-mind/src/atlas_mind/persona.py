"""Persona — assembling the system prompt.

Prompts are templates (`prompts/*.jinja`), dialect rules are data
(`dialects.py`), and preferences come from the vault.  Nothing here is a
scattered f-string, which is why changing Atlas's personality is a five-minute
edit instead of an archaeology dig.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from atlas_core.config import AppConfig
from atlas_core.contracts import LanguageTag
from atlas_mind.dialects import DialectPack, pack_for
from atlas_mind.mood import MoodView

PROMPT_DIR = Path(__file__).parent / "prompts"


class Persona:
    """Builds the system prompt for a turn."""

    def __init__(
        self,
        config: AppConfig,
        *,
        name: str = "Atlas",
        template_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.name = name
        self.owner = "the owner"  # replaced by the speaker's name once L4 lands
        self._env = Environment(
            loader=FileSystemLoader(str(template_dir or PROMPT_DIR)),
            undefined=StrictUndefined,
            autoescape=select_autoescape(enabled_extensions=()),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ── public API ───────────────────────────────────────────────────
    def system_prompt(
        self,
        *,
        language: LanguageTag | str = "ar-MA",
        mood: MoodView | None = None,
        preferences: dict[str, Any] | None = None,
        memory: str = "",
        tool_names: list[str] | None = None,
        max_sentences: int | None = None,
    ) -> str:
        pack = pack_for(str(language))
        template = self._env.get_template("persona.jinja")
        return template.render(
            name=self.name,
            owner=self.owner,
            call_name=self.config.app.call_name,
            language_label=pack.label,
            style_rules=pack.style_rules,
            humour=pack.humour,
            banned=pack.banned,
            language_examples=pack.example_replies,
            mood=mood,
            preferences=preferences or self._default_preferences(),
            memory=memory.strip(),
            tool_names=tool_names or [],
            max_sentences=max_sentences or self._sentence_budget(mood),
        ).strip()

    def switch_notice(self, language: LanguageTag | str) -> str:
        pack = pack_for(str(language))
        return self._env.get_template("language_switch.jinja").render(
            language_label=pack.label, code=pack.code
        ).strip()

    def dialect(self, language: LanguageTag | str) -> DialectPack:
        return pack_for(str(language))

    # ── helpers ──────────────────────────────────────────────────────
    def _default_preferences(self) -> dict[str, Any]:
        dialect = self.config.mind.dialect
        return {
            "default language": self.config.app.language,
            "secondary language": self.config.app.secondary_language,
            "script": dialect.script,
            "code switching": "allowed" if dialect.code_switch else "avoid",
            "answer length": "short",
        }

    @staticmethod
    def _sentence_budget(mood: MoodView | None) -> int:
        if mood is None:
            return 3
        if mood.label in {"serious", "frustrated"}:
            return 2
        if not mood.humor_allowed:
            return 3
        return 4


__all__ = ["PROMPT_DIR", "MoodView", "Persona"]
