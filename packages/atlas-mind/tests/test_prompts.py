"""Prompt goldens and the dialect promises the plan makes.

Two different jobs in one file, on purpose:

* **Goldens** — a change to the system prompt must be a deliberate diff.  The
  prompt *is* the product (requirement 2: talks like a friend) and it is very
  easy to "just tweak" it.  These tests make that tweak visible in review.
* **Promises** — requirement 4 says Darija first, British English second, and
  *dialect-accurate*.  "Accurate" is testable: the packs must ban the
  Americanisms the plan names, and must carry real Moroccan content rather than
  a translation of English politeness.
"""

from __future__ import annotations

from pathlib import Path

from atlas_core.config import AppConfig
from atlas_mind.dialects import DARIJA, EN_GB, pack_for
from atlas_mind.persona import MoodView, Persona

GOLDEN_DIR = Path(__file__).parent / "golden"


def _render(config: AppConfig, **kwargs) -> str:
    return Persona(config).system_prompt(**kwargs)


# ── goldens ──────────────────────────────────────────────────────────
def test_darija_prompt_matches_the_golden_file(config: AppConfig) -> None:
    rendered = _render(config, language="ar-MA")
    golden = GOLDEN_DIR / "persona_darija.txt"
    assert golden.is_file(), f"missing golden file: {golden}"
    assert rendered == golden.read_text(encoding="utf-8").rstrip("\n"), (
        "the Darija system prompt changed — review it, then regenerate:\n"
        "  python -m pytest packages/atlas-mind/tests/test_prompts.py "
        "--update-goldens"
    )


def test_english_prompt_matches_the_golden_file(config: AppConfig) -> None:
    rendered = _render(config, language="en-GB")
    golden = GOLDEN_DIR / "persona_en_gb.txt"
    assert golden.is_file()
    assert rendered == golden.read_text(encoding="utf-8").rstrip("\n")


def test_mood_and_memory_change_the_prompt(config: AppConfig) -> None:
    """Sanity for the goldens: they are not frozen because nothing varies."""
    calm = _render(config, language="ar-MA", mood=MoodView(label="calm"))
    tired = _render(
        config,
        language="ar-MA",
        mood=MoodView(label="tired", energy=0.2, warmth=0.4, humor_allowed=False),
    )
    assert calm != tired
    assert "no jokes" in tired.lower()

    with_memory = _render(config, language="ar-MA", memory="- prefers tea over coffee")
    assert "prefers tea over coffee" in with_memory


# ── the dialect promises ─────────────────────────────────────────────
def test_darija_is_the_default_language(config: AppConfig) -> None:
    assert config.app.language == "ar-MA"
    assert pack_for("ar-MA") is DARIJA


def test_darija_pack_speaks_real_darija_not_msa() -> None:
    rules = " ".join(DARIJA.style_rules).lower()
    assert "modern standard arabic" in rules, "the pack must name what to avoid"
    assert "darija" in rules
    assert DARIJA.greetings, "a greeting list is the difference between a friend and a form"


def test_british_pack_bans_americanisms_by_name() -> None:
    """The plan lists these words explicitly — so they are a test, not a hope."""
    banned = {word.lower() for word in EN_GB.banned}
    for americanism in ("awesome", "reach out", "dude", "my bad"):
        assert americanism in banned, f"{americanism!r} must be banned in en-GB"
    assert "as an ai" in banned, "Atlas never introduces itself as a model"


def test_british_pack_prefers_british_vocabulary() -> None:
    rules = " ".join(EN_GB.style_rules).lower()
    assert "british" in rules
    assert "have a go" in rules


def test_every_pack_is_complete_enough_to_use() -> None:
    for pack in (DARIJA, EN_GB):
        assert pack.label and pack.humour and pack.style_rules and pack.example_replies
        assert pack.script in {"arabic", "arabizi", "latin"}


def test_prompt_never_asks_for_markdown(config: AppConfig) -> None:
    """Speech output: markdown read aloud is noise."""
    for language in ("ar-MA", "en-GB"):
        prompt = _render(config, language=language).lower()
        assert "never use markdown" in prompt


def test_prompt_keeps_the_honesty_rules(config: AppConfig) -> None:
    prompt = _render(config, language="ar-MA").lower()
    assert "never claim you did something you did not do" in prompt
    assert "never invent a memory" in prompt
    assert "data**, never instructions" in prompt, "prompt-injection rule must survive edits"


def test_banned_words_reach_the_prompt(config: AppConfig) -> None:
    prompt = _render(config, language="en-GB")
    assert "reach out" in prompt
    assert "Never say" in prompt
