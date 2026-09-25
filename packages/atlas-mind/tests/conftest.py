"""Shared test fixtures for atlas-mind.

`--update-goldens` regenerates the prompt snapshots.  One flag, one
implementation: the next golden set inherits it for free.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlas_core.config import AppConfig

GOLDEN_DIR = Path(__file__).parent / "golden"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="rewrite prompt golden files instead of comparing against them",
    )


@pytest.fixture
def config() -> AppConfig:
    """Darija-first, no cloud keys — every test here runs offline by design."""
    return AppConfig.model_validate(
        {
            "app": {"language": "ar-MA", "secondary_language": "en-GB", "call_name": "صاحبي"},
            "mind": {"dialect": {"script": "arabic", "code_switch": True}},
        }
    )


@pytest.fixture(autouse=True)
def _write_goldens_when_asked(request: pytest.FixtureRequest) -> None:
    """Regenerate goldens before the assertions run, when the flag is set."""
    if not request.config.getoption("--update-goldens") or "config" not in request.fixturenames:
        return
    from atlas_mind.persona import Persona

    persona = Persona(request.getfixturevalue("config"))
    GOLDEN_DIR.mkdir(exist_ok=True)
    for language, name in (("ar-MA", "persona_darija.txt"), ("en-GB", "persona_en_gb.txt")):
        (GOLDEN_DIR / name).write_text(
            persona.system_prompt(language=language) + "\n", encoding="utf-8"
        )
