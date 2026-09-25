"""Language routing, Darija normalisation, dialect packs and the lexicon."""

from __future__ import annotations

import pytest

from atlas_mind.darija import DarijaNormalizer, Lexicon, arabic_ratio, has_arabic
from atlas_mind.dialects import DARIJA, EN_GB, pack_for
from atlas_mind.language import LanguageRouter


@pytest.fixture
def router() -> LanguageRouter:
    return LanguageRouter(default="ar-MA", secondary="en-GB")


# ── routing ──────────────────────────────────────────────────────────
def test_darija_is_the_default(router: LanguageRouter) -> None:
    assert router.route("") == "ar-MA"
    assert router.route("chno ljaw lyoum?") == "ar-MA"
    assert router.route("واش كاين شي حاجة جديدة؟") == "ar-MA"


def test_latin_text_without_darija_markers_is_english(router: LanguageRouter) -> None:
    assert router.route("what is the weather tomorrow?") == "en-GB"


def test_explicit_switch_to_english(router: LanguageRouter) -> None:
    assert router.route("speak English please") == "en-GB"
    assert router.current == "en-GB"


def test_explicit_switch_back_to_darija(router: LanguageRouter) -> None:
    router.route("speak English please")
    assert router.route("b darija 3afak") == "ar-MA"
    assert router.current == "ar-MA"


def test_sticky_language_survives_ambiguous_input(router: LanguageRouter) -> None:
    router.route("hello there, how are you?")  # → en-GB
    assert router.route("ok") == "en-GB", "short/ambiguous input keeps the session language"


def test_explicit_hint_beats_everything(router: LanguageRouter) -> None:
    assert router.route("chno ljaw?", hint="en-GB") == "en-GB"


def test_language_switch_command_is_not_confused_with_a_question(router: LanguageRouter) -> None:
    # "english" inside a Darija sentence should not flip the language
    assert router.route("wach nta katfehem english mzyan?") == "ar-MA"


def test_reset_returns_to_default(router: LanguageRouter) -> None:
    router.route("speak English")
    router.reset()
    assert router.current == "ar-MA"


# ── detection helpers ────────────────────────────────────────────────
def test_arabic_helpers() -> None:
    assert has_arabic("سلا م") is True
    assert has_arabic("salam") is False
    assert arabic_ratio("سلا م") == 1.0
    assert arabic_ratio("") == 0.0


# ── normalisation ────────────────────────────────────────────────────
def test_normalizer_is_idempotent() -> None:
    normalizer = DarijaNormalizer()
    once = normalizer.normalize_for_match("B3edd el-meghrib, wach kayn chi 7aja?")
    twice = normalizer.normalize_for_match(once)
    assert once == twice


def test_normalizer_maps_arabizi_for_matching_only() -> None:
    normalizer = DarijaNormalizer()
    canonical = normalizer.normalize_for_match("3afak")
    assert "ع" in canonical
    assert normalizer.normalize_for_tts("3afak") == "3afak", "display text is never rewritten"


def test_normalizer_strips_diacritics_and_unifies_letters() -> None:
    normalizer = DarijaNormalizer()
    assert normalizer.normalize_for_match("كَتَبَ") == normalizer.normalize_for_match("كتب")
    assert normalizer.normalize_for_match("إسلام") == normalizer.normalize_for_match("اسلام")


def test_elongation_collapses_but_double_consonants_survive() -> None:
    normalizer = DarijaNormalizer()
    # the property that matters: noisy spelling matches clean spelling
    assert normalizer.normalize_for_match("bezzzaaaf") == normalizer.normalize_for_match("bezzaf")
    assert normalizer.normalize_for_match("mzyaaaan") == normalizer.normalize_for_match("mzyan")
    assert normalizer.normalize_for_match("allah") == "allah"


def test_tts_normalisation_removes_markdown_and_emoji() -> None:
    normalizer = DarijaNormalizer()
    assert normalizer.normalize_for_tts("**salam** 😀 `safi`") == "salam safi"


# ── lexicon ──────────────────────────────────────────────────────────
def test_lexicon_round_trips_through_markdown() -> None:
    lexicon = Lexicon({"mohammedia": "Mohammedia"})
    text = lexicon.to_text()
    assert "- mohammedia => Mohammedia" in text
    assert Lexicon.from_text(text).entries == lexicon.entries


def test_lexicon_learns_and_applies_corrections() -> None:
    lexicon = Lexicon()
    lexicon.learn("obsidiane", "Obsidian")
    assert lexicon.apply("open obsidiane please") == "open Obsidian please"


def test_lexicon_prompt_hint_is_bounded() -> None:
    lexicon = Lexicon({f"wrong{i}": f"right{i}" for i in range(80)})
    hint = lexicon.as_prompt_hint(limit=10)
    assert hint.count("→") == 10


def test_lexicon_from_missing_file_is_empty(tmp_path) -> None:
    assert Lexicon.from_file(tmp_path / "nope.md").entries == {}


# ── dialect packs ────────────────────────────────────────────────────
def test_packs_are_complete() -> None:
    for pack in (DARIJA, EN_GB):
        assert pack.greetings and pack.acknowledgements and pack.style_rules
        assert pack.humour
        assert pack.banned, "every dialect needs a banned-words list"


def test_unknown_language_falls_back_to_darija() -> None:
    assert pack_for("fr-FR") is DARIJA


def test_british_pack_bans_americanisms() -> None:
    banned = " ".join(EN_GB.banned).lower()
    for americanism in ("awesome", "reach out", "my bad"):
        assert americanism in banned


def test_darija_pack_bans_stiff_msa() -> None:
    assert any("يمكنني" in phrase for phrase in DARIJA.banned)


def test_router_pack_matches_language(router: LanguageRouter) -> None:
    assert router.pack("en-GB") is EN_GB
    assert router.pack("ar-MA") is DARIJA
