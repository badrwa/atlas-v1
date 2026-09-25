"""What happens between raw ASR and the brain.

Darija and English are the two languages Atlas must get right, and the
post-processor is where a transcript is decided to be one or the other.  It is
also where the lexicon lives — the honest answer to a community Darija model's
error rate is to teach Atlas the words it keeps getting wrong.
"""

from __future__ import annotations

import pytest

from atlas_audio.postprocess import (
    DARIJA_MARKERS,
    AsrPostProcessor,
    detect_language,
    is_arabic,
    is_latin,
    learn_correction,
    load_lexicon,
    normalise_for_match,
    number_to_words,
    save_lexicon,
    strip_diacritics,
)


# ── script detection ─────────────────────────────────────────────────
def test_script_helpers():
    assert is_arabic("سلام") and not is_arabic("salam")
    assert is_latin("salam") and not is_latin("سلام")
    assert strip_diacritics("السَّلَامُ") == "السلام"
    assert normalise_for_match("إلى  نفسي") == normalise_for_match("الى نفسي")


def test_darija_and_english_are_told_apart():
    assert detect_language("شنو كاين فالجو اليوم؟") == "ar-MA"
    assert detect_language("what is the weather like today?") == "en-GB"


def test_latin_darija_is_still_darija():
    """The whole reason the marker table exists: Darija typed in Latin letters."""
    for text in ("salam, kifach labas", "bghit n3ref ljaw", "wach kayn chi haja"):
        assert detect_language(text) == "ar-MA"
    assert "salam" in DARIJA_MARKERS


def test_plain_english_is_not_stolen_by_the_marker_table():
    assert detect_language("please open the notes about the budget") == "en-GB"
    assert detect_language("") == "ar-MA"  # the default, from the plan
    assert detect_language("", default="en-GB") == "en-GB"


def test_a_single_arabic_word_flips_the_sentence():
    assert detect_language("ok do it, ولكن mzyan") == "ar-MA"


# ── numbers ──────────────────────────────────────────────────────────
def test_numbers_are_spoken_the_way_people_say_them():
    assert number_to_words(3, "ar-MA") == "tlata"
    assert number_to_words(7, "ar-MA") == "sb3a"
    assert number_to_words(7, "en-GB") == "seven"
    assert number_to_words(0, "en-GB") == "zero"
    assert number_to_words(42, "en-GB") == "forty two"


def test_big_numbers_stay_digits():
    """Nobody wants 20 000 spelled out, and the brain reads digits fine."""
    assert number_to_words(20_000, "ar-MA") == "20000"


def test_digits_inside_a_transcript_are_replaced_only_when_standalone():
    processor = AsrPostProcessor()
    assert processor.normalise_numbers_in("daba 3ndi 3 dqayq", "ar-MA") == "daba 3ndi tlata dqayq"
    assert processor.normalise_numbers_in("route 66 is closed", "en-GB") == "route sixty six is closed"


def test_number_normalisation_can_be_switched_off():
    processor = AsrPostProcessor(normalise_numbers=False)
    assert processor.process("3 dqayq", confidence=0.9) == "3 dqayq"
    assert AsrPostProcessor().process("3 dqayq", confidence=0.9) == "tlata dqayq"


# ── the processor ────────────────────────────────────────────────────
def test_low_confidence_gets_a_second_opinion_not_a_silent_drop():
    processor = AsrPostProcessor(minimum_confidence=0.35)
    assert processor.process("salam", confidence=0.9) == "salam"
    assert processor.process("salam", confidence=0.2) == "salam"  # words kept
    assert processor.confidence_band(0.9) == "high"
    assert processor.confidence_band(0.5) == "medium"
    assert processor.confidence_band(0.1) == "low"
    assert processor.needs_readback(0.1) is True
    assert processor.needs_readback(0.9) is False


def test_whitespace_is_normalised_but_words_are_not_invented():
    processor = AsrPostProcessor()
    assert processor.process("  salam   labas  ", confidence=0.9) == "salam labas"
    assert processor.process("", confidence=0.9) == ""


def test_the_lexicon_fixes_whole_words_only():
    processor = AsrPostProcessor({"zayd": "Zaid", "obsidian": "Obsidian"})
    assert processor.process("salam zayd, open obsidian", confidence=0.9) == "salam Zaid, open Obsidian"
    # a substring must not be rewritten
    assert processor.process("zaydoun is a name", confidence=0.9) == "zaydoun is a name"


def test_the_lexicon_feeds_cloud_hotwords():
    processor = AsrPostProcessor({"zayd": "Zaid", "darija": "Darija"})
    prompt = processor.lexicon_prompt()
    assert "Zaid" in prompt and "Darija" in prompt
    assert processor.lexicon_prompt(limit=1).count(",") == 0


# ── the lexicon file ─────────────────────────────────────────────────
def test_lexicon_roundtrip_through_a_markdown_file(tmp_path):
    path = tmp_path / "lexicon.md"
    save_lexicon({"zayd": "Zaid", "wrdi": "word"}, path)
    assert load_lexicon(path) == {"zayd": "Zaid", "wrdi": "word"}
    assert "misheard = correct" in path.read_text(encoding="utf-8")  # it documents itself


def test_learning_a_correction_persists_immediately(tmp_path):
    path = tmp_path / "lexicon.md"
    entries = learn_correction(path, "Atlas", "Atlas")
    assert entries["Atlas"] == "Atlas"
    assert learn_correction(path, "3ndi", "3ndi")["3ndi"] == "3ndi"
    assert load_lexicon(path) == {"Atlas": "Atlas", "3ndi": "3ndi"}
    # learned entries are in the file, and the header did not become an entry
    assert "3ndi" in path.read_text(encoding="utf-8")


def test_a_missing_lexicon_is_empty_not_fatal(tmp_path):
    assert load_lexicon(tmp_path / "nothing.md") == {}


def test_a_corrupt_lexicon_line_is_skipped(tmp_path):
    path = tmp_path / "lexicon.md"
    path.write_text("# Atlas lexicon\n\nzayd = Zaid\nthis line has no separator\n# comment\n", encoding="utf-8")
    assert load_lexicon(path) == {"zayd": "Zaid"}


def test_the_processor_can_be_built_from_config(tmp_path):
    from atlas_core.config import AppConfig

    config = AppConfig.model_validate({"app": {"language": "ar-MA"}})
    processor = AsrPostProcessor.from_config(config, lexicon_path=tmp_path / "lex.md")
    assert processor.language == "ar-MA"
    assert processor.lexicon == {}
    processor.lexicon["zayd"] = "Zaid"
    assert processor.process("zayd", confidence=0.9) == "Zaid"


@pytest.mark.parametrize("text", ["salam", "labas", "3afak", "mzyan", "wakha"])
def test_common_darija_words_are_recognised(text: str):
    assert detect_language(text) == "ar-MA"
