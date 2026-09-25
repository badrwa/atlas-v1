"""What happens between "the model heard something" and "Atlas understands you".

Three jobs, in order of how much they matter on this project:

1. **Lexicon corrections.** Darija ASR gets *names, apps and places* wrong, and
   those are exactly the words we already know. The lexicon is a file the owner
   can edit (`50_Atlas/darija-lexicon.md`), and every correction improves every
   later turn. This — not a better model — is the plan's real accuracy strategy.
2. **Script and language detection.** Decides the reply language from what was
   actually transcribed, so a Darija sentence with an English noun inside stays
   Darija.
3. **Spoken-number normalisation.** "2015" and "15 %" are read aloud badly;
   Darija and British English each have their own way of saying them.

Deliberately absent: rewriting the user's words. Normalise for *matching*, keep
the original for display, and never change meaning.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_core.contracts import LanguageTag

log = logging.getLogger(__name__)

ARABIC_RANGE = re.compile(r"[\u0600-\u06ff]")
# A Markdown bullet, stripped as a *prefix* — never with `lstrip`, which ate the
# leading digit of Darija words like "3ndi" and "7na".
BULLET = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
LATIN_RANGE = re.compile(r"[A-Za-z]")
DIACRITICS = re.compile(r"[\u064b-\u0652\u0670\u0640]")

# Darija markers in Latin script — the same evidence the text router uses, kept
# here because audio arrives before any language has been decided.
DARIJA_MARKERS = (
    "wach", "safi", "bezzaf", "mzyan", "daba", "kifach", "chkoun", "3lash",
    "bghit", "kayn", "ghadi", "zid", "wakha", "labas", "salam", "chwiya",
    "3afak", "3ndi", "3ndek", "dyal", "dyali", "walo", "hna", "rak", "rani",
    "chokran", "bslama", "bsah", "m3a", "3lik", "fin", "mnin", "bzaf",
)

# Numbers get spoken, not spelled. Kept small: a full num2words dependency
# would be the 200th package in a project that guards its RAM.
ONES_FR = (
    "zéro", "wahed", "juj", "tlata", "rb3a", "khamsa", "sitta", "sb3a", "tmnya", "ts3a",
)
TENS_FR = {
    10: "3achra", 20: "3achrin", 30: "tlatin", 40: "rb3in", 50: "khamsin",
    60: "sittin", 70: "sb3in", 80: "tmanin", 90: "ts3in", 100: "mya",
}
ENGLISH_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
)
ENGLISH_TEENS = {
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
}
ENGLISH_TENS = {
    20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
    60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety",
}


def is_arabic(text: str) -> bool:
    return bool(ARABIC_RANGE.search(text))


def is_latin(text: str) -> bool:
    return bool(LATIN_RANGE.search(text))


def strip_diacritics(text: str) -> str:
    """Tashkeel and tatweel — present in TTS output, absent in real typing."""
    return DIACRITICS.sub("", text)


def normalise_for_match(text: str) -> str:
    """A comparison key: unify alef/ya/ta-marbuta, strip diacritics, collapse space.

    Used for lexicon matching and deduplication. **Never** used to rewrite what
    the user or the model said.
    """
    cleaned = strip_diacritics(unicodedata.normalize("NFKC", text)).lower()
    cleaned = (
        cleaned.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    cleaned = re.sub(r"[^\w\s\u0600-\u06ff]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def detect_language(text: str, *, default: LanguageTag = "ar-MA") -> LanguageTag:
    """Arabic script → Darija. Latin → English *unless* Darija markers appear.

    This is the audio path's version of the language router: it runs on a
    transcript, where there is no punctuation and no capitalisation to help.
    """
    stripped = text.strip()
    if not stripped:
        return default
    arabic = len(ARABIC_RANGE.findall(stripped))
    latin = len(LATIN_RANGE.findall(stripped))
    lowered = stripped.lower()

    if arabic and arabic >= latin:
        return "ar-MA"
    if arabic and latin:
        # Mixed script — Darija with a Latin word or two inside — is still
        # Darija: the Arabic script settles it, no marker counting needed.
        return "ar-MA"
    if latin:
        hits = sum(1 for marker in DARIJA_MARKERS if marker in lowered)
        return "ar-MA" if hits >= 1 else "en-GB"
    return default


def number_to_words(value: int, language: LanguageTag = "ar-MA") -> str:
    """Speakable form of a small integer (0–999). Larger values stay digits."""
    if value < 0 or value > 999:
        return str(value)
    if language == "en-GB":
        return _english_number(value)
    return _darija_number(value)


def _english_number(value: int) -> str:
    if value < 10:
        return ENGLISH_ONES[value]
    if value < 20:
        return ENGLISH_TEENS[value]
    if value < 100:
        tens, rest = divmod(value, 10)
        return ENGLISH_TENS[tens * 10] + (f" {ENGLISH_ONES[rest]}" if rest else "")
    hundreds, rest = divmod(value, 100)
    prefix = f"{ENGLISH_ONES[hundreds]} hundred"
    return prefix + (f" and {_english_number(rest)}" if rest else "")


def _darija_number(value: int) -> str:
    """Darija numbers Moroccans actually say — French loans included, as they are."""
    if value < 10:
        return ONES_FR[value]
    if value < 20:
        return f"{ONES_FR[value - 10]} w 3achra"
    if value < 100:
        tens, rest = divmod(value, 10)
        base = TENS_FR[tens * 10]
        return base + (f" w {ONES_FR[rest]}" if rest else "")
    hundreds, rest = divmod(value, 100)
    base = "mya" if hundreds == 1 else f"{ONES_FR[hundreds]} mya"
    return base + (f" w {_darija_number(rest)}" if rest else "")


@dataclass
class AsrPostProcessor:
    """Turns raw ASR output into something the brain can be given."""

    lexicon: dict[str, str] = field(default_factory=dict)
    language: LanguageTag = "ar-MA"
    normalise_numbers: bool = True
    minimum_confidence: float = 0.35
    corrections_applied: list[tuple[str, str]] = field(default_factory=list)

    # ── construction ─────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config: Any, *, lexicon_path: str | Path | None = None) -> AsrPostProcessor:
        """Build from `[asr]` — an explicit path wins over the configured one."""
        lexicon: dict[str, str] = {}
        path = lexicon_path or getattr(getattr(config, "asr", None), "lexicon_path", "")
        if path and Path(path).exists():
            lexicon.update(load_lexicon(path))
        return cls(lexicon=lexicon, language=getattr(getattr(config, "app", None), "language", "ar-MA"))

    # ── the pass ─────────────────────────────────────────────────────
    def process(
        self,
        text: str,
        *,
        language: LanguageTag | None = None,
        confidence: float = 0.0,
    ) -> str:
        """Correct, then normalise — in that order, so numbers in names survive."""
        result = text.strip()
        result = self._apply_lexicon(result)
        if self.normalise_numbers:
            result = self.normalise_numbers_in(result, language or self.language)
        return re.sub(r"\s+", " ", result).strip()

    def _apply_lexicon(self, text: str) -> str:
        """Replace known mis-hearings. Whole words only — never inside a word."""
        if not self.lexicon:
            return text
        for wrong, right in self.lexicon.items():
            pattern = re.compile(rf"(?<![\w\u0600-\u06ff]){re.escape(wrong)}(?![\w\u0600-\u06ff])", re.IGNORECASE)
            if pattern.search(text):
                text = pattern.sub(right, text)
                self.corrections_applied.append((wrong, right))
        return text

    def normalise_numbers_in(self, text: str, language: LanguageTag) -> str:
        """Digits → words, for everything that will be spoken aloud."""

        def replace(match: re.Match[str]) -> str:
            raw = match.group(0)
            try:
                value = int(raw.replace(",", ""))
            except ValueError:  # pragma: no cover - regex already guarantees digits
                return raw
            return number_to_words(value, language)

        # Years and long numbers stay as digits: "1999" is shorter than
        # "alf w ts3 mya w ts3in w ts3a" and everyone reads it fine.
        return re.sub(r"\b\d{1,3}\b", replace, text)

    def confidence_band(self, confidence: float) -> str:
        if confidence >= 0.75:
            return "high"
        if confidence >= self.minimum_confidence:
            return "medium"
        return "low"

    def needs_readback(self, confidence: float) -> bool:
        """Low confidence before a destructive action must be read back (L7)."""
        return confidence < self.minimum_confidence

    # ── hotwords for cloud ASR ───────────────────────────────────────
    def lexicon_prompt(self, limit: int = 30) -> str:
        """The `prompt`/hotword string sent to cloud ASR: the words it gets wrong."""
        if not self.lexicon:
            return ""
        rights = list(dict.fromkeys(self.lexicon.values()))
        return ", ".join(rights[:limit])


def load_lexicon(path: str | Path) -> dict[str, str]:
    """Read `wrong = right` lines from a Markdown file.

    A Markdown list is the format because the owner edits it in Obsidian, not in
    a terminal. Headings, prose and commented lines are skipped, so the file can
    explain itself without teaching the parser nonsense entries.
    """
    entries: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return entries
    for line in text.splitlines():
        stripped = BULLET.sub("", line.strip()).strip()
        if not stripped or "=" not in stripped:
            continue
        if stripped[0] in "#<>" or "`" in stripped:  # heading, comment, or example
            continue
        wrong, _, right = stripped.partition("=")
        wrong, right = wrong.strip(), right.strip()
        if wrong and right:
            entries[wrong] = right
    return entries


def save_lexicon(entries: dict[str, str], path: str | Path) -> Path:
    """Write the lexicon back — how a correction survives the next restart."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Darija lexicon",
        "",
        "Words Atlas mis-hears, and what you actually said. One per line,",
        "in the form below — Atlas adds a line here every time you correct it.",
        "",
        "<!-- misheard = correct -->",
        "",
    ]
    lines += [f"- {wrong} = {right}" for wrong, right in sorted(entries.items())]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def learn_correction(path: str | Path, wrong: str, right: str) -> dict[str, str]:
    """Record one correction. This is how accuracy compounds over months."""
    entries = load_lexicon(path)
    if wrong and right and entries.get(wrong) != right:
        entries[wrong] = right
        save_lexicon(entries, path)
        log.info("lexicon_learned wrong=%s right=%s", wrong, right)
    return entries


__all__ = [
    "DARIJA_MARKERS",
    "AsrPostProcessor",
    "detect_language",
    "is_arabic",
    "is_latin",
    "learn_correction",
    "load_lexicon",
    "normalise_for_match",
    "number_to_words",
    "save_lexicon",
    "strip_diacritics",
]
