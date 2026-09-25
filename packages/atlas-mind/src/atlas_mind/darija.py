"""Darija text handling.

Darija has no standard orthography: the same sentence can be written in Arabic
script, in Arabizi (Latin with digits: 3=ع, 7=ح, 9=ق), or a mixture.  So Atlas
never tries to "fix" the user's spelling.  It keeps two views of every string:

* **canonical** (for matching/recall): diacritics stripped, letters unified,
  Arabizi mapped, repeats collapsed.
* **display/tts** (what the user sees, what the voice speaks): untouched.

Plus a *user lexicon* (`50_Atlas/darija-lexicon.md`) that corrects recurring ASR
mistakes — names, apps, places.  That file is how accuracy improves over months
without retraining anything.
"""

from __future__ import annotations

import re

ARABIC_RANGE = re.compile(r"[\u0600-\u06ff]")
DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u0640]")
REPEATS = re.compile(r"(.)\1+")
VOWELS = frozenset("aeiou")


def _collapse(match: re.Match[str]) -> str:
    """Elongation is noise: vowel runs → 1, consonant runs → 2 (so 'll' survives)."""
    char = match.group(1)
    return char if char.lower() in VOWELS else char * 2

# Arabizi → Arabic, used *only* for the canonical/matching view.
ARABIZI_MAP: dict[str, str] = {
    "3": "ع", "7": "ح", "9": "ق", "2": "ء", "5": "خ", "8": "غ",
    "kh": "خ", "gh": "غ", "ch": "ش", "sh": "ش", "th": "ث", "dh": "ذ",
    "ou": "و", "aa": "ا", "ee": "ي",
}

# Letter unification for matching (never applied to what we show or speak).
UNIFY = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ئ": "ي",
    "ؤ": "و", "ة": "ه", "گ": "ك", "ڤ": "ف", "پ": "ب", "چ": "ش", "ژ": "ج",
})

# Common Darija words — used by the language router as strong evidence.
DARIJA_MARKERS: frozenset[str] = frozenset({
    "wach", "wa9ila", "waxa", "safi", "bezzaf", "mzyan", "mezian", "daba", "dba",
    "kifach", "kifash", "chkoun", "chkoune", "3lash", "3lach", "ash", "chno", "shno",
    "bghit", "bghiti", "dir", "dert", "kayn", "kayna", "ghadi", "ghir", "zid", "sir",
    "3afak", "afak", "mrhba", "labas", "salam", "l9ahwa", "khoya", "sahbi", "sahbia",
    "dyal", "dyali", "dyalk", "nta", "nti", "hna", "hadak", "hadi", "hada",
})


def has_arabic(text: str) -> bool:
    return bool(ARABIC_RANGE.search(text))


def arabic_ratio(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ARABIC_RANGE.match(ch)) / len(letters)


def strip_diacritics(text: str) -> str:
    return DIACRITICS.sub("", text)


def unify_letters(text: str) -> str:
    return text.translate(UNIFY)


def arabizi_to_arabic(text: str) -> str:
    lowered = text.lower()
    for latin, arabic in sorted(ARABIZI_MAP.items(), key=lambda kv: -len(kv[0])):
        lowered = lowered.replace(latin, arabic)
    return lowered


def collapse_repeats(text: str) -> str:
    """'bezzzaaaf' → 'bezzaf' while 'allah' and 'mochkila' keep their shape."""
    return REPEATS.sub(_collapse, text)


class DarijaNormalizer:
    """Two views of one string: canonical (match) and display (show/speak)."""

    def normalize_for_match(self, text: str) -> str:
        """Order matters: strip → collapse elongation → Arabizi digits → unify letters."""
        step = strip_diacritics(text).lower()
        step = collapse_repeats(step)
        step = arabizi_to_arabic(step)
        return collapse_repeats(unify_letters(step))

    def normalize_for_tts(self, text: str) -> str:
        """Speech-ready: strip emoji/markdown, keep the user's own orthography."""
        cleaned = re.sub(r"[*_`#>\[\]]", " ", text)
        cleaned = re.sub(r"[\U0001f300-\U0001faff\u2600-\u27bf]", "", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def is_darija(text: str) -> bool:
        if has_arabic(text):
            return True
        tokens = set(re.findall(r"[a-z0-9']+", text.lower()))
        return len(tokens & DARIJA_MARKERS) >= 1


class Lexicon:
    """User-corrected ASR vocabulary, stored as Markdown in the vault.

    File format (one correction per line):
        - misheard => correct
    """

    HEADER = (
        "# Darija lexicon\n\n"
        "Corrections Atlas applies to ASR output, one per line.\n"
        "Entries look like:  - heard-word  =>  meant-word\n"
    )

    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self.entries: dict[str, str] = dict(entries or {})
        self._normalizer = DarijaNormalizer()

    @classmethod
    def from_text(cls, text: str) -> Lexicon:
        entries: dict[str, str] = {}
        for line in text.splitlines():
            line = line.strip()
            # entries are bullet lines: "- heard  =>  meant"
            if not line.startswith("-"):
                continue
            line = line.lstrip("-").strip()
            if "=>" not in line:
                continue
            wrong, _, right = line.partition("=>")
            if wrong.strip() and right.strip():
                entries[wrong.strip()] = right.strip()
        return cls(entries)

    @classmethod
    def from_file(cls, path: str | object) -> Lexicon:
        from pathlib import Path

        file = Path(str(path))
        if not file.exists():
            return cls()
        return cls.from_text(file.read_text(encoding="utf-8"))

    def to_text(self) -> str:
        lines = [self.HEADER]
        lines += [f"- {wrong} => {right}" for wrong, right in sorted(self.entries.items())]
        return "\n".join(lines) + "\n"

    def apply(self, text: str) -> str:
        """Literal, case-insensitive replacement of heard→meant words.

        Deliberately literal: any cleverness here would silently rewrite what the
        user actually said.  Arabic-script variants get their own entries.
        """
        if not self.entries:
            return text
        result = text
        for wrong, right in self.entries.items():
            result = re.sub(re.escape(wrong), right, result, flags=re.IGNORECASE)
        return result

    def learn(self, wrong: str, right: str) -> None:
        self.entries[wrong.strip()] = right.strip()

    def as_prompt_hint(self, limit: int = 40) -> str:
        """Hot-word list handed to the ASR provider (names/apps/places)."""
        if not self.entries:
            return ""
        joined = ", ".join(f"{wrong}→{right}" for wrong, right in list(self.entries.items())[:limit])
        return f"Vocabulary hints: {joined}"


__all__ = [
    "ARABIZI_MAP",
    "DARIJA_MARKERS",
    "DarijaNormalizer",
    "Lexicon",
    "arabic_ratio",
    "arabizi_to_arabic",
    "collapse_repeats",
    "has_arabic",
    "strip_diacritics",
    "unify_letters",
]
