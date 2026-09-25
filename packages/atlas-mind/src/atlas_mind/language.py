"""Language routing: Darija by default, British English on request.

The router decides the language; the model is *told* what it is.  Letting the
LLM guess produces language drift mid-conversation, which is exactly what makes
an assistant feel fake.

Evidence, strongest first:
    1. an explicit switch command ("b l'ingliziya", "speak English")
    2. script (Arabic characters)
    3. Darija/Arabizi vocabulary markers
    4. sticky session language (what we were just speaking)
"""

from __future__ import annotations

import re

from atlas_core.contracts import LanguageTag
from atlas_mind.darija import DarijaNormalizer, arabic_ratio, has_arabic
from atlas_mind.dialects import DEFAULT_PACK, DialectPack, pack_for


class LanguageRouter:
    """Decides which language Atlas answers in, per utterance."""

    # Explicit commands, checked in order. Keys are regex fragments.
    SWITCH_TO_DARIJA = (
        r"b\s*darija", r"b\s*darijaa", r"darija", r"b\s*l3arbiya", r"b\s*lmaghribiya",
        r"kellmeni\s*b\s*darija", r"hdar\s*m3aya\s*b\s*darija", r"be\s*darija",
    )
    SWITCH_TO_ENGLISH = (
        r"speak\s+english", r"in\s+english", r"b\s*l?ingliziya", r"b\s*l?inglizya",
        r"b\s*english", r"english\s+please", r"switch\s+to\s+english", r"b\s*l'?anglais",
    )

    def __init__(
        self,
        default: LanguageTag = "ar-MA",
        secondary: LanguageTag = "en-GB",
        *,
        sticky: bool = True,
    ) -> None:
        self.default = default
        self.secondary = secondary
        self.sticky = sticky
        self.current: LanguageTag = default
        self._normalizer = DarijaNormalizer()
        self._darija_re = re.compile("|".join(self.SWITCH_TO_DARIJA), re.IGNORECASE)
        self._english_re = re.compile("|".join(self.SWITCH_TO_ENGLISH), re.IGNORECASE)

    # ── public API ───────────────────────────────────────────────────
    def route(self, text: str, *, hint: LanguageTag | None = None) -> LanguageTag:
        """Decide the reply language for one utterance."""
        if hint in ("ar-MA", "en-GB"):
            self.current = hint
            return hint

        if switched := self.detect_switch(text):
            self.current = switched
            return switched

        detected = self.detect(text)
        if detected == "unknown" and self.sticky:
            return self.current
        self.current = detected if detected != "unknown" else self.current
        return self.current

    def detect_switch(self, text: str) -> LanguageTag | None:
        """Explicit request to change language (checked before anything else)."""
        if self._english_re.search(text):
            return self.secondary
        if self._darija_re.search(text):
            return self.default
        return None

    def detect(self, text: str) -> LanguageTag:
        """Best-effort detection from the text alone."""
        stripped = text.strip()
        if not stripped:
            return "unknown"
        if has_arabic(stripped) or arabic_ratio(stripped) > 0.2:
            return self.default
        if self._normalizer.is_darija(stripped):
            return self.default
        if re.search(r"[a-z]", stripped, re.IGNORECASE):
            # Latin script with no Darija markers → the secondary language.
            return self.secondary
        return "unknown"

    def pack(self, language: LanguageTag | str | None = None) -> DialectPack:
        """The dialect data for a language (Darija when unsure)."""
        return pack_for(str(language or self.current))

    @property
    def default_pack(self) -> DialectPack:
        return pack_for(self.default) or DEFAULT_PACK

    def reset(self) -> None:
        self.current = self.default

    def snapshot(self) -> dict[str, str]:
        return {"current": self.current, "default": self.default, "secondary": self.secondary}


__all__ = ["LanguageRouter"]
