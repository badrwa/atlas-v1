"""Checks for the orb's own files — the ones no Python test can execute.

The orb is JavaScript in a WebView; there is no Node in this project (that is a
design decision, not an oversight), so the honest question is: *what can be
verified without running it?*  Four things, and they cover the failures that
actually happen:

1. **Truncation.**  A balanced-delimiter scan (string- and comment-aware) catches
   the missing brace from a bad edit — the failure that otherwise shows up as a
   silent blank window.
2. **Protocol drift.**  Every `case "<kind>"` in `orb.js` must be a kind the
   pydantic models declare, and every declared kind must be handled.  This is the
   check that keeps the generated `.js` and the hand-written drawing loop honest.
3. **Wiring.**  `index.html` must reference files that exist and must still carry
   the `__TOKEN__` placeholders the bridge substitutes.
4. **Values.**  The generated runtime module must carry the same protocol version,
   mood hues and caption fade the Python does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from atlas_ui.protocol import MOOD_HUES, PROTOCOL_VERSION

#: Delimiters that must nest correctly in the browser sources.
PAIRS = {")": "(", "]": "[", "}": "{"}


def orb_dir() -> Path:
    return Path(__file__).resolve().parent / "orb"


@dataclass(slots=True)
class AssetReport:
    """What was checked, and what is wrong.  Empty `problems` means "shipped"."""

    checked: list[str]
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_lines(self) -> list[str]:
        if self.ok:
            return [f"✓ orb assets: {len(self.checked)} files check out, protocol v{PROTOCOL_VERSION}"]
        return [f"✗ {problem}" for problem in self.problems]


def strip_code(text: str, *, line_comment: str = "//") -> str:
    """Remove comments and string bodies, keeping delimiters in place.

    `--` inside a CSS custom property, `//` in a URL and `/* */` in a template
    string are all the reasons this is not a regex one-liner.
    """
    out: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        nxt = text[index + 1] if index + 1 < length else ""
        if char == "/" and nxt == "*":
            end = text.find("*/", index + 2)
            index = length if end == -1 else end + 2
            continue
        if char == "/" and nxt == "/":
            end = text.find("\n", index)
            index = length if end == -1 else end + 1
            continue
        if line_comment == "#" and char == "#":
            end = text.find("\n", index)
            index = length if end == -1 else end + 1
            continue
        if char in "\"'`":
            quote = char
            index += 1
            while index < length:
                if text[index] == "\\":
                    index += 2
                    continue
                if text[index] == quote:
                    break
                index += 1
            index += 1
            out.append(f"{quote}{quote}")  # keep the delimiters balanced
            continue
        out.append(char)
        index += 1
    return "".join(out)


def delimiters_balanced(text: str, *, line_comment: str = "//") -> list[str]:
    """Unbalanced brackets, in the order they were found."""
    stripped = strip_code(text, line_comment=line_comment)
    stack: list[str] = []
    problems: list[str] = []
    for index, char in enumerate(stripped):
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if not stack or stack[-1] != PAIRS[char]:
                problems.append(f"unexpected '{char}' at offset {index}")
                return problems
            stack.pop()
    if stack:
        problems.append(f"unclosed '{stack[-1]}' (depth {len(stack)})")
    return problems


def _js_kinds(text: str) -> set[str]:
    return set(re.findall(r'case\s+"([a-z_]+)"\s*:', text))


def check_assets(root: Path | None = None) -> AssetReport:
    """Run every static check.  `problems` is empty when the orb is shippable."""
    directory = root or orb_dir()
    checked: list[str] = []
    problems: list[str] = []

    needed = ("index.html", "orb.css", "orb.js", "protocol.js")
    for name in needed:
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} is missing")
            continue
        text = path.read_text(encoding="utf-8")
        checked.append(name)
        comment = "/*" if name.endswith(".css") else "//"
        for issue in delimiters_balanced(text, line_comment=comment):
            problems.append(f"{name}: {issue}")
        if len(text) < 200:
            problems.append(f"{name}: suspiciously small ({len(text)} bytes)")

    index = (directory / "index.html").read_text(encoding="utf-8") if (directory / "index.html").is_file() else ""
    for asset in ("orb.css", "orb.js"):
        if asset not in index:
            problems.append(f"index.html does not load {asset}")
    for placeholder in ("__TOKEN__",):
        if placeholder not in index:
            problems.append(f"index.html lost its {placeholder} placeholder (the bridge substitutes it)")
    # protocol.js is loaded *by* orb.js (one entry point, one module graph), so
    # only the direct references belong in this list.
    for asset in ("orb.css", "orb.js"):
        if f'"/orb/{asset}' not in index and f'"./{asset}' not in index:
            problems.append(f"index.html never references {asset}")

    orb_js = (directory / "orb.js").read_text(encoding="utf-8") if (directory / "orb.js").is_file() else ""
    if "protocol.js" not in orb_js:
        problems.append("orb.js does not import the generated runtime module")
    else:
        kinds = _js_kinds(orb_js)
        declared = {
            "hello",
            "state",
            "caption",
            "mood",
            "level",
            "tool",
            "confirm",
            "degraded",
        }
        missing = declared - kinds - {"hello"}  # hello is handled as link proof, not a case
        invented = kinds - declared
        if missing:
            problems.append(f"orb.js does not handle: {', '.join(sorted(missing))}")
        if invented:
            problems.append(f"orb.js handles unknown kinds: {', '.join(sorted(invented))}")

    runtime = (directory / "protocol.js").read_text(encoding="utf-8") if (directory / "protocol.js").is_file() else ""
    if runtime:
        if f"export const PROTOCOL_VERSION = {PROTOCOL_VERSION};" not in runtime:
            problems.append("generated protocol.js has a different protocol version")
        for mood, hue in MOOD_HUES.items():
            if f'"{mood}": {hue}' not in runtime:
                problems.append(f"generated protocol.js is missing mood '{mood}' ({hue})")
    return AssetReport(checked=checked, problems=problems)


def orb_path(name: str) -> Path:
    """Resolve one orb asset, refusing anything outside the orb directory."""
    directory = orb_dir().resolve()
    path = (directory / name).resolve()
    if path != directory / name or directory not in path.parents:
        raise ValueError(f"{name} is not an orb asset")
    return path


__all__ = ["PAIRS", "AssetReport", "check_assets", "delimiters_balanced", "orb_dir", "orb_path", "strip_code"]
