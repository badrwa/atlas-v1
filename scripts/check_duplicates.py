#!/usr/bin/env python
"""Duplicate-code gate (R5: zero duplicated code).

A real clone detector, not a line-diff: it tokenises each module, replaces
literals with placeholders (so `get("a")` and `get("b")` match), then looks for
identical token windows shared by two places in the tree.

Why not copy-paste a tool: this repo has no Node toolchain and pylint's R0801
is line-based and famously noisy. ~80 lines of `tokenize` gets an honest gate
that CI can run in under a second.

    python scripts/check_duplicates.py            # check src trees
    python scripts/check_duplicates.py --window 60

Exit code 1 when duplication is found. Mark an intentional repeat with a
`# dup-ok: reason` comment on any line of the window.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tokenize
from collections import defaultdict
from pathlib import Path

DEFAULT_ROOTS = ["packages", "apps"]
DEFAULT_WINDOW = 45  # significant tokens — a block of real logic, not two similar lines
IGNORED_NAMES = {"__init__.py", "conf.py"}
MIN_NAMES = 12  # a real block of logic names at least this many things


class Token:
    __slots__ = ("kind", "line", "path", "text")

    def __init__(self, kind: str, text: str, line: int, path: Path) -> None:
        self.kind = kind
        self.text = text
        self.line = line
        self.path = path


def strip_literals(tokens: list[Token]) -> list[Token]:
    """Drop string/number literals before matching.

    `__all__ = ["a", "b", ...]`, keyword lists and lookup tables are data: two
    long lists of strings always look alike, and matching them against each
    other drowns out the findings that matter.  What is left is the *shape* of
    the code — names, operators, calls — which is what a copy-paste shares.
    """
    kept = [token for token in tokens if token.kind not in {"STRING", "NUMBER"}]
    # `{"a", "b", "c"}` is data: once the literals are gone, only the punctuation
    # separates them.  Collapsing repeated punctuation keeps a wall of data from
    # masquerading as a block of logic.
    collapsed: list[Token] = []
    for token in kept:
        if collapsed and token.kind == "OP" and collapsed[-1].text == token.text:
            continue
        collapsed.append(token)
    return collapsed


def significant_tokens(path: Path) -> list[Token] | None:
    """Token stream with comments, layout and docstrings removed, literals blanked."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None

    tokens: list[Token] = []
    docstring_pending = True
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                              tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
                # Keep a marker for explicit opt-outs, then drop the comment itself.
                if token.type == tokenize.COMMENT and "dup-ok" in token.string:
                    tokens.append(Token("OP", "<dup-ok>", token.start[0], path))
                continue
            if token.type == tokenize.STRING:
                if docstring_pending and token.string.lstrip("rbfu").startswith(('"""', "'''")):
                    continue
                docstring_pending = False
                tokens.append(Token("STRING", "<str>", token.start[0], path))
                continue
            if token.type == tokenize.NUMBER:
                tokens.append(Token("NUMBER", "<num>", token.start[0], path))
                continue
            if token.type == tokenize.NAME:
                docstring_pending = False
                if token.string in {"async", "await"}:
                    continue  # noise that hides real clones
                tokens.append(Token("NAME", token.string, token.start[0], path))
                continue
            docstring_pending = False
            tokens.append(Token("OP", token.string, token.start[0], path))
    except (tokenize.TokenError, IndentationError):
        return None
    return strip_literals(tokens)


def python_files(roots: list[str]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        base = Path(root)
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            parts = path.parts
            if "tests" in parts or "__pycache__" in parts or path.name in IGNORED_NAMES:
                continue
            if "src" not in parts:  # only shipped code, never tooling/templates
                continue
            found.append(path)
    return found


def find_clones(files: list[Path], window: int) -> dict[str, list[Token]]:
    index: dict[str, list[Token]] = defaultdict(list)
    for path in files:
        tokens = significant_tokens(path)
        if tokens is None or len(tokens) < window:
            continue
        for start in range(len(tokens) - window + 1):
            chunk = tokens[start : start + window]
            if any(t.text == "<dup-ok>" for t in chunk):
                continue
            if sum(1 for t in chunk if t.kind == "NAME") < MIN_NAMES:
                continue  # not enough logic in this block to call it a clone
            digest = hashlib.sha1(
                "\x1f".join(f"{t.kind}:{t.text}" for t in chunk).encode()
            ).hexdigest()
            index[digest].append(chunk[0])
    return {digest: starts for digest, starts in index.items() if len(starts) > 1}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Detect duplicated code blocks.")
    parser.add_argument("roots", nargs="*", default=DEFAULT_ROOTS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                        help=f"matching block size in tokens (default {DEFAULT_WINDOW})")
    args = parser.parse_args(argv)

    files = python_files(args.roots or DEFAULT_ROOTS)
    clones = find_clones(files, args.window)

    if not clones:
        print(f"✓ no duplicated blocks (window={args.window} tokens, {len(files)} modules)")
        return 0

    # A clone of N tokens produces N overlapping windows; report each block once,
    # anchored at its first occurrence and at the first line of each repeat.
    seen: list[tuple[Path, int]] = []
    blocks: list[list[tuple[Path, int]]] = []
    for starts in clones.values():
        locations = sorted(
            ((s.path, s.line) for s in starts), key=lambda pair: (str(pair[0]), pair[1])
        )
        if any(
            path == path_seen and abs(line - line_seen) <= args.window
            for path, line in locations
            for path_seen, line_seen in seen
        ):
            continue  # the same copy-paste, reported from an earlier window
        seen.extend(locations)
        blocks.append(locations)

    print(f"✗ {len(blocks)} duplicated block(s), window={args.window} tokens:\n")
    for locations in blocks:
        print("  " + "  ←→  ".join(f"{path}:{line}" for path, line in locations))
    print("\nShare the code (base class, helper, or data) instead of copying it.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
