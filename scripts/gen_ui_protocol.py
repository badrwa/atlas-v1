#!/usr/bin/env python3
"""Generate `orb/protocol.ts` from the pydantic models — never hand-edited.

The orb is JavaScript; the protocol is Python.  Generation is the only way those
two can be trusted to agree, so this script is the *only* writer of that file and
`atlas ui protocol --check` fails the build when they drift.

    python scripts/gen_ui_protocol.py            # write the file
    python scripts/gen_ui_protocol.py --check    # exit 1 when it is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("atlas-core", "atlas-ui"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))

from atlas_ui.protocol import javascript, typescript  # noqa: E402

ORB = ROOT / "packages" / "atlas-ui" / "src" / "atlas_ui" / "orb"
#: Both artifacts come from the same models: `.ts` for editors, `.js` for the
#: browser (WebView2 has no build step, on purpose).
TARGETS: tuple[tuple[Path, str], ...] = (
    (ORB / "protocol.ts", "typescript"),
    (ORB / "protocol.js", "javascript"),
)


def render(kind: str) -> str:
    return (typescript() if kind == "typescript" else javascript()) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    parser.add_argument("--out", default="", help="write one file here instead of the defaults")
    args = parser.parse_args(argv)

    if args.out:
        target = Path(args.out)
        generated = render("javascript" if target.suffix == ".js" else "typescript")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(generated, encoding="utf-8")
        print(f"✓ wrote {target} ({len(generated.splitlines())} lines)")
        return 0

    status = 0
    for target, kind in TARGETS:
        generated = render(kind)
        if args.check:
            if not target.is_file():
                print(f"✗ {target.relative_to(ROOT)} is missing — run scripts/gen_ui_protocol.py")
                status = 1
            elif target.read_text(encoding="utf-8") != generated:
                print(f"✗ {target.relative_to(ROOT)} is stale — run scripts/gen_ui_protocol.py")
                status = 1
            else:
                print(f"✓ {target.name} matches the pydantic models")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(generated, encoding="utf-8")
        print(f"✓ wrote {target.relative_to(ROOT)} ({len(generated.splitlines())} lines)")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
