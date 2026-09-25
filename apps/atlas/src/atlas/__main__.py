"""Entry point: `python -m atlas <command>`."""

from __future__ import annotations

import sys

from atlas.cli import main

if __name__ == "__main__":
    sys.exit(main())
