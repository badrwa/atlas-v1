#!/usr/bin/env python3
"""Measure every configured provider — run this on your own PC, not in CI.

    python scripts/bench_providers.py            # 3 prompts per provider
    python scripts/bench_providers.py --runs 5

Reports time-to-first-token and total time, which is what actually decides
`mind.provider_order` (plan §4.2).  Uses a fixed prompt so the comparison is
fair, and a Darija prompt because that is Atlas's default language.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "atlas-mind" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "atlas-core" / "src"))

PROMPT = "جاوب بجملة واحدة قصيرة: شنو هي عاصمة المغرب؟"


async def bench(config, runs: int) -> int:
    from atlas_core.contracts import LlmRequest, Message, Role
    from atlas_mind.providers.factory import build_provider

    rows: list[tuple[str, float, float, float]] = []
    for provider_config in config.ordered_providers():
        provider = build_provider(provider_config)
        ttfts: list[float] = []
        totals: list[float] = []
        preview = ""
        try:
            for _ in range(runs):
                request = LlmRequest(
                    messages=[Message(role=Role.USER, content=PROMPT)],
                    max_output_tokens=48,
                    temperature=0.2,
                )
                started = time.perf_counter()
                first: float | None = None
                text = ""
                async for event in provider.stream(request):
                    if event.kind == "text" and event.text:
                        if first is None:
                            first = (time.perf_counter() - started) * 1000
                        text += event.text
                if first is None:
                    continue
                ttfts.append(first)
                totals.append((time.perf_counter() - started) * 1000)
                preview = (text.strip()[:32] or preview) if text.strip() else preview
        except Exception as exc:
            print(f"  ✗ {provider.name}: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        finally:
            close = getattr(provider, "aclose", None)
            if close:
                await close()

        if ttfts:
            rows.append(
                (
                    provider.name,
                    statistics.median(ttfts),
                    statistics.median(totals),
                    float(runs),
                )
            )
            print(f"  ✓ {provider.name:<14} ttft {statistics.median(ttfts):6.0f} ms   {preview}")

    print()
    if not rows:
        print("nothing answered — check .env and run: python -m atlas providers --live")
        return 1
    rows.sort(key=lambda row: row[1])
    print("fastest first (this is your provider_order):")
    for name, ttft, total, _ in rows:
        print(f"  {name:<14} ttft {ttft:6.0f} ms · total {total:6.0f} ms")
    print("\nsuggested config.toml:")
    print("  [mind]")
    print(f'  provider_order = [{", ".join(repr(row[0]) for row in rows)}]')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark configured LLM providers")
    parser.add_argument("--runs", type=int, default=3, help="runs per provider (default 3)")
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()

    from atlas_core.config import ConfigError, load_config

    try:
        config = load_config(Path(args.config))
    except ConfigError as exc:
        print(f"config error: {exc}")
        return 1
    if not config.ordered_providers():
        print("no usable providers — add a key to .env (see .env.example)")
        return 1
    return asyncio.run(bench(config, args.runs))


if __name__ == "__main__":
    raise SystemExit(main())
