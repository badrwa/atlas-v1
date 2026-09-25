#!/usr/bin/env python3
"""Measure the mouth on *this* machine: time-to-first-audio and realtime factor.

The L3 gate is a number a user feels: a normal answer starts speaking in under
three seconds, and a cached phrase plays in under 200 ms.  This script measures
both — the first pass is a cache miss, the second pass is the same sentences
again — and prints PASS/FAIL against the gate so nobody has to remember the
thresholds.

    python scripts/bench_tts.py                          # every voice that exists
    python scripts/bench_tts.py --engine piper_arabic    # one of them
    python scripts/bench_tts.py --cold                   # fresh process per sentence
    python scripts/bench_tts.py --no-cache --json        # raw numbers, no disk

`--cold` re-runs this same file as a subprocess per sentence, which is the only
honest way to include model load: a warm loop hides the 300–800 ms of ONNX
session creation, and that cost lands on the user's *first* sentence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "atlas-audio" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "atlas-core" / "src"))

from atlas_audio.cache import TtsCache  # noqa: E402
from atlas_audio.speech import SentenceStreamer  # noqa: E402
from atlas_audio.tts import build_voice_chain, engine_ready  # noqa: E402
from atlas_core.config import load_config  # noqa: E402
from atlas_core.timings import percentile  # noqa: E402

#: The gate, from LEVEL-03.
FIRST_AUDIO_BUDGET_MS = 3000.0
CACHED_BUDGET_MS = 200.0

DARIJA = [
    "Salam, ana Atlas.",
    "Chno ljaw dyalek lyoum?",
    "Safi, daba nkemmel lkhedma.",
    "Bghiti n3awd lik hadchi?",
    "Lweqt daba rah tnach w nusf.",
]

ENGLISH = [
    "Hello, this is Atlas.",
    "How is your day going so far?",
    "Right, let me finish that for you.",
    "Would you like me to repeat that?",
    "It is half past twelve right now.",
]


async def tokens(text: str) -> AsyncIterator[str]:
    """A token stream, so the measurement includes the sentence pipeline."""
    for word in text.split(" "):
        yield word + " "


async def measure(streamer: SentenceStreamer, text: str, *, language: str) -> dict[str, float]:
    spoken = [s async for s in streamer.stream(tokens(text), language=language, cap=False)]
    return {
        "first_audio_ms": streamer.first_audio_ms,
        "audio_s": sum(len(s.pcm) / 2 / max(1, s.sample_rate) for s in spoken),
        "synth_ms": sum(s.synth_ms for s in spoken),
        "audible": 1.0 if any(s.audible for s in spoken) else 0.0,
        "cached": 1.0 if spoken and spoken[0].cached else 0.0,
    }


async def run_pass(
    streamer: SentenceStreamer, sentences: list[str], *, language: str
) -> list[dict[str, float]]:
    return [await measure(streamer, text, language=language) for text in sentences]


def run_cold(sentences: list[str], args: argparse.Namespace) -> list[dict[str, float]]:
    """One fresh interpreter per sentence — the honest first-sentence number."""
    rows: list[dict[str, float]] = []
    for text in sentences:
        argv = [sys.executable, __file__, "--_one", text, "--language", args.language,
                "--cache", args.cache]
        if args.no_cache:
            argv.append("--no-cache")
        if args.engine:
            argv += ["--engine", args.engine]
        if args.config:
            argv += ["--config", args.config]
        result = subprocess.run(argv, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            print(f"  cold run failed: {result.stderr.strip().splitlines()[-1:]}", file=sys.stderr)
            continue
        rows.append(json.loads(result.stdout))
    return rows


def summarise(name: str, phase: str, rows: list[dict[str, float]]) -> dict[str, object]:
    if not rows:
        return {"engine": name, "phase": phase, "n": 0}
    first = [row["first_audio_ms"] for row in rows]
    audio = sum(row["audio_s"] for row in rows)
    compute = sum(row["synth_ms"] for row in rows) / 1000
    return {
        "engine": name,
        "phase": phase,
        "n": len(rows),
        "audible": int(sum(row["audible"] for row in rows)),
        "cache_hits": int(sum(row["cached"] for row in rows)),
        "first_audio_p50_ms": round(statistics.median(first), 1),
        "first_audio_p95_ms": round(percentile(first, 95), 1),
        "first_audio_max_ms": round(max(first), 1),
        "realtime_factor": round(audio / compute, 2) if compute else 0.0,
        "audio_s": round(audio, 2),
    }


def budget_for(phase: str) -> float:
    return CACHED_BUDGET_MS if phase == "cached" else FIRST_AUDIO_BUDGET_MS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", help="darija_tts_sidecar | piper | piper_arabic | sapi")
    parser.add_argument("--language", choices=["ar-MA", "en-GB"], default="ar-MA")
    parser.add_argument("--config", help="config.toml to read voices from")
    parser.add_argument("--cache", default="data/bench-tts-cache.sqlite3", help="scratch cache")
    parser.add_argument("--no-cache", action="store_true", help="bypass the cache entirely")
    parser.add_argument("--cold", action="store_true", help="one fresh process per sentence")
    parser.add_argument("--json", action="store_true", help="machine-readable output only")
    parser.add_argument("--out", default="", help="also write the summary JSON here")
    parser.add_argument("--_one", help=argparse.SUPPRESS)  # internal: the --cold worker
    args = parser.parse_args(argv)

    config_path = Path(args.config or "config.toml")
    config = load_config(args.config) if config_path.exists() else None
    sentences = DARIJA if args.language == "ar-MA" else ENGLISH

    if args._one is not None:
        chain = build_voice_chain(config, language=args.language, engine=args.engine)
        cache = None if args.no_cache else TtsCache(args.cache)
        streamer = SentenceStreamer(chain, cache=cache)
        print(json.dumps(asyncio.run(measure(streamer, args._one, language=args.language))))
        return 0

    chain = build_voice_chain(config, language=args.language, engine=args.engine)
    ready = [(engine, engine_ready(engine, config)) for engine in chain]

    if not args.json:
        print("ATLAS TTS bench")
        print(f"  {len(sentences)} sentences × 2 passes · language {args.language} · "
              f"{'cold process per sentence' if args.cold else 'warm loop'}")
        print(f"  gate: first audio < {FIRST_AUDIO_BUDGET_MS / 1000:.0f} s · cached < {CACHED_BUDGET_MS:.0f} ms")
        print()
        for engine, is_ready in ready:
            state = "ready" if is_ready else (getattr(engine, "load_error", "") or "not ready")
            print(f"  {engine.name:<28} {'✓' if is_ready else '·'} {state}")
        print()

    summaries: list[dict[str, object]] = []
    for engine, is_ready in ready:
        if not is_ready:
            continue
        if args.cold:
            fresh = run_cold(sentences, args)
            cached = run_cold(sentences, args)  # the scratch cache now has them
        else:
            cache = None if args.no_cache else TtsCache(args.cache)
            streamer = SentenceStreamer([engine], cache=cache)
            fresh = asyncio.run(run_pass(streamer, sentences, language=args.language))
            cached = asyncio.run(run_pass(streamer, sentences, language=args.language))
        summaries.append(summarise(engine.name, "first", fresh))
        if not args.no_cache:
            summaries.append(summarise(engine.name, "cached", cached))

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print(f"  {'engine':<28} {'pass':<8} {'n':>3} {'p50':>9} {'p95':>9} {'RTF':>7} cache")
        for row in summaries:
            if not row.get("n"):
                continue
            mark = "PASS" if float(row["first_audio_p50_ms"]) <= budget_for(str(row["phase"])) else "FAIL"
            print(
                f"  {row['engine']!s:<28} {row['phase']:<8} {row['n']:>3} "
                f"{row['first_audio_p50_ms']:>8}ms {row['first_audio_p95_ms']:>8}ms "
                f"{row['realtime_factor']:>6}x {row['cache_hits']}/{row['n']} {mark}"
            )

    usable = [row for row in summaries if row.get("audible")]
    if not usable:
        if not args.json:
            print()
            print("  no engine produced audio — nothing to gate")
            print("  → install a voice: pip install 'atlas-audio[local]' + a model in models/tts,")
            print("    or start the Darija sidecar (see vendor/darija-tts/README.md)")
        return 1

    if args.out:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        if not args.json:
            print(f"\n  written: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
