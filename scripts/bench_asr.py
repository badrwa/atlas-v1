#!/usr/bin/env python
"""Measure ASR latency and accuracy on *your* recordings, on *your* laptop.

The L2 gate asks for two numbers this sandbox cannot produce:

* cloud ASR p50 ≤ 1.2 s (Gemini / Groq — needs a network and a key)
* local Darija p50 ≤ 3.5 s (faster-whisper int8 — needs the model and 8 GB)

So the numbers are measured where they must be: on the machine that has the
microphone.  Point this script at recordings, not at synthetic audio — the
whole point is to catch what a real voice does to a real model.

    # one recording per utterance, named after what was said
    python scripts/bench_asr.py data/recordings/20260925-1400 --truth data/truth.tsv

    # or a capture dump, using the transcripts Atlas itself produced as truth
    python scripts/bench_asr.py data/recordings/20260925-1400 --from-manifest

    # engines
    python scripts/bench_asr.py rec/ --engines cloud,local:darija --repeat 3

Truth is either a TSV of `file<TAB>transcript` or the session manifest's own
transcripts (which is how you measure *stability*: same audio, same text?).

Latency is reported as p50/p95 of wall-clock milliseconds and as a realtime
factor (RTF = seconds of compute per second of audio).  WER is reported per
engine, normalised (diacritics stripped, digits untouched) — a Darija word
written two ways is not an error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "atlas-audio" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "atlas-core" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "atlas-mind" / "src"))


def load_truth(args) -> dict[str, str]:
    """`file → what was actually said`."""
    if args.truth:
        truth: dict[str, str] = {}
        for line in Path(args.truth).read_text(encoding="utf-8").splitlines():
            if not line.strip() or "\t" not in line:
                continue
            name, said = line.split("\t", 1)
            truth[Path(name.strip()).stem] = said.strip()
        return truth
    if args.from_manifest:
        from atlas_audio.capture import load_session

        session = load_session(args.path)
        return {
            Path(turn["pcm_path"]).stem: turn.get("transcript", "") for turn in session.turns
        }
    return {}


def collect_audio(path: Path, truth: dict[str, str]) -> list[tuple[str, bytes]]:
    """WAV files (or the turns of a session dump) as raw 16 kHz PCM."""
    from atlas_audio.capture import WavFile, load_session

    if path.is_dir() and (path / "session.json").exists():
        session = load_session(path)
        files = session.utterances
    elif path.is_dir():
        files = sorted(path.glob("*.wav"))
    else:
        files = [path]

    clips: list[tuple[str, bytes]] = []
    for wav in files:
        name = wav.stem
        if truth and name not in truth:
            continue
        samples = WavFile.read(wav)
        clips.append((name, samples.samples.tobytes()))
    return clips


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate on normalised words — 0 is perfect, 1 is nonsense."""
    from atlas_audio.postprocess import normalise_for_match

    ref = normalise_for_match(reference).split()
    hyp = normalise_for_match(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein over words, the standard definition.
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_word != hyp_word),  # substitution
                )
            )
        previous = current
    return previous[-1] / len(ref)


async def measure(engine_name, recognizer, clips, *, repeat: int, language: str) -> list[dict]:
    """Every clip, `repeat` times — latency *and* stability in one pass."""
    rows: list[dict] = []
    for _ in range(repeat):
        for name, pcm in clips:
            started = time.perf_counter()
            try:
                transcript = await recognizer.transcribe(pcm, language=language)
                ms = (time.perf_counter() - started) * 1000
                rows.append(
                    {
                        "clip": name,
                        "audio_s": len(pcm) / 2 / 16_000,
                        "ms": ms,
                        "text": transcript.text,
                        "engine": transcript.engine or engine_name,
                        "confidence": transcript.confidence,
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "clip": name,
                        "audio_s": len(pcm) / 2 / 16_000,
                        "ms": (time.perf_counter() - started) * 1000,
                        "text": "",
                        "engine": engine_name,
                        "confidence": 0.0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return rows


def summarise(name: str, rows: list[dict], truth: dict[str, str]) -> dict:
    """p50/p95/max latency, realtime factor, and WER against the truth."""
    from atlas_core.timings import percentile

    ok = [row for row in rows if not row["error"]]
    latencies = [row["ms"] for row in ok]
    audio_s = sum(row["audio_s"] for row in ok)
    compute_s = sum(latencies) / 1000
    scores = [wer(truth[row["clip"]], row["text"]) for row in ok if truth.get(row["clip"])]
    return {
        "engine": name,
        "n": len(rows),
        "failed": len(rows) - len(ok),
        "p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "p95_ms": round(percentile(latencies, 95), 1) if latencies else None,
        "max_ms": round(max(latencies), 1) if latencies else None,
        "mean_wer": round(statistics.fmean(scores), 3) if scores else None,
        # RTF: seconds of compute per second of audio. < 1 means faster than
        # realtime, which is the only figure that matters for a live loop.
        "realtime_factor": round(compute_s / audio_s, 3) if audio_s else None,
    }


async def run(args) -> int:
    from atlas_audio import build_recognizers
    from atlas_core.config import load_config

    target = Path(args.path).expanduser()
    if not target.exists():
        print(f"nothing to measure: {target} does not exist")
        print("record something first:  atlas listen --capture-dump")
        return 1

    config = load_config(args.config)
    built = build_recognizers(config)
    truth = load_truth(args)
    clips = collect_audio(target, truth)
    if not clips:
        print(f"no audio found at {args.path} (or none matched the truth file)")
        return 1

    engines = {
        "cloud": built["cloud"],
        "local:darija": built["local"]["local:darija"],
        "local:english": built["local"]["local:english"],
        "local:multilingual": built["local"]["local:multilingual"],
    }
    wanted = [name.strip() for name in args.engines.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in engines]
    if unknown:
        print(f"unknown engine(s): {', '.join(unknown)} — try: {', '.join(engines)}")
        return 1

    total_s = sum(len(pcm) / 2 / 16_000 for _, pcm in clips)
    print(
        f"Atlas ASR bench · {len(clips)} clip(s) · {total_s:.1f}s of audio · "
        f"{args.repeat} pass(es) · language={args.language}"
    )
    print(f"config: mode={config.asr.mode} cloud_audio={config.asr.cloud_audio}")
    print()

    summaries = []
    for name in wanted:
        recognizer = engines[name]
        if name.startswith("local"):
            try:
                await recognizer.load()
            except Exception as exc:
                print(f"{name}: unavailable — {exc}")
                continue
        elif not recognizer.backends:
            print(f"{name}: no API key configured (set GEMINI_API_KEY / GROQ_API_KEY)")
            continue
        rows = await measure(name, recognizer, clips, repeat=args.repeat, language=args.language)
        for row in rows:
            mark = "✗" if row["error"] else "·"
            print(f"  {mark} {name:18} {row['clip'][:22]:22} {row['ms']:7.0f}ms  {row['text'][:60]}")
            if row["error"]:
                print(f"      {row['error']}")
        summaries.append(summarise(name, rows, truth))

    print()
    header = f"{'engine':20} {'n':>3} {'p50':>8} {'p95':>8} {'max':>8} {'RTF':>6} {'WER':>6} {'fail':>4}"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary['engine']:20} {summary['n']:>3} "
            f"{_fmt(summary['p50_ms']):>8} {_fmt(summary['p95_ms']):>8} {_fmt(summary['max_ms']):>8} "
            f"{_fmt(summary['realtime_factor']):>6} {_fmt(summary['mean_wer']):>6} {summary['failed']:>4}"
        )

    out = Path(args.out or "data/bench-asr.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"clips": len(clips), "summary": summaries}, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}   (paste the table into docs/levels/NOTES-L02.md)")

    # The gate, stated out loud so nobody has to remember the thresholds.
    for summary in summaries:
        if summary["p50_ms"] is None:
            continue
        target = 1200 if summary["engine"] == "cloud" else 3500
        verdict = "PASS" if summary["p50_ms"] <= target else "FAIL"
        print(f"{verdict}: {summary['engine']} p50 {summary['p50_ms']:.0f}ms (target ≤{target}ms)")
    return 0


def _fmt(value) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}" if float(value) >= 10 else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="a WAV file, a folder of WAVs, or a capture-dump session")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--truth", help="TSV of <file>\t<what was said>")
    parser.add_argument("--from-manifest", action="store_true", help="use the dump's own transcripts")
    parser.add_argument("--engines", default="cloud,local:darija", help="comma-separated")
    parser.add_argument("--language", default="ar-MA", help="language hint (ar-MA, en-GB, unknown)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--out", default="data/bench-asr.json")
    args = parser.parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
