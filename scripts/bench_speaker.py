#!/usr/bin/env python3
"""Pick the speaker threshold from *your* voice — FAR first, FRR second.

The plan's rule for L4 is blunt: a false accept is a leak, a false reject is one
annoying re-ask, so choose the lowest threshold whose false-accept rate is zero,
even if that means a higher false-reject rate.  This script computes both from
your own clips, so the number in `config.toml` is measured rather than copied.

Layout it expects (all WAV, 16 kHz, one folder per voice)::

    data/speaker_clips/
      badr/  01.wav 02.wav 03.wav 04.wav 05.wav
      said/  01.wav 02.wav 03.wav
      stranger/ 01.wav 02.wav          # anyone else you can record

The first `--enrol` clips of every folder become that person's profile (the same
embeddings `atlas identity enrol` would store).  Every remaining clip is then a
*test trial*:

* scored against its own profile  → a genuine trial (a high score is required);
* scored against every other profile → an impostor trial (a high score is a leak).

    python scripts/bench_speaker.py                    # sweep and recommend
    python scripts/bench_speaker.py --enrol 3 --json   # machine-readable
    python scripts/bench_speaker.py --from-log         # what the log already says

`--from-log` cannot compute FAR — `data/speaker_log.jsonl` records the *decision*,
not the label — so it prints how many past decisions would flip instead.  That is
still the honest answer to "would a different threshold change my life?".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("atlas-core", "atlas-audio", "atlas-mind", "atlas-obsidian", "atlas-skills"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))

from atlas_audio.speaker import SherpaSpeakerVerifier, as_samples, speech_ms  # noqa: E402
from atlas_core.config import load_config  # noqa: E402
from atlas_core.identity import IdentityConfig, cosine  # noqa: E402

#: The sweep.  0.30 is "matches almost anybody", 0.90 is "only on a very good day".
THRESHOLDS = [round(0.30 + step * 0.05, 2) for step in range(13)]
#: Trial floors: a clip below this is not a usable trial, and the script says so.
MIN_SPEECH_MS = 1500.0


def sweep(genuine: list[float], impostor: list[float], thresholds: list[float]) -> list[dict]:
    """FAR/FRR for every threshold — the whole point of the script."""
    rows: list[dict] = []
    for threshold in thresholds:
        false_accepts = sum(1 for score in impostor if score >= threshold)
        false_rejects = sum(1 for score in genuine if score < threshold)
        rows.append(
            {
                "threshold": threshold,
                "far": false_accepts / len(impostor) if impostor else 0.0,
                "frr": false_rejects / len(genuine) if genuine else 0.0,
                "false_accepts": false_accepts,
                "false_rejects": false_rejects,
                "impostor_trials": len(impostor),
                "genuine_trials": len(genuine),
            }
        )
    return rows


def recommend(rows: list[dict]) -> dict:
    """Lowest threshold with FAR == 0; the plan's preference, in code."""
    clean = [row for row in rows if row["false_accepts"] == 0]
    if not clean:
        return {
            "threshold": max(row["threshold"] for row in rows),
            "reason": "no threshold reached FAR 0 — record more impostor clips",
            "far": rows[-1]["far"],
            "frr": rows[-1]["frr"],
        }
    best = min(clean, key=lambda row: row["threshold"])
    return {
        "threshold": best["threshold"],
        "reason": "lowest threshold with no false accepts (the plan's rule)",
        "far": best["far"],
        "frr": best["frr"],
    }


def read_clip(path: Path, verifier: SherpaSpeakerVerifier) -> list[float] | None:
    import wave

    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            rate = handle.getframerate()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, OSError) as exc:
        print(f"  ! {path}: {exc}", file=sys.stderr)
        return None
    if channels != 1 or rate != 16000 or width != 2:
        print(
            f"  ! {path}: needs 16 kHz mono 16-bit PCM (got {rate} Hz, "
            f"{channels} ch, {width * 8}-bit)",
            file=sys.stderr,
        )
        return None
    samples = as_samples(raw)
    voiced = speech_ms(samples)
    if voiced < MIN_SPEECH_MS:
        print(f"  ! {path}: only {voiced:.0f} ms of speech — skipped", file=sys.stderr)
        return None
    embedding = verifier.embed_or_none(samples)
    if embedding is None:
        print(f"  ! {path}: no embedding (model or audio problem)", file=sys.stderr)
    return embedding


def collect(clips_dir: Path, enrol: int, verifier: SherpaSpeakerVerifier) -> tuple[dict, dict]:
    """`{name: [embedding, …]}` for the profiles, `{name: [embedding, …]}` for tests."""
    profiles: dict[str, list[list[float]]] = {}
    trials: dict[str, list[list[float]]] = {}
    for folder in sorted(path for path in clips_dir.iterdir() if path.is_dir()):
        wavs = sorted(folder.glob("*.wav"))
        if not wavs:
            continue
        embeddings = [vector for wav in wavs if (vector := read_clip(wav, verifier)) is not None]
        if not embeddings:
            continue
        profiles[folder.name] = embeddings[:enrol]
        trials[folder.name] = embeddings[enrol:]
        print(
            f"  · {folder.name}: {len(profiles[folder.name])} enrol, "
            f"{len(trials[folder.name])} test clip(s)"
        )
    return profiles, trials


def score_trials(profiles: dict, trials: dict) -> tuple[list[float], list[float]]:
    """Every trial against its own profile (genuine) and everyone else's (impostor)."""
    genuine: list[float] = []
    impostor: list[float] = []
    for owner, vectors in trials.items():
        for vector in vectors:
            for name, window in profiles.items():
                score = max(cosine(vector, other) for other in window)
                (genuine if name == owner else impostor).append(score)
    return genuine, impostor


def print_table(rows: list[dict]) -> None:
    print(f"  {'threshold':>9}  {'FAR':>6}  {'FRR':>6}   {'accepts':>7}  {'rejects':>7}")
    for row in rows:
        print(
            f"  {row['threshold']:>9.2f}  {row['far']:>6.2f}  {row['frr']:>6.2f}   "
            f"{row['false_accepts']:>7}  {row['false_rejects']:>7}"
        )


def from_log(config) -> int:
    """What the recorded decisions already say — no model, no clips needed."""
    from atlas_core.identity import SpeakerLog

    log = SpeakerLog(config.identity.log_path)
    events = log.events
    if not events:
        print(f"no events in {config.identity.log_path} — run `atlas listen` first")
        return 1
    thresholds = sorted({round(event.threshold, 2) for event in events})
    print(f"{len(events)} utterance(s) recorded, threshold(s) used: {thresholds}")
    for event in events[-10:]:
        mark = "accept" if event.accepted else "reject"
        print(
            f"  {event.speaker or '(unknown)':<12} {event.score:.3f}  {mark:<6} "
            f"{event.reason}  {event.utterance_ms:.0f} ms"
        )
    scores = [event.score for event in events]
    print()
    print(f"  accepted scores: {sorted(s for s, e in zip(scores, events, strict=True) if e.accepted)}")
    print(
        "  → a threshold can be re-picked from these numbers, but only your own clips "
        "can tell a *stranger* from a quiet day: run this script with --clips."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config.toml", help="config file")
    parser.add_argument("--clips", default="data/speaker_clips", help="one folder per voice")
    parser.add_argument("--enrol", type=int, default=3, help="clips per person used as profile")
    parser.add_argument("--model", default="", help="override the speaker ONNX path")
    parser.add_argument("--from-log", action="store_true", help="read data/speaker_log.jsonl")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--out", default="", help="also write the JSON summary here")
    parser.add_argument("--markdown", default="", help="write a short verdict here")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    settings = IdentityConfig.from_config(config)
    if args.from_log:
        return from_log(config)

    clips_dir = Path(args.clips)
    if not clips_dir.is_dir():
        print(f"no clips at {clips_dir}")
        print("  → record a few WAV files per voice (16 kHz mono), one folder each:")
        print("      data/speaker_clips/you/ *.wav      (4–5 clips, 3–5 s each)")
        print("      data/speaker_clips/friend/ *.wav   (2 clips is enough)")
        print("  you can make them with: atlas audio replay --capture-dump, or any recorder")
        return 1

    verifier = SherpaSpeakerVerifier(args.model or settings.model_path)
    if not verifier.available():
        print(f"speaker model unavailable: {verifier.missing()}")
        print(f"  → expected at {settings.model_path} (see docs/levels/NOTES-L04.md)")
        return 1
    if not verifier.is_loaded():
        import asyncio

        asyncio.run(verifier.load())
    if not verifier.is_loaded():
        print(f"speaker model failed to load: {verifier.load_error or 'unknown error'}")
        return 1

    print(f"Reading clips from {clips_dir} (model {settings.model_path})")
    profiles, trials = collect(clips_dir, max(1, args.enrol), verifier)
    if len(profiles) < 2:
        print("need at least two voices: one to recognise, one to *refuse*")
        return 1

    genuine, impostor = score_trials(profiles, trials)
    if not genuine or not impostor:
        print("not enough test clips left after enrolment — record more per voice")
        return 1

    rows = sweep(genuine, impostor, THRESHOLDS)
    verdict = recommend(rows)
    print()
    print_table(rows)
    print()
    print(f"  genuine trials: {len(genuine)} · impostor trials: {len(impostor)}")
    print(
        f"  recommended threshold: {verdict['threshold']:.2f}  "
        f"(FAR {verdict['far']:.2f}, FRR {verdict['frr']:.2f}) — {verdict['reason']}"
    )

    result = {
        "model": settings.model_path,
        "clips_dir": str(clips_dir),
        "people": sorted(profiles),
        "genuine": genuine,
        "impostor": impostor,
        "rows": rows,
        "recommended": verdict,
        "current_threshold": settings.threshold,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  wrote {args.out}")
    if args.markdown:
        lines = [
            "# Speaker threshold",
            "",
            f"- model: `{settings.model_path}`",
            f"- people: {', '.join(sorted(profiles))}",
            f"- genuine trials: {len(genuine)} · impostor trials: {len(impostor)}",
            f"- current threshold: {settings.threshold:.2f}",
            f"- **recommended: {verdict['threshold']:.2f}** "
            f"(FAR {verdict['far']:.2f}, FRR {verdict['frr']:.2f})",
            f"- why: {verdict['reason']}",
            "",
            "| threshold | FAR | FRR | false accepts | false rejects |",
            "| --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {row['threshold']:.2f} | {row['far']:.2f} | {row['frr']:.2f} | "
            f"{row['false_accepts']} | {row['false_rejects']} |"
            for row in rows
        ]
        Path(args.markdown).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {args.markdown}")
    return 0 if verdict["far"] == 0.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
