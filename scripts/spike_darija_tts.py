#!/usr/bin/env python3
"""The Darija voice spike: ten sentences, ten WAVs, one honest verdict.

LEVEL-03 asks for this *before* anything is built on top of the Darija voice,
because "the open-source Darija TTS is good enough" is a claim that needs
evidence, and the evidence is a Moroccan listening to ten files.

    python scripts/spike_darija_tts.py                 # synthesise into data/spike-darija
    python scripts/spike_darija_tts.py --score         # listen first, then score
    python scripts/spike_darija_tts.py --piper         # same sentences through Piper-Arabic
    python scripts/spike_darija_tts.py --sidecar-only  # do not auto-start the server

Scoring writes `data/spike-darija/scores.json` and a markdown table you can paste
straight into `docs/levels/NOTES-L03.md`.  Nothing here is imported by Atlas: it
is a measurement tool, like `bench_asr.py` and `bench_tts.py`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "atlas-audio" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "atlas-core" / "src"))

from atlas_audio.speech import SentenceStreamer  # noqa: E402
from atlas_audio.tts import (  # noqa: E402
    DarijaTtsSidecarSynthesizer,
    PiperSynthesizer,
    SidecarConfig,
    VoiceConfig,
)

#: Ten sentences that between them exercise everything a voice can get wrong:
#: greeting, question, numbers, a name, an address, a joke, a long sentence and
#: the code-switching that Darija does constantly.
SENTENCES: list[tuple[str, str]] = [
    ("greeting", "Salam 3likom, ana Atlas, lmasa3id dyalek."),
    ("question", "Chno ljaw dyalek lyoum? Bghiti n3awnek b chi haja?"),
    ("numbers", "Lweqt daba rah t3a w rb3, w lkhedma ghadi tsali f 5.30."),
    ("name", "Smitek Badr, w kantsken f Mohammedia."),
    ("address", "Sift lih hadchi f 12, zanqa Hassan II, lwla."),
    ("joke", "Chno kayn? Hada ghir l'ordinateur, machi lmu3alim!"),
    ("code-switch", "Safi, daba ghadi nchecker l'email dyalek, ghir chwiya."),
    ("long", "Ila bghiti, n9dar nkellemek 3la l'code dyalek, nghir lik l'erreurs, "
             "w n3tik des conseils 3la kifach t7سن l performance."),
    ("confirmation", "Rak mt2akked men hadchi? Bghiti nkemmel?"),
    ("count", "Wahed, jouj, tlata, rb3a, khamsa, setta, seb3a, tmnya, tse3a, 3chra."),
]


async def synthesize(
    engine_name: str,
    *,
    out_dir: Path,
    sidecar_only: bool,
    config_path: str | None,
) -> list[dict[str, object]]:
    from atlas_core.config import load_config

    config = load_config(config_path) if Path(config_path or "config.toml").exists() else None
    rows: list[dict[str, object]] = []

    if engine_name == "piper_arabic":
        voices = VoiceConfig.from_config(config)
        path = voices.piper_path(voices.arabic_voice)
        engine: object = PiperSynthesizer(
            path, voice=voices.arabic_voice, language="ar-MA", config_path=path + ".json"
        )
        if not engine.available():
            print(f"  ✗ {engine.missing()}")
            return rows
    else:
        engine = DarijaTtsSidecarSynthesizer(SidecarConfig.from_config(config))
        if not sidecar_only:
            started = await engine.sidecar.ensure_started()
            if not started:
                print(f"  ✗ sidecar not reachable at {engine.config.url}")
                print("    start it: vendor/darija-tts/.venv/Scripts/python vendor/darija-tts/server.py")
                return rows

    streamer = SentenceStreamer([engine])  # type: ignore[list-item]
    await engine.load() if hasattr(engine, "load") else None

    for label, text in SENTENCES:
        sentences = await streamer.say(text, language="ar-MA")
        if not sentences or not sentences[0].audible:
            print(f"  ✗ {label}: no audio")
            continue
        target = out_dir / f"{label}.wav"
        _write_wav(target, sentences)
        rows.append(
            {
                "label": label,
                "text": text,
                "file": str(target),
                "seconds": round(sum(len(s.pcm) / 2 / s.sample_rate for s in sentences), 2),
                "synth_ms": round(sum(s.synth_ms for s in sentences), 1),
                "engine": sentences[0].engine,
            }
        )
        print(
            f"  ✓ {label:<13} {rows[-1]['seconds']:>5}s  "
            f"{rows[-1]['synth_ms']:>7} ms  {target.name}"
        )
    return rows


def _write_wav(target: Path, sentences: list) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sentences[0].sample_rate)
        for sentence in sentences:
            handle.writeframes(sentence.pcm)


def score(rows: list[dict[str, object]], out_dir: Path) -> int:
    """Ask the only judge that matters, one question per file."""
    print("\nScore each file: 1 = would a Moroccan understand it first time? (y/n)")
    print("               2 = naturalness 1-5   3 = would you use this voice? (y/n)")
    scores: list[dict[str, object]] = []
    for row in rows:
        print(f"\n  {row['label']}: {row['file']}")
        try:
            understood = input("    intelligible first listen? [y/n] ").strip().lower()
            natural = input("    naturalness 1-5 [3] ").strip() or "3"
            usable = input("    would you use it? [y/n] ").strip().lower()
        except EOFError:  # piped input: no answers, no scores
            print("    (no tty — scoring skipped)")
            return 1
        scores.append(
            {
                **row,
                "intelligible": understood.startswith("y"),
                "naturalness": int(natural) if natural.isdigit() else 3,
                "usable": usable.startswith("y"),
            }
        )

    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    understood = sum(1 for s in scores if s["intelligible"])
    usable = sum(1 for s in scores if s["usable"])
    natural = sum(int(s["naturalness"]) for s in scores) / max(1, len(scores))

    table = ["| sentence | intelligible | naturalness | usable |", "|---|---|---|---|"]
    for item in scores:
        table.append(
            f"| {item['label']} | {'✓' if item['intelligible'] else '✗'} "
            f"| {item['naturalness']}/5 | {'✓' if item['usable'] else '✗'} |"
        )
    table.append("")
    table.append(f"**Verdict:** {understood}/{len(scores)} intelligible first listen, "
                 f"{natural:.1f}/5 natural, {usable}/{len(scores)} good enough to ship.")
    (out_dir / "verdict.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    print(f"\n  {understood}/{len(scores)} intelligible · {natural:.1f}/5 natural · {usable}/{len(scores)} usable")
    print(f"  written: {out_dir / 'scores.json'} and {out_dir / 'verdict.md'}")
    print("  → paste verdict.md into docs/levels/NOTES-L03.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/spike-darija", help="where the WAVs go")
    parser.add_argument("--score", action="store_true", help="score the files already generated")
    parser.add_argument("--piper", action="store_true", help="use Piper-Arabic instead of the sidecar")
    parser.add_argument("--sidecar-only", action="store_true", help="do not auto-start the sidecar")
    parser.add_argument("--config", help="config.toml to read voices from")
    parser.add_argument("--json", action="store_true", help="print the manifest as JSON")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    manifest = out_dir / "manifest.json"

    if args.score:
        rows = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
        if not rows:
            print("  nothing to score — run the spike first")
            return 1
        return score(rows, out_dir)

    engine_name = "piper_arabic" if args.piper else "darija_tts_sidecar"
    print(f"ATLAS Darija voice spike · engine {engine_name}")
    rows = asyncio.run(
        synthesize(
            engine_name,
            out_dir=out_dir,
            sidecar_only=args.sidecar_only,
            config_path=args.config,
        )
    )
    if not rows:
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    total = sum(float(row["seconds"]) for row in rows)
    print(f"\n  {len(rows)} files · {total:.1f}s of speech · {out_dir}")
    print("  → listen to them, then: python scripts/spike_darija_tts.py --score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
