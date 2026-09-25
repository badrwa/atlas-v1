# Darija TTS sidecar

The Moroccan voice for Atlas. It runs **in this folder, in its own virtualenv**,
and Atlas talks to it over `http://127.0.0.1:8125`. Nothing in here is imported
by the core environment — that separation is what keeps the core torch-free
(`scripts/check_no_torch.py` fails the build otherwise) and keeps a crash here
from taking the assistant down with it.

## Why this model

`KandirResearch/DarijaTTS-v0.1-500M` — Apache-2.0, a fine-tune of OuteTTS-0.2-500M
for Moroccan Darija, with a GGUF Q8_0 build that runs on two Skylake cores. It is
the only openly licensed Darija voice that does not need a GPU. It is *young*:
expect mispronunciations, and treat the Piper-Arabic fallback as a real fallback.

## Install (once)

```powershell
cd vendor\darija-tts
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Download the GGUF (≈550 MB) from
<https://huggingface.co/KandirResearch/DarijaTTS-v0.1-500M> into
`vendor\darija-tts\models\`, then point Atlas at it if the filename differs:

```powershell
setx ATLAS_DARIJA_MODEL "%CD%\models\darija-tts-v0.1-500m.Q8_0.gguf"
```

`config.toml` already expects the interpreter at
`vendor/darija-tts/.venv/Scripts/python.exe`; `atlas voice` tells you if it is not
there. You do not have to start the server yourself — Atlas starts it on demand
through the resource lease and stops it when the lease expires. Starting it by
hand is supported too:

```powershell
.venv\Scripts\python server.py --port 8125 --preload -v
```

## Check it

```powershell
# health, from another terminal
curl http://127.0.0.1:8125/health

# the real test: does a Moroccan understand it on the first listen?
python scripts\spike_darija_tts.py --out data\spike-darija
```

The spike writes ten WAVs and a scoring sheet. Its verdict — and the files
themselves — belong in `docs/levels/NOTES-L03.md`, because "the Darija voice is
good enough" is a claim that needs evidence.

## Endpoints

| Method | Path | Body | Answer |
|---|---|---|---|
| GET | `/health` | — | `{"ok", "ready", "model", "error"}` |
| POST | `/synthesize` | `{"text", "voice", "rate"}` | `audio/wav` (16-bit PCM) |

`rate` is honoured only with `--honour-rate`, because it is applied by resampling:
speed and pitch move together. Atlas's prosody table only ever asks for 0.92–1.05,
so the artefact is small — but it is an artefact, and the flag exists so the
choice is yours rather than hidden.

## Troubleshooting

| Symptom | What it means |
|---|---|
| `503 outetts is not installed` | the venv is missing or pip install failed — run the install above |
| `503 model not found at …` | the GGUF is not where `ATLAS_DARIJA_MODEL` points |
| `atlas voice` says *not installed yet* | Atlas cannot find `.venv/Scripts/python.exe` (path in `config.toml`) |
| First sentence takes ~10 s | model load; `--preload` (or just talking to Atlas) pays it once |
| Audio sounds wrong/garbled | outetts API drift — check `vendor/darija-tts/server.py` against the model card's demo script and note it in NOTES-L03 |
