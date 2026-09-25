# atlas-v1

**ATLAS** — a voice-first, Jarvis-style assistant that lives inside a Dell Latitude 5480 (i5-6200U, 8 GB RAM, Intel HD 520, Windows 10 x64).

Speaks **Moroccan Darija** by default and **British English** second · knows your voice · keeps its brain in **Obsidian** · minimal emotion-reactive orb UI · jokes like a friend · OOP, no duplicated code · studies open source and publishes its own libraries.

## Documentation map

| Doc | What |
|---|---|
| [`docs/ATLAS_PLAN.md`](docs/ATLAS_PLAN.md) | **Start here.** Requirements → decisions, hardware verdicts, tech stack with why/where/how, budgets, risks, level roadmap, 20 acceptance tests |
| [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) | Layers, OOP contracts, **16 diagrams** (component, class, sequences, states, ER, deployment, flows), coding standards, no-duplication rules, test strategy |
| [`docs/levels/LEVELS.md`](docs/levels/LEVELS.md) | The level index + gates |
| [`docs/levels/LEVEL-00…10`](docs/levels/) | Step-by-step implementation per level, each with classes, tests, done-when, pitfalls |
| [`vault-template/`](vault-template/) | The Obsidian vault skeleton Atlas expects (folders, templates, memory/jokes/preferences, folder contract) |

## Level status

| Level | Name | State |
|---|---|---|
| L0 | Foundation (workspace, contracts, DI, events, FSM, CI) | **done — see [NOTES-L00](docs/levels/NOTES-L00.md)** |
| L1 | Brain (text Darija + en-GB, providers, persona, timings) | **done — see [NOTES-L01](docs/levels/NOTES-L01.md)** |
| L2 | Ears (wake word, VAD, ASR cloud + local Darija) | **done — see [NOTES-L02](docs/levels/NOTES-L02.md)** |
| L3 | Mouth (Piper en-GB, DarijaTTS, cache, prosody) | **done — see [NOTES-L03](docs/levels/NOTES-L03.md)** |
| L4 | Identity (voice enrolment, verification, restricted mode) | **done — see [NOTES-L04](docs/levels/NOTES-L04.md)** (laptop: enrol + sweep) |
| L5 | Face (orb UI, captions, tray, hotkeys) | not started |
| L6 | Second brain (Obsidian vault, FTS5, git journal, MCP) | not started |
| L7 | Hands (skills, PC control, permissions, red-team) | not started |
| L8 | Soul (friend voice, dialect packs, humour, mood) | not started |
| L9 | Growth (GitHub study loop, own packages, HF artifacts) | not started |
| L10 | Residency (autostart, watchdogs, governor, soak) | not started |

## Run it

```bash
uv sync --all-packages          # or: pip install -e packages/* -e apps/atlas
cp .env.example .env            # add your keys (Gemini first: aistudio.google.com/apikey)
python -m atlas doctor          # what works on this machine, honestly
python -m atlas chat            # talk to it — Darija by default, /help for commands
python -m atlas chat --structured   # also get language/emotion metadata per turn

python -m atlas audio devices   # is there a microphone, and can Python see it
python -m atlas audio test      # record one second and play it back
python -m atlas listen --status # wake engine, VAD backend, ASR mode, voice chain
python -m atlas listen          # say "atlas" and speak — Atlas answers out loud
python -m atlas listen --ptt    # push-to-talk: press Enter, talk, press Enter
python -m atlas listen --no-voice   # the same, captions only

python -m atlas voice           # which voice would speak, and what it still needs
python -m atlas voice warm      # cache the lines Atlas repeats (greetings, "safi")
python -m atlas say "Salam, ana Atlas" --language ar-MA      # speak one line
python -m atlas say "Right." --out out.wav --language en-GB  # or write a WAV

python -m atlas identity              # who Atlas knows, and what each voice may reach
python -m atlas identity enrol --name badr --owner   # 3 samples × 10 s, by microphone
python -m atlas identity verify clip.wav             # score one clip against every profile
python -m atlas identity log                         # every score, every decision
python -m atlas identity forget --name said          # forget one voice (or --all --yes)
python scripts/bench_speaker.py                      # pick the threshold from YOUR clips
```

Inside `chat`: `/lang en-GB` switches language, `/provider groq` pins a brain,
`/mood` shows how Atlas reads the room, `/timing` shows where the milliseconds
go. A reply line looks like:

```
atlas ▸ safi, dakhla daba.
  [groq · ttft 380ms · total 1420ms · ar-MA · happy · mood happy]
```

A fresh clone with no keys still runs: `doctor` reports what is missing, and
`chat` answers with one honest sentence instead of silence. Nothing pretends to
work before its level lands: `atlas listen --status` names the wake engine it
will actually use, the VAD backend, and whether any cloud key is present.

The ears and the mouth need optional extras, and say so rather than failing:

```bash
pip install -e "packages/atlas-audio[audio]"   # sounddevice + numpy: microphone and speakers
pip install -e "packages/atlas-audio[local]"   # sherpa-onnx (wake + VAD), faster-whisper, piper
pip install -e "packages/atlas-audio[all]"     # everything above, plus the hotkey extra
```

The British voice is a 63 MB ONNX file — download `en_GB-alan-medium.onnx` (+
`.onnx.json`) from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices)
into `models/tts/`. The Darija voice is a separate process in its own venv
(`vendor/darija-tts/`, ~550 MB model) because the open Darija TTS needs an
inference stack Atlas refuses to import; until it is installed, Piper-Arabic
speaks Darija and the chain says so. `atlas voice` prints exactly what is missing.

Half duplex is not a setting: the microphone is closed (reference counted) for as
long as any audio is playing, so Atlas cannot hear itself.   # sounddevice + numpy: a microphone at all
pip install -e "packages/atlas-audio[local]"   # sherpa-onnx (wake + VAD) and faster-whisper
```

What is real today: the kernel (`atlas-core`: contracts, event bus, FSM, leases,
timings, config, fakes), the brain (`atlas-mind`: seven providers, quota/retry/
timing chain, Darija normalisation, lexicon, dialect packs, persona, streaming
chat), the vault writer (`atlas-obsidian`: git-journaled, undoable), the skill
registry with an owner gate, and the CLI. The brain holds a Darija-first
conversation, switches to British English on request, answers with one honest
sentence when the network is gone, and keeps its own per-turn overhead at 0.2 ms
— so the only thing a user waits for is the model.

The ears (`atlas-audio`: frames, Silero VAD, sherpa-onnx wake word, cloud-first
ASR with a faster-whisper Darija fallback) and the mouth (Piper en-GB + Arabic,
the Darija sidecar, a SQLite TTS cache, mood-driven prosody, sentence streaming,
half-duplex playback) are on top of that. Streaming means the first word arrives
after one sentence is written, not after the whole answer: `atlas listen` speaks
while the model is still typing.

Identity (L4) sits in front of all of it: one sherpa-onnx speaker embedding per
utterance, one verdict, one place that decides what a voice may reach
(`ContextGuard`). The owner gets everything; an enrolled other person gets
general conversation plus their own note; an unrecognised voice gets general
conversation and a polite refusal in Darija — and the log records *why*, so
"nobody is enrolled" and "the model is missing" can never pass silently as
"everyone is the owner". Voice prints live in `data/` (gitignored), never in the
vault and never on the wire. **581 tests**, ruff + mypy + duplicate-code and
architecture checks in CI.

## Ground rules

1. No PyTorch, no CUDA, no GPU assumptions in the core. Native Windows, no Docker/WSL2, no Electron.
2. Heavy engines are **leased** (loaded on demand, released on idle) — that's how this survives 8 GB of RAM.
3. Darija is a first-class language path (ASR, TTS, normalisation, humour, lexicon), not a translation layer.
4. Every vault write is git-committed and undoable by voice. Nothing is ever deleted, only archived.
5. Permissions: `SAFE` / `CONFIRM` / `BLOCKED` **and** a capability set from the speaker verdict, gated in `SkillRegistry`. No shell, ever. Tool output is data, never instructions. A voice profile is a convenience gate, not a cryptographic identity: destructive actions still confirm.
6. No API keys in git (`.env` + gitleaks pre-commit). Models live in `%LOCALAPPDATA%\atlas\models`.
7. `pytest` + `ruff` + `mypy` + duplicate-code check in CI — requirement 5 (OOP, no duplication) is enforced by the build, not by discipline.
