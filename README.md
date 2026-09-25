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
| L2 | Ears (wake word, VAD, ASR cloud + local Darija) | not started |
| L3 | Mouth (Piper en-GB, DarijaTTS, cache, prosody) | not started |
| L4 | Identity (voice enrolment, verification, restricted mode) | not started |
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
work before its level lands — `atlas listen` says the ear arrives in L2.

What is real today: the kernel (`atlas-core`: contracts, event bus, FSM, leases,
timings, config, fakes), the brain (`atlas-mind`: seven providers, quota/retry/
timing chain, Darija normalisation, lexicon, dialect packs, persona, streaming
chat), the vault writer (`atlas-obsidian`: git-journaled, undoable), the skill
registry with an owner gate, and the CLI. The brain holds a Darija-first
conversation, switches to British English on request, answers with one honest
sentence when the network is gone, and keeps its own per-turn overhead at 0.2 ms
— so the only thing a user waits for is the model. 267 tests, ruff + mypy +
duplicate-code and architecture checks in CI.

## Ground rules

1. No PyTorch, no CUDA, no GPU assumptions in the core. Native Windows, no Docker/WSL2, no Electron.
2. Heavy engines are **leased** (loaded on demand, released on idle) — that's how this survives 8 GB of RAM.
3. Darija is a first-class language path (ASR, TTS, normalisation, humour, lexicon), not a translation layer.
4. Every vault write is git-committed and undoable by voice. Nothing is ever deleted, only archived.
5. Permissions: `SAFE` / `CONFIRM` / `BLOCKED`, gated by voice identity. No shell, ever. Tool output is data, never instructions.
6. No API keys in git (`.env` + gitleaks pre-commit). Models live in `%LOCALAPPDATA%\atlas\models`.
7. `pytest` + `ruff` + `mypy` + duplicate-code check in CI — requirement 5 (OOP, no duplication) is enforced by the build, not by discipline.
