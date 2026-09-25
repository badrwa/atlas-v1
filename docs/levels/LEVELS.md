# ATLAS — The Levels

Eleven levels, strictly ordered. **No level starts before the previous level's *done-when* passes on the real laptop.** That gate is the only thing standing between you and an abandoned project.

| Level | File | Goal | Effort | Gate (short form) |
|---|---|---|---|---|
| L0 | [LEVEL-00-foundation.md](LEVEL-00-foundation.md) | Skeleton that makes the rest pleasant: workspace, contracts, DI, events, FSM, config, tests, CI | 1 weekend | `atlas doctor` passes, CI green |
| L1 | [LEVEL-01-brain.md](LEVEL-01-brain.md) | Text Atlas in Darija + en-GB, streamed, with provider fallback and timings | ~1 week | CLI chat, < 1.5 s to first token |
| L2 | [LEVEL-02-ears.md](LEVEL-02-ears.md) | Wake word, VAD, ASR (cloud + local Darija), language routing | ~1.5 weeks | Hands-free command in both languages |
| L3 | [LEVEL-03-mouth.md](LEVEL-03-mouth.md) | Piper en-GB + DarijaTTS + cache + prosody; sentence streaming | ~1 week | < 3 s to first spoken word |
| L4 | [LEVEL-04-identity.md](LEVEL-04-identity.md) | Enrolment, verification, restricted mode, per-person profiles | 3–4 evenings | Stranger voice blocked from private actions |
| L5 | [LEVEL-05-ui.md](LEVEL-05-ui.md) | Orb UI, mood visuals, captions, tray, hotkeys | ~1 week | UI mirrors every state in < 100 ms |
| L6 | [LEVEL-06-brain-obsidian.md](LEVEL-06-brain-obsidian.md) | Vault schema, FTS5 index, writes, git journal, REST/MCP | ~1.5 weeks | Voice note → vault → undo |
| L7 | [LEVEL-07-hands.md](LEVEL-07-hands.md) | Skill registry, Windows control, confirmations, MCP bridge, red-team | ~2 weeks ongoing | 15 spoken commands, destructive ones confirmed |
| L8 | [LEVEL-08-personality.md](LEVEL-08-personality.md) | Friend voice, dialect packs, humour engine, mood calibration | ~1 week + tuning | It feels like a friend, jokes land, tone adapts |
| L9 | [LEVEL-09-github.md](LEVEL-09-github.md) | Study pipeline, own packages, releasing, HF artifacts | ~1.5 weeks ongoing | 5 repos studied, 1 package released |
| L10 | [LEVEL-10-residency.md](LEVEL-10-residency.md) | Autostart, watchdogs, governor, updater, soak | ~1 week | 24 h soak, survives reboot, ≤ 700 MB idle |

### Rules that apply to every level

1. **Definition of done includes a hardware measurement**, not just "it works".
2. **Everything heavy is leased** (`ResourceLease`), never imported into the resident core.
3. **No new dependency without a row in §4 of the master plan** (why + alternative rejected).
4. **Every new capability ships with a fake** so CI can test it without a mic, a cloud key or Windows.
5. **Every level leaves the repo in a runnable state** — `python -m atlas doctor` must pass at the end of each level.
6. **Keep a `docs/levels/NOTES-L0X.md`** with real numbers you measured (cold start, RAM, latency). Future-you will need them.
