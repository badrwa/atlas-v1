# 50_Atlas — Atlas's own brain space

| File | Purpose |
|---|---|
| `memory.md` | durable facts about you, edited by voice ("Atlas, remember…") |
| `preferences.md` | how you like to be talked to: humor level, language default, pet peeves |
| `jokes.md` | running gags and callbacks — this is what makes it feel like a friend, not a bot |
| `log/` | per-session transcripts + timings + speaker + emotion tags |
| `study/` | summaries of open-source repos Atlas studied, and the patterns it extracted |

`memory.md` / `preferences.md` / `jokes.md` are injected into the system prompt (token-budgeted, newest first).
Keep them short and factual: 200–400 words total beats a 5,000-word dump for latency *and* quality.
