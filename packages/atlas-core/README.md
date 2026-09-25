# atlas-core

The ATLAS kernel — the only package every other Atlas package may depend on.

| Module | What it is |
|---|---|
| `contracts.py` | The ABCs: `Sense`, `Engine`, `LlmProvider`, `SpeechRecognizer`, `SpeechSynthesizer`, `WakeWordEngine`, `SpeakerVerifier`, `Skill`, `Store` + shared value types |
| `fsm.py` | The interaction state machine (dormant → waking → listening → thinking → speaking), table-driven and side-effect free |
| `events.py` | Typed events + async `EventBus` (UI, logger, tray and vault-log subscribe; core never imports them) |
| `resources.py` | `ResourceLease` + `ResourceGovernor` — how Atlas fits in 8 GB (load on demand, evict on idle) |
| `di.py` | 40-line dependency injector (register / resolve / override-in-tests) |
| `config.py` | pydantic v2 config models, TOML + `.env` loading, profile overlays |
| `timings.py` | `timings.jsonl` recorder with p50/p95 summaries |
| `fakes.py` | A fake for every contract, so the whole turn can run in CI without hardware or keys |

Design rules (see `docs/architecture/ARCHITECTURE.md` §3): ports & adapters, one
abstraction per concept, no UI imports in the core, every heavy engine leased.
