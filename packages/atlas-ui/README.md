# atlas-ui

The orb: a minimal, emotion-reactive interface that shows what Atlas is doing
without becoming a dashboard you stop looking at.

**Status: protocol only — the window and the orb ship in L5.**

| Module | State | What it holds |
|---|---|---|
| `protocol.py` | ✅ working | `UiMessage` models (state, caption, mood, level, tool, confirm, degraded), topic→message mapping, mood→hue table, and `typescript()` which **generates** the browser contract from the Python models |
| `bridge.py` | 🚧 L5 | FastAPI + WebSocket event bridge (30 Hz level throttle, token batching) |
| `window.py` | 🚧 L5 | pywebview/WebView2 frameless orb, tray icon, global hotkeys |

Why the schema is generated: the browser cannot import Python, so the one thing
guaranteed to drift is the wire format. `typescript()` makes the `.ts` interfaces
a build artifact — never hand-edited — which keeps the no-duplication rule (R5)
true across languages too.

```bash
python -m atlas ui-protocol > packages/atlas-ui/orb/protocol.ts
```
