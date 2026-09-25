# L5 — Face (the orb UI)

**Goal:** a minimal, beautiful orb that mirrors what Atlas is doing and how it feels — without a 300 MB browser engine, without a build step, and without slowing the core.
**Effort:** ~1 week. **Depends on:** L0 (usable any time; best after L3 so audio levels are real). **Gate:** every state change appears in < 100 ms, UI process ≤ 150 MB, core unaffected if the UI dies.

---

> **Implementation notes: [`NOTES-L05.md`](NOTES-L05.md)** — built and
> sandbox-verified (`pytest` 651 passed, ruff clean, mypy 101 files, both
> generator gates, and a live WebSocket run against uvicorn). Implemented:
> `atlas-ui` (protocol + generated `.ts`/`.js`, `ThemeEngine`, `UiHub` with the
> 30 Hz level and 20 Hz token throttles, `EventBridge`, `OrbWindow`/`WindowState`/
> `InstanceLock`/`TrayIcon`/`HotkeyManager`), the orb itself (HTML + Canvas2D,
> no framework), `atlas ui run|serve|status|protocol`, `atlas listen --ui`, the
> `data/ui-endpoint.json` handshake that lets the window attach to a running
> conversation, and the loop's own publishing (`_set_state` → `EventBus`).
> Ten bugs the tests caught are in §4 — the three that matter: `socket: WebSocket`
> became a required query parameter under `from __future__ import annotations`
> (every connection died with close code 1008); the window was created and never
> started (invisible orb, no error); and `system.error` was in neither the
> protocol table nor the bridge's subscription list, so an error the core raised
> would never have reached the orb.
> **Still the laptop's job (§5):** the window on real WebView2, the 100/125/150 %
> DPI pass, the transparent-vs-opaque decision on the HD 520 driver, RSS/CPU
> numbers, and the kill-the-orb test. The 30 fps cap and the 12-state union are
> checked in CI; "does it look like Atlas" is not.

## Deliverables

- `atlas-ui` package: `EventBridge` (FastAPI + WebSocket), `OrbWindow` (pywebview), `Tray`, `Hotkeys`, `protocol.py` (schema) → generated `orb/protocol.ts`.
- The orb itself: HTML + Canvas2D, no framework, no bundler.
- Tray menu (state, mute, restart audio, open vault, quit) and global hotkeys.

## Why pywebview + WebView2

WebView2 ships with Windows 10/11 (Edge), so there's no runtime to install; HTML/Canvas gets you the prettiest orb per MB; the UI lives in its own process so a UI crash can't take Atlas down; and there's no Rust/Node toolchain on a 2-core laptop. (Rejected: Tauri = toolchain + build step; Qt = slow to look good; Electron = 300 MB for a circle.)

## Design language (keep it minimal)

- One floating orb, ~140 px, frameless, transparent, drag-anywhere, position remembered.
- **State = motion**: dormant (slow breathing), waking (quick pulse + expanding ring), listening (inner rings rotate with input level), thinking (particles orbit, cool drift), speaking (amplitude-reactive blob, warm), confirming (amber, pauses), muted (grey, frozen, slash), offline/degraded (desaturated, small "offline" chip), error (brief red flash then back).
- **Mood = hue + warmth**: calm teal, happy warm gold, focused blue, frustrated red-orange, serious deep indigo, tired muted violet. Blend over ~800 ms so it feels alive, never flickers.
- Captions: last user utterance + streaming reply, max 2 lines, auto-fade after 4 s, Arabic (RTL) and Latin text both correct.
- Tool activity: a thin "thought" line ("🔎 searching the web…", "📝 writing to your notes…") — honesty, not decoration.
- Privacy affordance: a mic glyph that is unmistakably on/off, and a cloud glyph whenever audio or text leaves the machine.

## Steps

1. **`protocol.py`** — pydantic models for every UI message (`StateMsg`, `TokenMsg`, `MoodMsg`, `LevelMsg`, `CaptionMsg`, `ToolMsg`, `DegradedMsg`, `ConfirmMsg`), each with a `v` field. Generate TypeScript interfaces (`scripts/gen_ui_protocol.py`) — the JS side is generated, never hand-edited (R5 rule 4).
2. **`EventBridge`** — FastAPI + `/ws`; subscribes to the `EventBus`; throttles `LevelMsg` to 30 Hz and coalesces token deltas every ~50 ms; drops messages if the socket is behind (never blocks the core); binds `127.0.0.1` with a per-run random token in the WS URL.
3. **`OrbWindow`** — pywebview window: `frameless=True, transparent=True, on_top=config`, size ~220×220 (expandable to a 420 px panel with the caption card + a small "last actions" list). Persist position/size in `%LOCALAPPDATA%\atlas\ui.json`. Single instance; if a second launch happens, focus the first.
4. **The orb render.** Canvas2D, 30 fps cap (this is a 2-core laptop — 60 fps would steal from ASR). Layers: soft radial glow → blob (perlin-ish noise deformed by audio level + mood) → inner orbiting particles (think speed) → ring (listening) → caption card (DOM, not canvas — better text rendering incl. Arabic shaping).
5. **Mood rendering.** CSS custom properties driven by `MoodMsg`; blend with `requestAnimationFrame` easing. `prefers-reduced-motion` respected (accessibility: slower, no particles).
6. **Hotkeys & tray.** `keyboard` for global hotkeys (`Ctrl+Alt+M` mute, `Ctrl+Alt+Space` PTT, `Ctrl+Alt+S` stop, `Ctrl+Alt+A` show/hide orb); `pystray` for the tray with live state in the tooltip. Tray must work even if the window is hidden.
7. **Confirmation UI.** When `ConfirmMsg` arrives, the orb turns amber, the caption shows the spoken question text, and a timeout ring counts down (default 10 s → NO). Voice is the primary input; a click on the orb = yes only if you enable it.
8. **Failure isolation.** If WebView2 is missing or the process dies: log, keep running headless, restart the window up to 3 times, and tell the user once (*"UI crashed, running without it — say 'restart interface'."*).
9. **Resource check.** Measure UI process RSS and GPU/CPU use while idle and while speaking; target ≤ 150 MB and ≤ 4 % CPU idle. If WebView2's GPU process is noisy on HD 520, launch with software rendering flags and re-measure.
10. **Polish pass.** Walk every state in the FSM by hand and check the visual matches the meaning; check with the orb at 100 %, 125 % and 150 % DPI scaling (Windows scaling is the classic breaker of frameless windows).

## Classes

`EventBridge` · `WsSession` · `UiProtocol` (pydantic) · `OrbWindow` · `WindowState` · `TrayIcon` · `HotkeyManager` · `ThemeEngine` · `MoodRenderer` (server-side message shaping; the drawing is JS).

## Tests

- Protocol: generated TS matches the pydantic models byte-for-byte in CI (fails on drift).
- Bridge: fake EventBus → assert throttling (≤ 30 Hz level, ≤ 20 token batches/s) and that a stalled socket never blocks publishers.
- UI smoke (playwright/WebView2 not required): open the orb page with a fixture stream and screenshot-compare the 8 states (golden PNGs at a fixed size).
- Resilience: kill the window process → core keeps answering → window restarts.
- DPI: run at 100/125/150 % and assert the window/caption geometry stays correct (manual checklist + screenshots).
- Budget: UI RSS ≤ 150 MB, idle CPU ≤ 4 %, and a full turn adds ≤ 25 ms to core latency.

## Done when

- You can tell Atlas's state and mood across the room without reading text.
- Arabic and English captions both render correctly (bidi and shaping).
- Killing the UI mid-conversation doesn't interrupt Atlas.
- Numbers recorded in `NOTES-L05.md`.

## Pitfalls

- ❌ A dashboard. This is an orb; anything more becomes clutter you'll stop looking at.
- ❌ 60 fps canvas animation on 2 Skylake cores → cap at 30 fps and pause rendering when the orb is hidden.
- ❌ Frameless + transparent window without DPI handling → invisible-on-150 %-scaling windows.
- ❌ Rendering captions in canvas → Arabic shaping/bidi bugs; use DOM.
- ⚠️ Transparent frameless windows on Windows 10 can flicker with some GPU drivers — test early, and keep a fallback "opaque rounded card" theme if it does.
