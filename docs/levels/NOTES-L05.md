# NOTES-L05 — The face (the orb, built and verified, and what only your laptop can prove)

**Status:** implemented, tested, committed; sandbox-verified end to end over a real
socket. Everything that needs a **display, a GPU driver or a DPI setting** is marked
**[laptop]** and listed in §5.

**Gate (LEVEL-05):** every state change appears in < 100 ms, the UI process stays
≤ 150 MB, and the core is unaffected if the UI dies.

---

## 1. What was built

| File | What it is |
|---|---|
| `packages/atlas-ui/src/atlas_ui/protocol.py` | the 8 pydantic messages, `PROTOCOL_VERSION`, `MOOD_HUES`, `TOPIC_MESSAGES` (the translation table), `SILENT_TOPICS`, `typescript()` + `javascript()` |
| `packages/atlas-ui/src/atlas_ui/theme.py` | meaning in Python: `STATE_MOTIONS`, `OVERLAY_STATES`, `ORB_STATES`, `MOOD_HUE*`, `clamp_caption`, `is_rtl`, `TOOL_PHRASES`, `CAPTION_FADE_S` |
| `packages/atlas-ui/src/atlas_ui/hub.py` | `UiHub` + `UiSession`: throttling (levels at 30 Hz, token batches at 20 Hz), the growing-caption accumulator, the hello snapshot, `policy_topics()` |
| `packages/atlas-ui/src/atlas_ui/bridge.py` | `EventBridge`: FastAPI app, token-gated `/`, `/orb/{name}`, `/health`, `/ws`; the bus→hub pump; `UI_TOPICS` **derived** from `TOPIC_MESSAGES`; `ui-endpoint.json` handshake |
| `packages/atlas-ui/src/atlas_ui/window.py` | `OrbWindow` (frameless, transparent, on-top, drag-anywhere, position memory), `run(backend)` (GUI on the main thread), `WindowState`, `InstanceLock`, `TrayIcon`, `HotkeyManager` |
| `packages/atlas-ui/src/atlas_ui/orb/` | `index.html` + `orb.css` (177 lines) + `orb.js` (264 lines) + the two generated files — no framework, no bundler |
| `packages/atlas-ui/src/atlas_ui/assets.py` | `check_assets()`: files present, delimiters balanced, every `kind` the JS handles is declared, generated values match the models |
| `scripts/gen_ui_protocol.py` | writes `orb/protocol.ts` **and** `orb/protocol.js`; `--check` fails on drift |
| `scripts/check_ui_assets.py` | the gate for the four shipped assets |
| `apps/atlas/src/atlas/cli.py` | `atlas ui run/serve/status/protocol` (+ the `ui-protocol` alias), `--demo`, `--opaque`, `--no-attach`; **`atlas listen --ui`** |
| `apps/atlas/src/atlas/doctor.py` | `orb` row + `ui · bridge` / `ui · window` rows |
| `packages/atlas-audio/src/atlas_audio/loop.py` | publishes `state.changed`, `audio.level`, `speaker.matched` — one writer (`_set_state`), fire-and-forget via `EventBus.emit` |

**The one-rule design.** The core owns the vocabulary (`LoopState` in the ears,
the FSM in the core), the protocol owns the translation, and the orb owns the
pixels. Nothing in `atlas-ui` imports `atlas-audio`: the orb must install on a
machine with no audio stack, and the app-level test checks the union of the two
vocabularies where both are visible.

## 2. Decisions worth remembering

1. **The orb is a viewer, not a component.** Two processes: `atlas listen --ui`
   serves the bridge for its conversation and writes `data/ui-endpoint.json`;
   `atlas ui run` finds that file, probes `/health`, and opens a window on
   *somebody else's* bus. Kill the window — Atlas keeps answering. Kill Atlas —
   the window shows "bridge unreachable" and calls nothing broken.
2. **The GUI loop owns the main thread.** That is a WebView2 rule, not a style
   choice, so the bridge (or the whole voice loop) runs in pywebview's worker
   thread via `webview.start(backend)`. The old shape created the window and
   never started a message loop, which on Windows is an invisible window.
3. **Captions are the growing reply, not the batch.** The first delta of a reply
   is flushed at once (the orb looks alive from the first word); after that the
   caption is re-sent as the sentence grows. Sending only new deltas made the
   card flicker through fragments (`"lam, kifash"`) instead of reading.
4. **A meter is not a log.** A late audio frame describes the past: the newest
   value always replaces the pending one, and `tick()` flushes it so the orb
   settles instead of freezing mid-syllable.
5. **Levels are fire-and-forget, and cheap.** 30 Hz cap in the hub, `TIME_EPSILON`
   against float-clock equality, `stats.throttled/coalesced/dropped` visible in
   `atlas ui status`. A slow socket drops frames; it never stalls a turn.
6. **Silence is declared.** `SILENT_TOPICS` names the events the orb deliberately
   does not draw, so a new core event is a failing test rather than a feature
   nobody can see.
7. **Errors are visible.** `system.error` was in the core and in neither table:
   a transient error now flashes the orb red (`state: error`) and the next state
   change puts it back; a `fatal` one leaves the amber "degraded" chip on with
   the failing part's name. The text stays in the log — a 420 px card is not a
   stack trace.
8. **Two generated artifacts, two consumers.** `protocol.ts` is for editors,
   `protocol.js` is what WebView2 actually loads (no build step allowed). Both are
   written only by `scripts/gen_ui_protocol.py`, and `--check` fails on drift.
9. **DOM captions, canvas blob.** Arabic shaping and bidi come free in the DOM;
   the blob is where per-pixel work belongs. `is_rtl` in Python decides the text
   direction so the JS never guesses.
10. **Honest glyphs.** The mic glyph is off while Atlas speaks (half duplex is
    visible, not hidden), the cloud glyph appears when text leaves the machine,
    and the amber confirm ring is a countdown you can see across the room.

## 3. Sandbox verification (no display, no GPU, no microphone — which is the point)

| Check | Result |
|---|---|
| `pytest` (with the extras) | **651 passed** (581 → 651: +70 L5 tests) |
| `pytest -m "not live and not hardware"` in a bare venv (CI, no extras) | **639 passed, 12 skipped** — the same command that was red since L4 |
| `ruff check .` | clean |
| `mypy packages apps` | **101 source files**, no issues |
| `check_import_rules` / `check_no_torch` / `check_duplicates` | ✓ (`window=45 tokens`, 57 modules) |
| `gen_ui_protocol.py --check` | ✓ both files byte-match the pydantic models |
| `check_ui_assets.py` | ✓ 4 files, protocol v1 |
| live WS run (uvicorn + a real client) | hello → snapshot (12 states) → `listening` → level → caption `"Salam"` → `"Salam sahbi"` → transient `error` → fatal `degraded(mode="error")` |
| HTTP token gate | bad token → 403, good token → the page with `__TOKEN__` substituted, `/orb/../../etc/passwd` → 404 |
| `atlas ui serve --demo` (9 s) | clean, no traceback, endpoint file removed on Ctrl+C |
| `atlas ui status` | bridge ready; window/tray/hotkeys rows name the missing extra and the install line |
| `atlas ui run` with no WebView2 | window row explains itself, bridge still served, URL printed — never a crash |
| `atlas listen --ui` wiring | the bridge and the loop share **one** bus; endpoint advertised; cleaned up afterwards |

## 4. Bugs the tests caught (each one would have shipped silently)

1. **`socket: WebSocket` became a required *query parameter*.** With
   `from __future__ import annotations`, a function-local
   `from fastapi import WebSocket` leaves the annotation unresolvable at module
   scope, so FastAPI treated `socket` as a query arg and every connection died
   with close code 1008 (`missing query param 'socket'`). `bridge.py` now keeps
   module-level optional imports (`Any` placeholders + `FASTAPI_ERROR`).
2. **The window was created and never started.** `supervise()` called
   `webview.create_window()` but nothing called `webview.start()`: invisible
   window, no error, on the one platform that matters. `OrbWindow.run(backend)`
   now owns the main thread and runs the bridge beside it.
3. **`system.error` reached the hub and nobody else.** `UI_TOPICS` was a second,
   hand-typed list; the bridge never subscribed to the error topic. It is now
   derived from `TOPIC_MESSAGES`, and a test asserts the two sets are equal.
4. **`system.error` had no policy at all** — the core could publish an error the
   orb would ignore forever. Found by a vocabulary test that walks every dataclass
   in `atlas_core.events` against the protocol table.
5. **The token throttle lost the first word of a reply.** Counting the first delta
   against the batch clock delayed `"Salam"`, so the orb stayed silent for the
   50 ms that mattered most. First delta flushes immediately.
6. **Float clocks compared exactly.** `1/30` intervals fire *at* the boundary:
   `clock.advance(1.0 / LEVEL_HZ)` produced no message. `TIME_EPSILON = 1e-6`.
7. **`stats.emitted` was counted per message, not per batch** — the number in
   `atlas ui status` was the sum of a different quantity than `received`.
8. **Two orb states had no motion.** The loop says `idle`/`stopped`, the face says
   `dormant`: unmapped states fell back to "breathing" silently. `STATE_MOTIONS`
   now covers the union, and the app-level test walks the real `LoopState`.
9. **The orb's asset checker only saw direct references.** `protocol.js` is loaded
   *by* `orb.js`, so the first version would have shipped a 404 on the import the
   browser makes. It now follows the real load graph.
10. **Four tests assumed the `[local]` extra.** `pytest` in CI installs the
    workspace without extras (that is the point: the core must not need numpy or
    fastapi), and three L2 ASR tests plus one frames test imported numpy anyway.
    They now `importorskip` — this had been making the whole CI job red since L4.
11. **A dead `data/ui-endpoint.json` hijacked the orb.** Written by a bridge that
    was killed (Task Manager, power loss), it pointed at a port nobody owned.
    `_live_endpoint()` probes `/health` and ignores stale files; a crashed bridge
    cannot put a mute orb on your screen.

## 5. What the laptop must prove **[laptop]**

Nothing here needs the cloud. Do it in this order.

```powershell
# 1. the extras (the orb is useless without a window)
pip install -e "packages/atlas-ui[orb]"        # pywebview + fastapi + uvicorn
pip install -e "packages/atlas-ui[tray,hotkeys]"   # pystray + keyboard

# 2. does the machine think it can draw?
atlas ui status          # bridge/window/tray/hotkeys must all be ready
atlas ui serve           # bridge only: open the printed URL in Edge first

# 3. the visual pass, no microphone: every state, in order, every ~2.4 s
atlas ui serve --demo
#    watch it once with the orb small, once expanded (the panel opens on captions)
#    → does "thinking" look like thinking?  Does the red flash read as an error?
#    → do the Arabic captions join their letters and read right-to-left?

# 4. the real thing: conversation behind the orb (two windows)
#    window A:
atlas listen --ui
#    window B (attaches to A's conversation; leaves A running when you close it):
atlas ui run
#    - say "atlas, chno ljaw?" → the orb must leave "listening" the moment you
#      stop talking, and the caption must appear with the first word, not the
#      first sentence
#    - while Atlas speaks: the mic glyph is off (that is half duplex, visibly)
#    - drag the orb; close it; reopen with `atlas ui run` → it comes back where
#      you left it, and the conversation in window A never noticed

# 5. numbers (paste into §6)
atlas ui run             # idle, window open, 1 minute
#    Task Manager → Details → "pythonw.exe"/"python.exe" (the orb process) and the
#    WebView2 child processes: sum them, write RSS in §6.  Budget ≤ 150 MB.
#    Then watch CPU at idle (≤ 4 %) and while speaking (front page of the laptop
#    is fine: the 30 fps cap plus the tick should keep it flat)
#    Also: a full turn adds ≤ 25 ms — `atlas listen` prints turn timings

# 6. DPI (Windows scaling is the classic breaker of frameless windows)
#    Settings → Display → Scale: 100 % → 125 % → 150 %, sign out/in between
#    at each: the orb must stay round, the caption card must stay readable,
#    the window must not jump to the top-left corner, and dragging must not
#    drift.  Screenshot each step.
#    If a transparent window flickers on the Intel HD 520 driver:
atlas ui run --opaque    # the rounded-card theme: same orb, opaque background

# 7. the failure path (the part of the gate nobody tests on purpose)
#    while a conversation is running, Task Manager → kill the orb process
atlas listen --ui        # (window A) → must keep answering, no error, no restart
atlas ui run             # bring the face back, same conversation
```

## 6. [laptop] results

```
$ atlas ui status
(paste)

$ atlas ui serve --demo
(the 8-state notes: which motion did not match its meaning?)

DPI 100 %:  (screenshot / ok?)
DPI 125 %:  (screenshot / ok?)
DPI 150 %:  (screenshot / ok?)
transparent window: stable / flickers → opaque fallback used?

$ atlas invoke ... / Task Manager
orb process RSS idle:        ___ MB   (budget ≤ 150)
WebView2 children total:     ___ MB
CPU at idle:                 ___ %     (budget ≤ 4)
CPU while speaking:          ___ %
turn overhead with the orb:  ___ ms    (budget ≤ 25)
state change → visible:      ___ ms    (budget ≤ 100; watch `listening`)

window position remembered after reopen: yes / no
killing the orb mid-conversation: Atlas kept talking: yes / no
verdict: (your words)
```

## 7. Known limits — say these out loud, do not paper over them

- **WebView2 is the renderer.** It ships with Windows 10/11 (Edge), but a machine
  with Edge removed gets a bridge and no window: `atlas ui serve` in a browser is
  the fallback, and `atlas ui status` says so before you start.
- **The orb is a viewer with no write path.** Nothing in the page can mute,
  confirm, forget a speaker or touch the vault: it draws and it displays. The
  voice is the interface, and the tray/hotkeys are the only local controls.
- **Transparency is driver-dependent.** Frameless + transparent can flicker on
  older Intel drivers; `--opaque` is a real theme, not a debug flag.
- **One instance, by design** (`data/ui.lock`). A stale lock from a hard kill is
  stolen if the pid is gone — but two *live* orbs cannot be opened.
- **The demo is not a test of the hardware.** `--demo` proves the messages are
  right and the animations run; only §5's conversation proves the ears, the brain
  and the face agree about *when*.
- **RSS numbers in §6 are the honest ones.** WebView2's child processes are the
  bulk of the cost on an 8 GB laptop; if the sum is over budget, the fix is the
  opaque theme and a shorter caption fade before it is a code change.
