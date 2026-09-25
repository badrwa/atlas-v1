# L10 — Residency (it lives inside the PC)

**Goal:** Atlas starts with Windows, survives crashes, sleep/resume, mic theft and bad networks, manages its own memory budget, updates itself safely, and runs for days without you thinking about it.
**Effort:** ~1 week. **Depends on:** all previous. **Gate:** a 24 h soak test with zero manual intervention and idle RAM ≤ 700 MB.

---

## Deliverables

- Logon autostart via **Task Scheduler** (not the Startup folder), silent (`pythonw`), single-instance mutex, restart-on-failure.
- `Supervisor`: watchdogs for every thread, subprocess and sidecar, with backoff and honest spoken notices.
- `ResourceGovernor` (D15 implementation): free-RAM floors, lease priorities, forced degradation ladder, nightly maintenance.
- `Scheduler`: reminders/timers that survive reboots, quiet hours, weekly study run (L9), monthly vault summarisation (L6).
- Sleep/resume + audio-device-change handling (`WM_POWERBROADCAST`, device hot-plug).
- `Updater`: git pull → lockfile sync → tests → restart, with one-command rollback (and Atlas's own memory of what changed).
- Desktop shortcuts, `atlas doctor --fix`, uninstall path, and a 1-page user guide.

## Steps

1. **Autostart.** `schtasks /create /tn Atlas /sc onlogon /rl highest? …` — **do not** run elevated (UAC breaks app control); use `/sc onlogon` with the current user, `-WindowStyle Hidden`, `pythonw.exe -m atlas run`. Settings: *restart on failure (3×, 1 min)*, *stop if runs longer than: off*, *do not start a new instance* (plus your own mutex anyway).
2. **Single instance + IPC.** Named mutex at startup; a second launch focuses the orb (or prints "Atlas is already running: `<pid>`"). Local IPC via a named pipe or `127.0.0.1` socket for the CLI to talk to the running instance (`atlas say "test"`, `atlas status`, `atlas mute`).
3. **`Supervisor`.** Every long-lived component registers a heartbeat. On death: restart with exponential backoff (max 5 tries), publish an event, and tell the user **once** in their language (*"l'audio tqata3، kanejjed…"*). After N failures: degrade (e.g. wake word off, push-to-talk only) and say what changed — silent degradation is worse than a broken feature.
4. **`ResourceGovernor`.**
   - Sample RSS/available RAM every 5 s (`psutil`), plus a HUD on demand (`"Atlas, chhal bqa f la mémoire?"`).
   - Rules: never load a lease if free RAM < cost + 400 MB floor; evict ASR before loading TTS; evict everything idle > 5 min; in danger (< 500 MB free) force Lean mode, unload sidecars, drop the TTS cache to disk, and speak the degradation.
   - Nightly maintenance at 03:00: rebuild FTS index, vacuum SQLite, prune old logs/caches, warm the cache for the morning.
5. **Sleep/resume & device changes.** Subscribe to power events and `WM_DEVICECHANGE`: on resume, re-open the audio stream, re-check the default device, re-arm KWS, and log a resume line. If the default mic changed, ask once whether to keep the new device. This is the #1 everyday bug for always-on voice assistants.
6. **Mic contention.** Detect "device busy" (Teams/Zoom/OBS/Discord), retry with backoff, and offer PTT mode until it frees up. Store the last known-good device in config and never silently fall back to a webcam mic.
7. **Quiet hours & presence.** `quiet_hours` in `preferences.md` → captions only, no wake chime, quieter voice; optional webcam-free presence heuristic (input idle time) so a timer doesn't announce itself at 03:00, and reminders during quiet hours queue to the morning digest.
8. **Reliability of reminders.** Store in SQLite; on boot, report missed ones in one line; never double-fire; test a reboot with a reminder pending.
9. **Updater.** `atlas update` → `git fetch` → show diff summary in the vault → `uv sync --frozen` → run the test suite → backup current `config.toml` → restart via Task Scheduler → on failure `git checkout <previous-sha> && uv sync --frozen` and tell the user what happened. Never update while a turn is in flight.
10. **Crash forensics.** Structured logs (`logs/atlas-YYYY-MM-DD.log`, rotated, 7-day retention), a `atlas doctor --bundle` command that produces a zip (logs + config + timings + versions, **no keys, no personal notes**) for GitHub issues.
11. **Soak test.** 24 h with normal use: monitor RSS every 10 s (`scripts/soak.py`), assert: no growth trend > 5 %/hour, zero crashes, wake word still responds after 24 h, zero false wakes overnight, ≤ 700 MB idle, ≤ 5 % idle CPU, and correct behaviour across 2 reboots and 3 sleep cycles.
12. **Final wrap-up.** `README.md` user guide (install, first run, folder contract, permissions, privacy statement incl. what leaves the machine), `docs/security/` (red-team results), `CHANGELOG.md`, a printable 1-page cheat sheet of Darija + English commands, and the L9 publish step for v1.0.

## Classes

`StartupRegistrar` · `InstanceGuard` · `IpcServer` · `Supervisor` · `ComponentHealth` · `ResourceGovernor` · `LeasePolicy` · `MaintenanceJob` · `PowerEventListener` · `AudioDeviceWatcher` · `Scheduler` · `Updater` · `CrashReporter` · `SoakMonitor`.

## Tests

- Autostart: task exists, runs at logon, single instance holds, second launch focuses.
- Supervisor: kill each component (including the TTS/LLM sidecars) → restarts, user told once, no crash loop.
- Governor: simulate low RAM → correct eviction order, degraded mode spoken, no OOM.
- Power: simulated power events → audio re-init; unchanged when irrelevant.
- Reminders: reboot with pending reminder → fires once, correctly.
- Updater: injected failing test → automatic rollback to previous SHA with a spoken explanation.
- Logs: rotation, no secrets, bundle excludes personal data (assert by scanning the zip).
- Soak: `scripts/soak.py` thresholds enforced; report committed to `docs/levels/NOTES-L10.md`.

## Done when

- Reboot → log in → *"Atlas, bonjour"* → answer, with no window flashing and nothing clicking.
- 24 h soak: zero crashes, idle RAM ≤ 700 MB, no leak, wake word alive at hour 24, zero overnight false wakes.
- Sleep/resume ×3 and a mic handover (open Teams, close it) handled without user action.
- A deliberate crash is recovered and *explained*, not hidden.
- The README's quickstart gets a friend to a running Atlas on a similar machine without asking you anything.

## Pitfalls

- ❌ Startup-folder autostart → a console window flashing on every login. Use Task Scheduler + `pythonw`.
- ❌ Running the whole thing elevated → UAC-elevated target apps become uncontrollable, and you widen your blast radius. Don't.
- ❌ Silent restarts/degradation → users lose trust; speak what changed.
- ❌ Auto-updating without tests → the fastest way to wake up to a dead assistant.
- ⚠️ Sleep/resume and device changes are where always-on systems actually break. Budget real time here.
- ⚠️ Windows Defender real-time scanning can slow first loads — exclude the venv, model folder and vault folder (a documented, visible decision).
