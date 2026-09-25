# L0 — Foundation

**Goal:** the skeleton that makes levels 1–10 pleasant. No user-visible behaviour — and that's deliberate.
**Effort:** one weekend. **Depends on:** nothing. **Gate:** `python -m atlas doctor` passes and CI is green.

---

## Deliverables

- A **uv workspace** with six installable packages: `atlas-core`, `atlas-audio`, `atlas-mind`, `atlas-obsidian`, `atlas-skills`, `atlas-ui`.
- `atlas-core` implementing the contracts from `docs/architecture/ARCHITECTURE.md` (D4): `Sense`, `Engine`, `LlmProvider`, `SpeechRecognizer`, `SpeechSynthesizer`, `WakeWordEngine`, `SpeakerVerifier`, `Skill`, `Store`, plus `EventBus`, `InteractionFSM`, `ResourceLease`, `Config`, `Timings`, `DI`.
- **Fakes** for every contract so CI runs with no mic, no key, no GPU.
- `python -m atlas doctor` — self-checks hardware, audio devices, keys, vault path, disk, RAM headroom.
- Test harness with **contract suites** (one suite per ABC, run against every implementation).
- CI: ruff + mypy + pytest on Windows and Ubuntu + `uv build` + the import-rule test.
- `docs/levels/NOTES-L00.md` with your measured baselines.

## Why this level exists

Every "no duplicated code" rule (R5) is cheap *now* and expensive later. Contracts + DI + events + one pipeline shape = the whole architecture of a 10-level project, priced at one weekend. Skip it and L7 becomes a rewrite.

---

## Steps

1. **Install the toolchain.** `winget install astral-sh.uv`, Python 3.12 (64-bit, PATH), Git for Windows, VS Code + Ruff/mypy extensions. Verify: `uv --version`, `python -c "import platform;print(platform.machine())"` → `AMD64`.
2. **Scaffold the workspace.**
   ```powershell
   uv init --lib atlas-core --package
   # repeat for atlas-audio, atlas-mind, atlas-obsidian, atlas-skills, atlas-ui
   # root pyproject.toml → [tool.uv.workspace] members = ["packages/*"]
   uv sync
   ```
   Layout: `packages/<name>/src/<name>/…`, tests beside each package (`tests/`), `apps/atlas/` for the composition root.
3. **Write `atlas-core/config.py` as pydantic v2 models** — `AppConfig`, `AudioConfig` (device names, sample rates), `WakeConfig` (threshold, pre-roll), `AsrConfig` (provider order, model ids, lease TTL), `TtsConfig` (voice per language, cache dir), `MindConfig` (provider order, timeouts, token budgets), `ObsidianConfig` (vault path, folder contract, git), `GovernorConfig` (RAM floors), `UiConfig`. All loaded from `config.toml` + `.env`, with `--profile lean|bunker` overlay.
4. **`atlas-core/contracts.py`** — the ABCs. Keep them small and boring: 2–5 methods each. Add a docstring stating the *contract test* that every implementation must satisfy.
5. **`atlas-core/events.py`** — typed events via pydantic: `WakeDetected`, `InteractionStateChanged`, `TokenStreamDelta`, `MoodChanged`, `AudioLevelChanged`, `ToolStarted`, `ToolFinished`, `DegradedModeChanged`, `SpeakerMatched`, `SessionLogWritten`, `TimingRecorded`. Async `EventBus` with per-subscriber queues and `slow_subscriber` warnings.
6. **`atlas-core/fsm.py`** — the interaction FSM (D6) as a table of `(state, event) -> state`, with `on_enter/on_exit` hooks and an FSM transition unit test for **every** edge, including timeouts and defaults.
7. **`atlas-core/resources.py`** — `ResourceLease` (D15): `acquire(engine, cost_hint)`, `release()`, TTL eviction, veto callback from the governor, RSS logging on load/unload.
8. **`atlas-core/di.py`** — a 40-line injector: `container.register(iface, factory, singleton=True)`, `container.resolve(iface)`, plus `TestContainer` overrides. No framework.
9. **`atlas-core/timings.py`** — `timings.jsonl` writer: `{ts, session, turn, wake_ms, asr_ms, ttft_ms, tts_first_ms, total_ms, provider, mode, asr_model, tts_engine}`.
10. **Fakes** in `atlas-core/fakes.py`: `FakeMic` (replays WAVs), `FakeSpeaker` (writes WAVs), `FakeProvider` (scripted token streams incl. tool calls and 429s), `FakeRecognizer`, `FakeSynthesizer`, `FakeVault` (tmp dir), `FakeClock`.
11. **Contract suites** in `packages/atlas-core/tests/contracts/`: `test_llm_provider_contract.py`, `test_recognizer_contract.py`, `test_synthesizer_contract.py`, `test_skill_contract.py`, `test_store_contract.py`. Each is parametrised over implementations registered in a fixture → adding an engine automatically inherits ~15 tests.
12. **`apps/atlas/__main__.py`** — CLI with `doctor`, `chat` (stub), `listen` (stub), `devices`, `bench`. `doctor` checks: Python arch, free RAM, disk, WASAPI devices, WebView2 present, keys present, vault path writable + git repo, model folder, config schema validity, import-rule test.
13. **CI** (`.github/workflows/ci.yml`): matrix `windows-latest` + `ubuntu-latest`, `uv sync --frozen`, `ruff check`, `ruff format --check`, `mypy packages`, `pytest -q`, `uv build`, plus `scripts/check_import_rules.py`.
14. **Pre-commit**: ruff + ruff-format + a secret scanner (`gitleaks`) so API keys can never be committed.
15. **Baseline measurements** → `docs/levels/NOTES-L00.md`: idle RAM with nothing running, cold `import atlas` time, Python startup, free RAM after disabling startup apps, whether memory is single- or dual-channel (Task Manager → Performance → Memory → *Speed* / *Slots used*).

## Classes you should have at the end

`Config` (pydantic) · `Container` · `EventBus` · `Event` tree · `InteractionFSM` · `ResourceLease` · `ResourceGovernor` (stub) · `TimingsRecorder` · the 9 ABCs · `Fake*` × 8 · `Doctor` + individual `Check` subclasses (`RamCheck`, `AudioCheck`, `VaultCheck`, `KeyCheck`, `WebViewCheck`, `DiskCheck`).

## Tests

- FSM: all transitions + illegal events raise.
- DI: singleton vs transient, override in tests, no import cycles.
- EventBus: publish/subscribe, subscriber crash isolation, backpressure.
- Config: TOML + env + profile overlay precedence; invalid config fails with a readable message.
- Contract suites run against fakes and pass.
- Import rules: `check_import_rules.py` fails if `atlas-ui` imports `atlas-mind`, etc.

## Done when

- `uv sync && python -m atlas doctor` prints a green report on your Latitude (audio devices named, vault path writable, free RAM ≥ 2 GB).
- `pytest -q` green locally and in CI on both OSes.
- `ruff` + `mypy` clean, `duplicate-code` check clean.
- `docs/levels/NOTES-L00.md` has real measured numbers.

## Pitfalls

- ❌ Building a "framework". 40 lines of DI is enough; Django-sized abstractions here would kill the project.
- ❌ Adding `torch` "just for tests". Never.
- ❌ Letting the FSM own I/O. It's pure: states in, states out.
- ❌ Writing secrets into `config.toml`. Keys go in `.env` only.
- ⚠️ Windows + `asyncio`: use `ProactorEventLoop` (default) and `asyncio.to_thread` for blocking calls; don't fight it.
- ⚠️ Keep `atlas doctor` fast (< 3 s) or you'll stop running it.
