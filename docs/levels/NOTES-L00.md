# NOTES — L0 (foundation)

Measured numbers, decisions taken, and what still has to be confirmed on the
Dell Latitude 5480. Nothing in this file is estimated; if a cell says "not yet",
it means exactly that.

## Where these numbers come from

The build ran in a Linux container (Python 3.11.2, 3.8 GB RAM, no audio devices,
network limited to PyPI and GitHub). **That is not the target machine.** The
sandbox is useful for correctness and for catching absurd assumptions — it can
never tell us how Atlas behaves on a 2-core Skylake with a SATA SSD. The rows in
"On the laptop" below are the ones that decide whether L0 passes its gate.

## Measured here

| Measurement | Value | How |
| --- | --- | --- |
| `import atlas_core … atlas_ui` | 237 ms | best of 3, cold process |
| `python -m atlas version` | 185 ms | best of 3 |
| `python -m atlas doctor` | 196 ms | best of 3 (budget: < 3 s) |
| `python -m atlas providers` | 179 ms | best of 3 |
| `python -m atlas skills` | 182 ms | best of 3 |
| `pytest` (whole workspace) | 2.0 s | 181 tests |
| `ruff check` | 8 ms | workspace-wide |
| `mypy` | 187 ms | 51 files, strict-ish |
| `python scripts/check_duplicates.py` | 0.5 s | 45-token window, 32 modules |
| Source (shipped) | 5 823 lines | `packages apps`, excluding tests |
| Tests | 1 985 lines | 181 tests |

Replies with no network: the router tries every provider, reports each failure,
then speaks one honest Darija sentence — 2.3 s (five TLS failures plus fallback).

## On the laptop (to fill in after `uv sync`)

```powershell
uv sync --all-packages
uv run python -m atlas doctor
uv run python -m atlas providers --live     # per-provider TTFT, first real number
uv run pytest -q
```

| Measurement | Latitude 5480 | Notes |
| --- | --- | --- |
| `atlas doctor` wall time | | must stay under 3 s or nobody runs it |
| free RAM at idle | | doctor prints it; ~2.0 GB expected |
| `providers --live` TTFT (Gemini) | | first real feel for how the assistant responds |
| `providers --live` TTFT (Groq) | | the fallback that must feel fast |
| `pytest` wall time | | ~2 s here; expect it slower on the SATA SSD |
| first-run disk cost | | repo + venv + `data/`, `logs/`, `models/` |

## Decisions taken while building

- **Providers**: one `OpenAiCompatibleProvider` serves Groq, OpenRouter, Together,
  HuggingFace, Ollama and llama.cpp; Gemini has its own because its wire format
  differs. What they share (SSE decoding, status→error mapping, timeouts) lives
  once in `HttpStreamingProvider` — the duplicate-code gate caught two copies
  before they could drift.
- **Failed providers are data, not crashes.** The router reports who failed and
  falls back; a truncated reply ends with `StreamEnd(reason="truncated")` rather
  than being silently retried, because a half-spoken sentence cannot be unsaid.
- **Keys are never printed**, not even by `doctor` — it reports the *shape* of a
  key when that shape is wrong.
- **`ATLAS_CONFIG`** overrides the config path (used by every CLI test).
- **Duplicate gate**: `scripts/check_duplicates.py` tokenises and strips literals
  before matching, so a wall of strings in a lookup table is not reported as
  copy-paste. Window 45 tokens. `# dup-ok: reason` opts a block out.
- **`DeclaredSkill`**: skills declare name/description/schema as class attributes
  instead of each writing the same `spec()` body.
- **`FakeEngineMixin`**: the three fake engines share one load/unload lifecycle.

## Divergences from the plan (and why)

- **Gemini model**: the plan's config named `gemini-2.0-flash`; that model was
  retired on 2026-06-01, so the shipped config uses `gemini-2.5-flash-lite`
  (free tier: 30 RPM / 1 000 requests per day — the largest free allowance).
  The router treats a model rejection like any other provider failure.
- **CI uses pip for the checks, and `uv build` in its own job.** `uv` is what the
  laptop uses; the check matrix stays pip-based so a broken `uv.lock` cannot hide
  a failing test.
- **`doctor` also prints a hint when a Gemini key does not start with `AIza`** —
  a wrong-shaped key looks fine until the first request fails.

## Known gaps (deliberately left to later levels)

- No audio: `atlas-audio` exposes `list_devices()` and stub engines; nothing opens
  a microphone until L2. `atlas listen` says so.
- No UI window: `atlas-ui` is the protocol plus its TypeScript generator; the orb
  arrives in L5.
- No vault writes from chat: `remember` needs L6 wiring, and unknown voices get
  restricted mode from L4.
- `bench_providers.py` must run on the laptop — the sandbox cannot reach any
  provider, so there are no live latency numbers in this file yet.
