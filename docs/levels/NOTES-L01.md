# NOTES — L1 (brain)

Measured numbers, decisions taken, and the parts of the gate that need the
Latitude 5480. Nothing here is estimated: if a cell says "not yet", it means the
number has not been measured yet.

## Where these numbers come from

Measured in a Linux container (Python 3.11.2, 3.8 GB RAM, no network to any
provider). **The one number that decides L1's gate — first token < 1.5 s — cannot
be measured here**, because the sandbox cannot reach Groq or Gemini. The
pipeline overhead *can*, and it is small enough that the whole gate is decided by
the model's time-to-first-token, not by Atlas.

## Measured here (fake providers, so the model costs ~0)

| Measurement | Value | Notes |
| --- | --- | --- |
| Pipeline overhead, fast path | **0.18 ms/turn** | median of 40, warm |
| Pipeline overhead, structured | **0.20 ms/turn** | same turn, envelope path |
| Envelope parse + validate | **4 µs** | 2 000 iterations |
| System prompt (Darija) | 2 179 chars ≈ **619 tokens** | the largest fixed cost in a request |
| JSON schema sent to providers | 768 chars ≈ 200 tokens | structured turns only |
| Context budget | 1 200 tokens | enforced; oldest turns dropped first |
| `pytest` (whole workspace) | **267 tests in 3.2 s** | includes CLI end-to-end |
| `check_duplicates` | 35 modules | window 45 tokens |

The plan's budget is "< 50 ms of our own overhead per turn". Measured: 0.2 ms —
250× under. Everything else a user feels is the provider.

## On the laptop (the rows L1's gate actually needs)

```powershell
uv run python -m atlas providers --live     # per-provider TTFT for a 40-word answer
uv run python -m atlas chat
```

| Measurement | Latitude 5480 | Target |
| --- | --- | --- |
| TTFT, Gemini 2.5 Flash-Lite | not yet | < 1.5 s |
| TTFT, Groq llama-3.3-70b | not yet | < 1.5 s |
| Full 40-word answer | not yet | < 5 s |
| TTFT, local qwen3-1.7b (bunker) | not yet | no target — it is the offline path |
| Cold start, llama-server | not yet | ~12 s expected; measured once L2/L10 wire it up |

Run `python scripts/bench_providers.py` once the keys work: it prints p50/p95 per
provider for a **fixed** prompt, which is the only way to compare them honestly.

## Decisions taken while building

### Two reply paths, one session

- **Fast path (default):** stream text tokens straight through. Lowest possible
  time-to-first-token — this is what being talked to feels like.
- **Structured path (`--structured`, `/structured`):** one JSON envelope with
  `reply`, `language`, `emotion`, `mood_delta`, `followup`. It costs the model
  extra tokens, so it is opt-in and the orb (L5) can enable it per situation.

`ChatSession._envelope_or_repair()` gives a malformed envelope **exactly one**
repair attempt, then speaks the salvaged text anyway. A salvaged reply reports
no emotion on purpose: it never claimed one.

### Tools are deliberately *not* in the L1 schema

Listing `tool_calls` on every turn made small models invent calls they could not
execute. The field returns in L7, together with the registry that can actually
run — and permission-check — a call. The `ToolIntent` model already exists for
that day.

### Mood is local and predictable

`MoodEngine` takes the model's own read when the envelope is available, and a
small marker table otherwise, then decays towards calm. It answers exactly two
questions: *what colour is the orb* and *is a joke welcome*. L8 grows it; the
contract (`label`, `energy`, `warmth`, `humor_allowed`, `intensity`) is stable.

### Language belongs on the request

`LlmRequest.language` was added after a test caught the offline fallback
apologising in Darija right after the owner said "speak English". The turn's
language is a property of the turn, not of the router that was constructed at
startup.

## Bugs this level's tests caught (and fixed)

1. **Silence counted as success.** A provider that streamed nothing and raised
   nothing was treated as a valid answer — Atlas would simply say nothing. The
   router now treats an empty stream as a failure and moves on.
2. **Darija apology after a language switch** — see "Language belongs on the
   request" above.
3. **Emotion was not actually validated.** The JSON schema said `enum`, but the
   validator accepted any string, so `"emotion": "extremely warm, like tea"`
   could have reached the orb. `ReplyEnvelope` now validates both `emotion` and
   `mood_delta` against the same tables the schema advertises.
4. **The markdown rule was stated twice** in the Darija prompt — once
   universally, once in the dialect pack. The golden prompt made it visible;
   formatting rules now live only in `persona.jinja`.

## Prompt goldens

`packages/atlas-mind/tests/golden/persona_{darija,en_gb}.txt` are the system
prompts verbatim. The persona is the product, so changing it must be a visible
diff:

```bash
pytest packages/atlas-mind -q --update-goldens
```

The golden set also proved its worth immediately: it is how the duplicated
markdown rule and the doubled en-GB rules were found.

## Style rules that are now tests, not hopes

- Darija pack must name Modern Standard Arabic as the thing to avoid, and must
  carry real greetings/acknowledgements.
- en-GB pack must ban, by name: `awesome`, `reach out`, `dude`, `my bad`, plus
  `as an AI`.
- Every language pack stays complete enough to use (label, humour, rules,
  examples, script).

## Known gaps (deliberately left to later levels)

- Live cloud numbers: no provider is reachable from this sandbox. The laptop run
  fills in the table above.
- Local model: `llama-server` and a GGUF are not installed here, so the sidecar
  logic is tested through an injected process, not a real one. The plan's path is
  `vendor/local-llm/llama-server.exe` + `models/*.gguf`; `atlas providers` says
  which of the two is missing.
- Interruption mid-reply ("stop talking") needs a voice, so it lands with L3.
