# L1 — Brain (text first, voice later)

**Goal:** a working text Atlas that speaks **Darija by default** and **British English** on request, streams its answer, falls back between providers, and records timings. No microphone yet.
**Effort:** ~1 week. **Depends on:** L0. **Gate:** `python -m atlas chat` holds a coherent in-character conversation, first token < 1.5 s, survives the network being unplugged.

---

## Deliverables

- `LlmProvider` implementations: `GeminiProvider`, `GroqProvider`, `LlamaCppProvider` (HTTP `llama-server`), `FakeProvider`.
- `LlmRouter` with decorators: `TimedProvider`, `RetryingProvider`, `QuotaGuardedProvider`, `FallbackChain`.
- `Persona` + prompt templates (`prompts/persona.jinja`, `language_policy.jinja`, `dialect_darija.jinja`, `dialect_en_gb.jinja`).
- `LanguageRouter` with Darija text normalisation.
- Structured output: `{reply, language, emotion, mood_delta, tool_calls, needs_followup}` from one call.
- CLI chat REPL with language switching, streaming print, per-turn timings.

## Why text first

The brain is where 80 % of the product's quality lives (persona, dialect, tone, structured output, fallbacks). Debugging that through a microphone and a UI is masochism. Text mode also gives you a permanent test harness for every later level.

---

## Steps

1. **Keys.** Google AI Studio key (Gemini free tier) + Groq key → `.env`. Add `python-dotenv` load in `config.py`. Verify with a 3-line script that each provider answers `ping`.
2. **`providers/base.py`** — freeze the interface:
   ```python
   class LlmProvider(ABC):
       name: str
       @abstractmethod
       def stream(self, messages: list[Message], tools: list[ToolSchema] | None = None,
                  schema: type[BaseModel] | None = None) -> AsyncIterator[LlmEvent]: ...
       @abstractmethod
       async def health(self) -> HealthReport: ...
   ```
   `LlmEvent` = `TextDelta | StructuredResult | ToolCallRequest | Usage | ErrorEvent`.
3. **`providers/gemini.py`** — `google-genai` SDK. Model: **2.5 Flash-Lite** (default) → **2.5 Flash** (harder questions) → Gemini Live (L2/L3 audio, later). Use structured output (`response_schema`) for the JSON envelope; stream text separately.
   *Caveat:* confirm current model ids in the console — Free tier is Flash/Flash-Lite only, Pro is paid-only since April 2026.
4. **`providers/openai_compatible.py`** — one class, two uses: Groq (`base_url=https://api.groq.com/openai/v1`, models `gpt-oss-20b` / `qwen3-27b` / `llama-3.3-70b-versatile`) **and** the local llama.cpp server. Zero duplicated HTTP code (R5).
5. **`providers/llama_cpp.py`** — wrapper that ensures the sidecar is running (start `vendor/local-llm/llama-server.exe -m qwen3-1.7b-q4.gguf --port 8123 -t 3`), health-checks `/health`, and leases it via `ResourceLease`. Bunker mode only.
6. **Decorators** (write once, use everywhere):
   * `TimedProvider` → emits `TimingRecorded(ttft_ms, total_ms, tokens)`.
   * `RetryingProvider` → exponential backoff, max 2 retries, retry only on 429/5xx/timeout.
   * `QuotaGuardedProvider` → token-bucket per provider per day from config; on exhaustion, mark unhealthy and notify once.
   * `FallbackChain` → `[Gemini, Groq, (Bunker: llama.cpp), CannedReply]`, logging who answered.
7. **`language_router.py`** — decide language per utterance:
   - Explicit command wins: *"b darija" / "b l'ingliziya" / "speak English" / "b English"*.
   - Script heuristic: Arabic script → `ar-MA`; Latin → `en-GB` unless strong Darija markers (Arabizi: `3`, `7`, `9`, `kh`, `gh`, `ch`, `wa`, `dyal`, `bezzaf`, `mzyan`, `daba`, `safi`, `wach`).
   - Sticky: once switched mid-conversation, stay until switched back or a topic change is detected.
   - `DialectPack` objects (data, not code) hold per-language style: how to greet, how to joke, taboo topics, politeness forms, numbers/units convention.
8. **Darija normalisation (`darija.py`)** — Darija has no standard orthography, so normalise, don't fight: strip diacritics, unify alef forms, map Arabizi digits (`3→ع`, `7→ح`, `9→ق`) **as a lookup for matching only** (never rewrite the user's meaning), collapse repeated letters, and keep a per-user lexicon file (`50_Atlas/darija-lexicon.md`) that the ASR post-processor and the LLM prompt both read. This file is how accuracy improves over months.
9. **`persona.py` + prompts.**
   - Voice: a friend who's known you for years — warm, direct, funny, never fawning. British English pack: understatement, dry humour, "have a go", "bits and bobs", "sorted"; avoid Americanisms explicitly (no "reach out", "awesome", "dude").
   - Darija pack: real Darija patterns, Moroccan humour (self-deprecating, teasing, tea/football/traffic references *only when relevant*), Latin/Arabic script per user preference (`preferences.md`).
   - Hard rules: short spoken sentences; no markdown/lists/emoji; never claim an action it didn't take; "I don't know" is allowed and preferred to invention; one clarifying question instead of a guess.
10. **Structured output envelope** (one call, all metadata):
    ```json
    {"reply": "...", "language": "ar-MA", "emotion": "amused",
     "mood_delta": "warm", "tool_calls": [], "followup": false}
    ```
    Validate with pydantic; on schema failure, one silent repair retry, then fall back to plain text.
11. **`context.py` (stub for L6)** — `ContextBuilder` with a hard token budget: persona (fixed) → preferences → recent turns (windowed) → recalled facts (stub now) → current turn. Cap ~1200 tokens total so TTFT stays low.
12. **CLI** (`python -m atlas chat`): streaming print, `/lang`, `/provider`, `/timing`, `/mood`, `/quit`. Colour-coded tokens, timings table after each turn.
13. **Latency work.** Measure TTFT per provider/model for a 40-word answer. Try: Flash-Lite vs Flash, Groq gpt-oss-20b vs qwen3-27b, `max_output_tokens` cap, prompt length trimming. Pick defaults from **your** numbers.
14. **Robustness.** Kill the network mid-turn → Atlas must say (in the current language) that the cloud is unreachable and offer the local path. Simulate 429 → fallback chain must switch silently and note it once.

## Classes

`Message` · `LlmEvent` tree · `LlmProvider` (ABC) · `GeminiProvider` · `OpenAiCompatibleProvider` · `LlamaCppProvider` · `ProviderDecorator` + 4 subclasses · `FallbackChain` · `QuotaState` · `HealthReport` · `LanguageRouter` · `DialectPack` · `DarijaNormalizer` · `Persona` · `ContextBuilder` · `ChatSession` · `ChatRepl`.

## Tests

- Contract suite on all four providers (fake included).
- Language routing table tests (~30 cases incl. Arabizi, code-switching, explicit switches).
- Darija normalisation: idempotence, diacritics, Arabizi mapping, no data loss.
- Persona: golden-prompt snapshots — a change in the system prompt must be a deliberate diff.
- Fallback: 429 → next provider; all down → canned reply in the right language; timeouts bounded.
- Token budget: context never exceeds the configured cap.
- Latency smoke: mocked provider, assert the pipeline adds < 50 ms overhead.

## Done when

- `python -m atlas chat` gives a genuinely good Darija conversation; `speak English` switches mid-conversation to British English with British vocabulary.
- First token < 1.5 s, full 40-word answer < 5 s, on your connection.
- Unplug the network mid-sentence → honest fallback, no hang, no crash.
- `timings.jsonl` shows provider per turn; `docs/levels/NOTES-L01.md` records your measured TTFT per model.

## Pitfalls

- ❌ Prompt-building with f-strings scattered across files → prompts are Jinja templates in one place, versioned.
- ❌ Letting the model decide the language blindly → the router decides, the model is told.
- ❌ Trusting Darija spelling from the model for anything stored → normalise + read back before saving.
- ❌ Comparing providers without a fixed prompt → freeze the prompt, vary only the model.
- ⚠️ Gemini free tier is 5–15 RPM: a burst of tool calls can 429. The fallback chain isn't optional.
- ⚠️ Don't put the whole conversation in the prompt "because it's free" — TTFT is what you're protecting.
