# L8 — Soul (friend voice, dialect packs, humour, emotion)

**Goal:** Atlas stops being a tool and becomes *your* friend: it talks like a friend, jokes like a friend, matches your mood, and speaks real Darija and real British English — not translated textbook versions of them.
**Effort:** ~1 week, then continuous tuning. **Depends on:** L1, L3, L4, L6. **Gate:** the 6 personality tests below pass with a Moroccan friend's honest verdict.

---

## Deliverables

- `DialectPack` data files: `darija.yaml`, `en_gb.yaml` (greetings, fillers, reactions, humour styles, taboo list, number/unit conventions, apology/recovery lines).
- `PersonalityEngine`: `Persona` + `MoodEngine` + `HumorPolicy` + `RelationshipModel`.
- `HumorPolicy`: joke types (tease, self-deprecating, callback, absurd), gates, and a callback bank in `50_Atlas/jokes.md`.
- Mood-aware prosody wiring (L3 `ProsodyDirector`) and mood-aware UI (L5).
- Calibration harness: 30 scripted scenarios with expected tone → regression tested.

## What "like a friend" actually means (operationalise it)

| Friend behaviour | Implementation |
|---|---|
| Uses your name / nickname, remembers context | `RelationshipModel` reads `memory.md` + `preferences.md` + last sessions; opening lines reference real, recent things |
| Teases you, and takes it back | `HumorPolicy` with a **tease budget** (max 1 tease per topic, never in `Serious`/`Frustrated`) |
| Doesn't lecture | Answer length caps by intent; no unsolicited advice; a `/lecture` permission you can revoke out loud |
| Admits ignorance | "Ma3reftch, w ma-bghitch nkdeb 3lik" — and offers to look it up |
| Matches your energy | Mood → rate/warmth/joke policy; tired you = shorter, quieter, no bits |
| Has its own bit | Running gags in `jokes.md`, callbacks weeks later, a consistent opinionated streak (tea, traffic, Mondays) |
| Knows when to be serious | Sensitive-topic detector → `Serious` mood → humour off, shorter sentences, no jokes *even if asked* |

## Dialect packs (the difference between "Darija" and Darija)

- **Darija pack must contain**: real greetings (`salam, kifach nta?`, `labas 3lik?`, `ash khbarek?`), agreement (`wah`, `safi`, `mzyan`, `d'accord`), hedges (`wa chwiya`, `3la 9bal`, `insha'allah` used naturally), and Moroccan humour register (teasing, hyperbole, self-deprecation, `bach nfhem` callbacks). Explicitly avoid MSA forms sounding stiff (`هل يمكنك أن...`) — those get flagged by a lint list.
- **British pack must contain**: understatement ("not bad", "a bit of a mess"), `have a go`, `sorted`, `bits and bobs`, `cheers`, `proper` as intensifier, dry irony, and a banned-words list of Americanisms ("awesome", "reach out", "dude", "my bad", "24/7").
- **Numbers/units**: Moroccan French-influenced counting (`dix dirhams`, `melyoun`), Celsius, `kilometre`, metric cooking units, 24-hour time in Darija / 12-hour in en-GB if you prefer (config).
- **Script choice**: Arabic script or Arabizi per `preferences.md`; voices get Arabic-script text regardless (TTS needs it), the caption follows your preference.

## Steps

1. **`DialectPack` schema + YAML files.** Load, validate (pydantic), and unit-test that every pack has all required sections. Adding a language later = adding a file.
2. **`RelationshipModel`** — builds a small, honest picture of you from the vault: name, what you're working on, mood patterns by hour, 3 recent topics, known preferences, running gags. Token-budgeted (≤ 200 tokens) and **never** invented: if it's not in the vault, it doesn't exist.
3. **`MoodEngine`** (D7) — inputs: prosody features from audio (energy, rate, pitch variance — cheap, no model needed), the LLM's `emotion` field on the user's message, session signals (ASR retries, interruptions, time of day, quiet hours), and history decay. Output `MoodState(label, energy, warmth, humor_allowed, rate_mult, voice_style)`.
4. **`HumorPolicy`** — decides *whether* to joke (`humor_allowed` ∧ topic-safe ∧ not first interaction after a serious turn), *what kind* (weighted by mood: calm → tease/callback; happy → absurd/wordplay; tired → dry one-liner), and *how to recover* (if the user doesn't laugh — detectable via "safi", silence, or "kml" — drop the bit immediately and continue).
5. **Callback bank.** `50_Atlas/jokes.md` holds setups with dates; the policy may reference one when a related topic arises. Cap: one callback per session, never more than 3 weeks old unless it's a legend-tier gag. This single file does more for "feels like a friend" than any model upgrade.
6. **Honesty rules** (non-negotiable, encoded in the persona prompt *and* tested):
   - Never claim an action it didn't take.
   - Never invent a memory ("I don't remember that" is a complete answer).
   - Say when it's guessing, and say when it's offline.
   - Never flatter repeatedly; one genuine compliment beats five.
7. **Calibration harness** — 30 scripted scenarios (`tests/personality/scenarios.yaml`), each with: user line(s) in Darija/en-GB, a mood context, and assertions (tone, length, joke allowed?, banned phrases?). Run with a fake provider returning fixed text to test the *policy*, plus a live run for qualitative review.
8. **Qualitative loop (the honest part).** Every Friday: 10 real conversations → you and one Moroccan friend score tone/naturalness 1–5 → adjust packs and prompts. Keep the scores in `50_Atlas/log/` so improvement is visible over months, not vibes.
9. **Emergency modes.** *"Atlas, wa9ef l'ghizi"* (stop joking), *"b sirious, 3afak"* (be serious) → immediate mode switch stored in `preferences.md` for the session; a "no jokes in the morning before my tea" rule is a config, not a code change.
10. **Speech-style enforcement.** Post-processor that rewrites LLM output for speech: numbers, dates, units, abbreviations (`RAM` → "رام"? no — leave technical terms as-is but spelled for TTS), no markdown, no emoji, no parentheses, max ~25 words per sentence, and prosody hints (pause markers, emphasis) that `ProsodyDirector` consumes.

## Classes

`DialectPack` · `DialectPackLoader` · `PhraseBank` · `RelationshipModel` · `MoodEngine` · `MoodState` · `MoodSignal` · `HumorPolicy` · `CallbackBank` · `Persona` · `SpeechStyleRewriter` · `CalibrationHarness`.

## Tests

- Pack completeness + no-banned-words lint (Americanisms in en-GB, MSA-stiffness in Darija).
- Mood: signal → state table tests; decay; quiet-hours override.
- HumorPolicy truth table: (mood × topic-safety × turn index) → joke allowed/denied. Serious topics must never produce a joke, including when explicitly asked.
- Honesty: fake provider that claims a fake action → post-check must not emit it as fact.
- Speech style: 20 golden responses → rewritten for speech correctly (numbers, no markdown, sentence length).
- Scenario suite green; a live-run checklist in `NOTES-L08.md` with the friend's scores.

## Done when

Six personality tests, judged by you **and** one Moroccan friend:

1. **The friend test** — a 10-minute free conversation that neither of you would describe as "talking to a chatbot".
2. **The joke test** — 5 jokes in Darija that land (laughter or a grin), and one that fails without breaking the conversation.
3. **The callback test** — it references something from a previous day, correctly.
4. **The mood test** — you sound exhausted; Atlas gets quieter, shorter, no jokes.
5. **The serious test** — bad news → zero humour, warm and brief, no advice unless asked.
6. **The dialect test** — a native speaker can't tell the Darija was machine-templated (and the British English never slips into American).

## Pitfalls

- ❌ "Add a joke instruction to the prompt" → one-liner spam. Humour needs policy, budget, callbacks and recovery.
- ❌ Sycophancy drift (models love "Great question!") → banned-phrase lint + persona tests.
- ❌ Gags that ignore mood → the fastest way to make Atlas feel tone-deaf.
- ❌ Over-fitting to one friend's scoring → keep your own taste primary in `preferences.md`.
- ⚠️ Humour in Darija is *very* register-sensitive; get a native reviewer before trusting your own ear.
- ⚠️ Personality costs latency (longer prompts). Keep `RelationshipModel` ≤ 200 tokens and re-measure TTFT after every prompt change.
