# L3 — Mouth (TTS: Piper British, Darija, cache, prosody)

**Goal:** Atlas speaks. **Darija voice by default**, a real **British** voice on request, streamed sentence-by-sentence, with a cache so repeated lines are instant.
**Effort:** ~1 week. **Depends on:** L2. **Gate:** first spoken word < 3 s for a normal answer; a cached phrase plays in < 200 ms; no crackle, no drift.

---

> **Implementation notes: [`NOTES-L03.md`](NOTES-L03.md)** — what was built, the
> measurements, the seven bugs the tests caught, and the laptop command sequence.
> Implemented: `speech.py` (splitter + streaming pipeline + `Mouth`), `tts.py`
> (Piper, the Darija sidecar, SAPI, the fallback policy), `cache.py`, `prosody.py`,
> `playback.py`, `vendor/darija-tts/server.py`, `scripts/bench_tts.py`,
> `scripts/spike_darija_tts.py`, `atlas say` / `atlas voice`.
> **Pitch is deliberately absent** from `Prosody` — Piper and OuteTTS have no
> pitch control, so the table's pitch column became `expressiveness` (see §3 of
> the notes). The Darija spike and every latency/RAM number still need the laptop.

## Deliverables

- `SpeechSynthesizer` implementations: `PiperEnGbSynthesizer`, `PiperDarijaSynthesizer` (bootstrap), `DarijaTtsSidecarSynthesizer` (the real Darija voice), `SapiFallbackSynthesizer`.
- `SentenceStreamer`: LLM token stream → sentences → synthesis queue → playback (the single biggest perceived-latency win).
- `TtsCache`: SQLite + hash(text, voice, rate) → opus/wav file; LRU to a size cap.
- `ProsodyDirector`: rate/pitch/energy per mood and per sentence role (question, joke, warning, confirmation).
- Audio ducking + gain, and hard half-duplex enforcement.

## The Darija voice problem (read before coding)

Open-source Darija TTS exists but is young: **KandirResearch/DarijaTTS-v0.1-500M** (Apache-2.0, fine-tuned from OuteTTS-0.2-500M, GGUF available) is the realistic candidate. It is a *text-generation* TTS (OuteTTS style: text → tokens → vocoder), so:

* run it as a **sidecar process** in an isolated venv (`vendor/darija-tts/`) so torch/vocoder deps never enter Atlas's core;
* speak **sentence-level** (it's slower than Piper) and cache aggressively;
* validate quality in step 3 below **before** you build anything on top of it;
* keep two fallbacks ready: Piper with an Arabic voice, and cloud TTS.

If quality is unacceptable, that's a *finding*, not a failure: the L9 project "train a Piper Darija voice on open Darija data and publish it to HF" is the real long-term answer, and your fallback stack keeps Atlas usable until then.

---

## Steps

1. **Piper en-GB first** (fastest path to a working mouth). Pick an `en_GB` medium voice; wrap `piper` CLI or the ONNX API. Verify: instant startup, ~10× realtime on your CPU, "water/answer/bath" sound British.
2. **Sentence streaming.** Consume the L1 token stream; split on sentence boundaries with an abbreviation-safe splitter (`.`, `؟`, `!`, `،`, and Darija particles). Synthesise sentence *n+1* while sentence *n* plays. Never wait for the full answer.
3. **Darija voice spike (timeboxed: 1 evening).**
   - Set up `vendor/darija-tts/.venv`, pull `unsloth.Q8_0.gguf`, run the model's own demo script.
   - Speak 10 real Darija sentences (greeting, question, joke, numbers, your name, an address).
   - Score: intelligibility (would a Moroccan understand it on the first listen?), naturalness, speed, RAM.
   - Write the verdict in `docs/levels/NOTES-L03.md` **with the audio files kept** for comparison.
4. **`DarijaTtsSidecarSynthesizer`** — start `python -m darija_tts.server --port 8125` on demand via `ResourceLease`, POST text, receive WAV, stream to output. Health check, timeout, restart-once, then fall back to Piper-Arabic → SAPI.
5. **PyTorch isolation rule.** The sidecar venv may contain torch; the core venv must not (`pip list | findstr torch` → empty). CI check: `scripts/check_no_torch.py` scans the core lockfile and fails the build if torch appears.
6. **`TtsCache`** — `hash(text_normalised + voice + rate)`. Cache: canned replies, greetings, "sure/safi/daba", time/date announcements, timer alerts, your most repeated 50 phrases. LRU to ~300 MB. First-run warming at boot (synthesise the canned set in a background thread — 2 cores, so *below normal* priority).
7. **`ProsodyDirector`** — from `MoodState` + sentence role:
   | Mood/role | rate | pitch | energy |
   |---|---|---|---|
   | Calm answer | 1.0 | 1.0 | 1.0 |
   | Amused (joke) | 1.05 | +5 % | 1.1 |
   | Serious / bad news | 0.92 | −5 % | 0.9 |
   | Late-night (quiet hours) | 0.95 | −3 % | 0.75 |
   | Confirmation question | 0.95 | +3 % | 1.0 |
   Piper exposes length-scale/noise; the Darija sidecar gets rate only — accept and document.
8. **Playback & mixing.** WASAPI shared-mode output, `sounddevice`. Ring buffer per sentence, seamless join (no gaps between sentences), 40 ms fade-in/out to kill clicks, master gain from `preferences.md`. **Mute the mic while speaking** (half-duplex); re-arm on the last sentence's tail.
9. **Audio-level metering.** Emit `AudioLevelChanged` at ~30 Hz to the UI for orb amplitude. Compute from the playback buffer, not a separate capture (cheap, exact).
10. **Interruption policy.** "Stop" / "safi" / "سكت" / pressing the stop hotkey → flush the queue, short fade, return to Listening. Also cap: never speak more than ~3 sentences of a long answer without offering *"bghiti nkemmel?"*.
11. **Streaming polish.** If a sentence is > 220 chars, split at the nearest comma/`و` boundary so first audio comes sooner.
12. **Failure handling.** Sidecar crash → mid-answer switch to Piper-Arabic + a note in the log; both TTS engines down → SAPI; everything down → captions only, UI says so.
13. **Measure.** `scripts/bench_tts.py`: time-to-first-audio and realtime factor per engine, hot vs cold, cache hit vs miss, in `NOTES-L03.md`.

## Classes

`SpeechSynthesizer` (ABC) · `PiperSynthesizer` (voice-configurable; en-GB and Arabic voices) · `DarijaTtsSidecarSynthesizer` · `SapiFallbackSynthesizer` · `SentenceStreamer` · `SentenceSplitter` (abbreviation-safe) · `TtsCache` · `ProsodyDirector` · `AudioPlayer` · `DuckingController` · `MicGate` (half-duplex) · `VoiceConfig`.

## Tests

- Splitter: golden cases for English, Darija, mixed, abbreviations ("Dr.", "e.g.", "daba.", numbers).
- Streaming: fake token stream → assert first audio callback fires before the stream ends (the whole point).
- Cache: hit/miss/LRU eviction/size cap; hash stability across processes.
- Prosody: mood → parameter mapping table tests.
- Half-duplex: assert mic gate closes before the first chunk is played and reopens after the tail.
- Sidecar: crash simulation → fallback engine used, user hears a complete answer.
- No-torch check passes.

## Done when

- A normal 3-sentence answer starts speaking in **< 3 s** end-to-end (wake → first audio), Lean profile.
- 20 cached phrases play in < 200 ms.
- Darija sentence spike audio is intelligible to you **and** to a Moroccan relative/friend (test outside your own head).
- Zero torch in the core lockfile; sidecar starts in < 10 s cold, < 2 s warm.
- `NOTES-L03.md` has the audio files and the honest verdict on each engine.

## Pitfalls

- ❌ Synthesising the whole answer before playing → kills the entire experience.
- ❌ Letting the Darija sidecar live in the core venv → a 2 GB torch install in your "lightweight" assistant.
- ❌ Playing while the mic is open → Atlas answers itself (and looks insane).
- ❌ Emotional variety via pitch on a vocoder that can't do it → use rate + energy + wording, not fantasy knobs.
- ⚠️ Piper voice licensing: check each voice's model card before shipping a demo.
- ⚠️ Arabic text without diacritics may be mispronounced by Arabic-capable engines — keep a small pronunciation override list in the vault lexicon.
