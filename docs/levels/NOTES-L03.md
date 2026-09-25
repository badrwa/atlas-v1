# NOTES — L3, the mouth

**Status: implemented and tested in the repo; the microphone-and-speakers half of
the gate needs the Latitude.** Every number below that could be measured without
audio hardware was measured; the rest are listed with the exact command that
produces them at the end of this file.

---

## 1. What is actually there now

| Piece | Module | What it does |
|---|---|---|
| Sentence splitter | `speech.SentenceSplitter` | Incremental, abbreviation-safe (`.`, `!`, `?`, `؟`, `…`), `3.5` stays a number, "Dr." is not an ending, `،` cuts an over-long sentence |
| Streaming pipeline | `speech.SentenceStreamer` | tokens → sentences → audio, three tasks, **bounded** queues, synthesises *n+1* while *n* plays |
| Composed mouth | `speech.Mouth` | engines + cache + prosody + player + mic gate; the only object the CLI speaks through; `warm()` for the canned lines |
| Prosody | `prosody.ProsodyDirector` | mood + sentence role + clock → rate / energy / expressiveness (pitch deliberately absent, see §3) |
| Cache | `cache.TtsCache` | SQLite, LRU by bytes, key = sha256(text + voice + language + engine + rate) |
| Voices | `tts.PiperSynthesizer`, `tts.DarijaTtsSidecarSynthesizer`, `tts.SapiSynthesizer` | en-GB + Arabic Piper, the Darija sidecar over localhost HTTP, Windows `System.Speech` last |
| Engine policy | `tts.build_voice_chain`, `build_synthesizer`, `voice_problems`, `tts_status` | one place decides the fallback order per language |
| Playback | `playback.AudioPlayer`, `MicGate`, `DuckingController`, `SoundDeviceWriter` | `RawOutputStream` (no numpy), 40 ms fades, gain, ~30 Hz metering, half duplex |
| Sidecar | `vendor/darija-tts/server.py` | stdlib-only HTTP server, `/health` + `/synthesize`, own venv, lazily imports outetts |
| Spike | `scripts/spike_darija_tts.py` | ten sentences → ten WAVs → a scoring sheet and a verdict table |
| Bench | `scripts/bench_tts.py` | time-to-first-audio and RTF per engine, cold/warm, cache miss/hit, PASS/FAIL vs the gate |

CLI additions: **`atlas say "…" [--out file.wav] [--engine …]`**, **`atlas voice
[status|test|cache|warm]`**, and `atlas listen` now speaks (`--no-voice`,
`--voice-engine`) with `doctor` and `listen --status` reporting the voice chain and
the cache.

## 2. The streaming decision, in numbers

The point of sentence streaming is that time-to-first-audio tracks *one sentence*,
not the whole answer. The test that pins it down feeds "Salam. " and then sleeps
500 ms before the rest of the text, and asserts the first sentence is out in under
400 ms:

```
test_the_first_word_does_not_wait_for_the_last_token        PASS
test_the_second_sentence_is_synthesised_while_the_first_plays  PASS
```

Measured in the sandbox with a stub voice (50 ms per sentence, 0.4 s of audio), so
the *pipeline* is what is being measured, not the model:

| Pass | first-audio p50 | p95 | RTF | cache |
|---|---|---|---|---|
| first (cache miss) | 52 ms | 53 ms | 7.6× | 0/5 |
| second (cache hit) | 1.0 ms | 1.4 ms | — | **5/5** |

The pipeline adds ~2 ms of its own overhead per sentence; everything else in the
real numbers will be the voice. The cache row is the L3 gate's "< 200 ms" with
three orders of magnitude to spare — a cache hit is a `SELECT` and a `write()`.

Queue depths are the RAM budget: two sentences of text and two of audio. On this
laptop a sentence of 16 kHz mono PCM is ~30 KB/s, so the pipeline holds well under
1 MB of audio no matter how long the answer is.

## 3. Decisions worth defending

**No pitch.** Piper's ONNX voices and OuteTTS-style vocoders have no pitch control.
`Prosody` therefore carries `rate`, `energy` (a gain the player applies) and
`expressiveness` (Piper's `noise_scale`/`noise_w`). Faking pitch by resampling
would move speed and formants together — the tape-warp sound the plan's pitfall
list warns about. The table's "pitch" column is honest about what it became.

**The cache key includes the rate.** A cheerful "Salam." is different audio from a
calm one; serving one as the other is a bug you can hear. This also means the
cache is per-mood, which is fine: the canned lines are always calm.

**A connection per SQLite operation.** Synthesis runs on a worker thread, and
sqlite3 objects are not shareable across threads by default. One connection per
call costs microseconds and removes an entire class of "database is locked"
failures. `:memory:` uses a shared-cache URI with a unique name plus one keeper
connection, which is what makes it behave like the real file in tests.

**The engine chain is a policy with one home.** `build_voice_chain(config,
language)` decides the order (Darija: sidecar → Piper-Arabic → SAPI; English:
Piper → SAPI), `Mouth` keeps the whole chain, and the streamer falls *forward* per
sentence. A dead engine mid-answer must not shorten the answer: the same sentence
is retried on the next voice, and only when all of them fail does Atlas speak
nothing and show the words as captions (`degraded = True`, `engine = "none"`).

**The mic gate is reference counted.** The reply holds the mic shut *and* the
mouth's gate does; a plain boolean would let whichever finished first reopen it —
the classic "Atlas answers itself" bug. `VoiceLoop.mute()/unmute()` now count, and
a test pins the ordering (gate opens → still muted until the reply releases).

**The sidecar is a process, not a dependency.** `vendor/darija-tts/` has its own
`requirements.txt` and venv; `check_no_torch.py` keeps torch out of the core, and
the sidecar's absence is reported as *one honest sentence with the install line*
(HTTP 503 body → `voice_problems()` → `atlas voice` → `atlas doctor`).

**Long answers stop.** After `max_sentences` (default 3) Atlas stops and asks
"bghiti nkemmel?" The brain keeps the conversation, so "kemmel" continues it in the
next turn; nothing is buffered waiting for a yes, and a wall of speech never
happens by accident.

## 4. Bugs the tests caught (all fixed in this level)

1. **Eviction deleted the whole batch.** `prune()` fetched 50 LRU rows and deleted
   all of them, because the size check only happened before the batch — a 1 kB cap
   wiped the cache instead of dropping one clip.
2. **`:memory:` caches saw an empty database.** A connection per operation means
   each call opened a *new* in-memory database: `no such table: clips`. Fixed with
   a shared-cache URI, a keeper connection, and a **unique name per instance** (a
   shared in-memory database is shared by the whole process — two caches in one
   test run read each other's clips).
3. **A lazy `TypeError` escaped the Piper version check.** Generators raise on
   first iteration, not on call; the fallback for old Piper builds materialises
   *inside* the `try` so it actually catches the builds it exists for.
4. **`engine_index` walked off the end.** When every voice failed, `stats()` raised
   `IndexError` — because the engine property indexed the list with the past-the-end
   index. Clamped, with the degraded state reported as `"none"`.
5. **`atlas say` wrote a silent WAV and reported success.** It now fails with "no
   voice" and a pointer to `atlas voice`. Silence is not an answer.
6. **`--no-voice`-less runs built the mouth before the loop existed** (the gate
   needs `loop.mute`), so the mouth is now built *after* the loop, and only when
   there is an output device.
7. **`atlas voice warm` was wired into the wrong command** (`cmd_audio` has no
   `config`): an F821 in CI would have caught it, and mypy did.

## 5. The gate, and where its numbers come from

| Gate | How it is measured | Status |
|---|---|---|
| First spoken word < 3 s for a normal answer | `scripts/bench_tts.py` → `first_audio_p50_ms` (first pass) | needs the laptop |
| A cached phrase plays in < 200 ms | same table, `phase = cached` | sandbox: **1.0 ms** with a stub voice |
| No crackle, no drift | 40 ms fades + one resample per sentence + a kept-open stream | by construction; ears needed |
| 20 cached phrases in < 200 ms each | `atlas voice warm` then `atlas listen`, then `bench_tts --no-cache` off | needs the laptop |
| Darija spike intelligible to a Moroccan | `scripts/spike_darija_tts.py --score` → `verdict.md` | needs the laptop |
| Zero torch in the core | `scripts/check_no_torch.py` | ✓ (also checks the running venv) |
| Sidecar cold < 10 s, warm < 2 s | `atlas voice` / `/health` `load_s` | needs the laptop |

### The laptop session, in order

```powershell
# 1. one-time: Piper voice (63 MB) into models/tts/
#    https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/alan/medium
#    en_GB-alan-medium.onnx + en_GB-alan-medium.onnx.json

# 2. what can speak, and with which voice
python -m atlas voice
python -m atlas voice warm                 # the canned lines, cached

# 3. does it sound British (and does one sentence come out before the next is ready)
python -m atlas say "Right, let me finish that for you." --language en-GB
python -m atlas say "Salam, ana Atlas. Chno ljaw dyalek lyoum?" --language ar-MA

# 4. the gate numbers, both languages
python scripts/bench_tts.py --language en-GB
python scripts/bench_tts.py --language ar-MA
python scripts/bench_tts.py --cold           # includes model load: the first-sentence number

# 5. Darija: the sidecar (README in vendor/darija-tts), then the spike
python scripts/spike_darija_tts.py
python scripts/spike_darija_tts.py --score   # listen to all ten first!

# 6. the whole thing: ears + brain + mouth, with the mic shut while Atlas talks
python -m atlas listen --ptt
python -m atlas listen
```

Record in this file: the bench table per engine, the sidecar's `load_s`, the spike
`verdict.md`, and one sentence on how the first spoken word felt.

## 6. What L3 deliberately did *not* do

- **No cloud TTS.** It is the third fallback in the plan and it is not needed while
  Piper runs at 7–10× realtime; adding it would mean a second privacy switch and a
  second bill. Revisit in L9 if the Darija voice stays unusable.
- **No voice barge-in over Atlas's own speech.** Echo cancellation on this laptop is
  a research project, so the microphone stays shut for the whole reply (that is what
  half duplex means) and the **stop hotkey (Ctrl+Alt+S) is the supported interrupt**.
  `InterruptPolicy.hears_stop()` is implemented and tested — the word list is parsed
  and ready for L5, which owns the mic-open policy — but with `[tts] barge_in = false`
  (the default) nothing listens for "safi" while Atlas is talking, because listening
  would mean hearing Atlas itself.
- **No OS-level ducking** of other applications' audio (a WASAPI session feature).
  `DuckingController` owns *Atlas's* level, which is what the mood table and quiet
  hours need.
- **No Piper CLI fallback.** One synthesis path (the Python API) instead of two that
  drift; if `piper` is missing the chain says so and the next voice speaks.
