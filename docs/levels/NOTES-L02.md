# NOTES — L2 (ears)

Measured numbers, decisions taken, and the parts of the gate that only the
Latitude 5480 can answer. Nothing here is estimated: where a cell says "not
yet", the number has not been measured yet — and the command that measures it is
written next to it.

## What shipped

| Piece | File | What it does |
| --- | --- | --- |
| Frames, ring buffer, bus, capture thread | `atlas_audio/frames.py` | 80 ms frames at 16 kHz mono, 1.5 s pre-roll, drop-don't-stall fan-out |
| Microphone health | `atlas_audio/devices.py` · `capture.py` | device probing, WAV in/out, record-and-play-back self-test |
| Endpointing | `atlas_audio/vad.py` | Silero VAD via sherpa-onnx, energy fallback, utterance boundaries |
| Wake word | `atlas_audio/wake.py` | sherpa-onnx KWS, openWakeWord alternative, energy demo, hit log |
| Speech recognition | `atlas_audio/asr.py` | Gemini/Groq cloud first, faster-whisper Darija local fallback |
| Text cleanup | `atlas_audio/postprocess.py` | language ID, dictation numbers, owner-editable lexicon |
| The turn loop | `atlas_audio/loop.py` | wake → listen → transcribe → reply → 2.5 s follow-up, half duplex |
| Push-to-talk | `atlas_audio/ptt.py` | Enter-driven PTT, optional global hotkeys |
| Commands | `apps/atlas/src/atlas/cli.py` | `atlas audio devices\|test\|replay`, `atlas listen [--ptt --replay --status]` |
| Bench | `scripts/bench_asr.py` | latency + WER + realtime factor on your own recordings |

Shared plumbing that L2 forced out of the packages, rather than copying it:
`atlas_core/jsonl.py` (timings and the wake log are the same object),
`atlas_core/http.py` (the HTTP client lifecycle, previously duplicated between
the cloud ASR and the provider base), `atlas_core/engines.py` (`LoadedFlag`).
The duplicate gate found all three; none of them would have been noticed by
reading.

## Measured here (no microphone, no network)

Sandbox: Linux container, Python 3.11.2, 2 cores — *weaker* than the Latitude for
single-thread work, so these are conservative.

| Measurement | Value | How it was measured |
| --- | --- | --- |
| VAD alone, 1 h of audio | **1.65 s wall · 2 183× realtime · 0.046 % of one core** | 21 000 synthetic frames through `VadSegmenter` |
| Full loop, 28.8 min of audio | **1.30 s wall · 1 327× realtime** | wake + VAD + stub ASR + reply, 300 turns |
| Loop overhead per turn | **4.34 ms** | 300 turns, ASR and network excluded |
| `rms()` per frame | 50 µs | 20 000 iterations |
| Post-processing per utterance | **7.7 µs** | lexicon + numbers + whitespace |
| `detect_language()` | **6.0 µs** | marker table + script scan |
| `pcm_to_frames`, 8 s clip | 0.07 ms | 1 000 iterations |
| 1.5 s pre-roll ring | 2 KiB (18 frames) | `tracemalloc` |
| Audio bandwidth | 31 KiB/s · 2 560 B/frame | 16 kHz mono PCM16 |
| `pytest` (whole workspace) | **401 tests in 4.3 s** | includes CLI end-to-end |
| `check_duplicates` | 46 modules | window 45 tokens |

Audio never touches the event loop's critical path at these numbers: the whole
capture-and-detect chain costs ~0.02 % of one core on the sandbox CPU, and the
Latitude's Skylake core is faster per-thread than this container.

## Bugs found by writing the tests (all fixed in L2)

1. **The energy VAD swallowed its own silence.** Its "hangover" grew with every
   loud frame, so after 15 frames of speech it took 15 quiet frames to let go —
   the segmenter's 500 ms silence run could never complete, and every utterance
   was closed by `flush()` instead of by endpointing. Hangover is now capped,
   defaults to zero, and the comment says why.
2. **`lstrip("-*0123456789. ")` ate Darija.** Stripping Markdown bullets from
   the lexicon file also stripped the leading digit of `3ndi` → `ndi`, silently
   corrupting every learned correction that starts with a numeral. Bullets are
   now stripped as a regex *prefix*, never as a character class.
3. **Resampling per frame stretched the audio.** Reading a 48 kHz stereo WAV
   converted each 1 280-sample chunk independently, padding every chunk to a
   full frame: 500 ms became 1 520 ms. Conversion now happens once over the
   whole buffer (`resample_pcm`), which is also the function the live mic uses.
4. **Three local models shared one lease identity.** `LocalWhisperRecognizer`
   used a class-level `name`, so Darija, English and multilingual collided in
   `ResourceLease.entries` — eviction would have freed the wrong model. Each
   instance now carries `whisper:<key>`.
5. **`min_utterance_ms` padded short noises instead of dropping them.** A cough
   was topped up to the minimum and transcribed. The measure is now *speech*
   frames, not total frames.
6. **The wake word's own tail could lead the transcript.** 160 ms after a hit is
   dropped (`wake_guard_ms`), so the command never starts with "-las".
7. **`RecognizerPolicy` had dead code** (`"localhost" if False else …`) and the
   factory kept a second copy of the preference order; both now read one list.

## Decisions taken (and why)

| Decision | Why it is right for this machine |
| --- | --- |
| Frames are `array("h")`, not numpy | `atlas-audio` imports without numpy; numpy appears only in `to_numpy()`, Silero and faster-whisper |
| Pre-roll 1.5 s, but lead-in capped at 200 ms | the ring is what stops a clipped first syllable, the cap is what stops paying to upload room tone |
| A long quiet lead is trimmed, a loud one is kept | continuous speech keeps its run-up (the wake tail), silence does not |
| `followup_ms = 2500`, then re-arm | one "atlas" per conversation, but the television never gets a turn |
| Half duplex by muting frames, not by closing the device | reopening a WASAPI device costs ~100 ms and crackles; ignoring frames costs nothing |
| Every frame is recorded when `--capture-dump` is on, including during speech | a dump that omits the wake word cannot reproduce the failure it exists for |
| Per-utterance WAVs *and* one session WAV | the small files are what `bench_asr.py` replays; the big one is what debugs the loop |
| Transcript confidence: cloud = fixed 0.8, local = `1 + avg_logprob` | Gemini returns no confidence at all; inventing a number would be worse than a constant that is documented |
| The lexicon is a Markdown file of `wrong = right` | the owner edits it in Obsidian, and the same values become cloud ASR hotwords |
| `cloud_audio = false` forces `local_only` in the policy | a privacy switch that can be bypassed by a fallback is not a privacy switch |

## The gate: what is verified here, and what needs the laptop

| L2 done-when | Status | How it is checked |
| --- | --- | --- |
| 10 hands-free commands in a row, transcripted correctly | **needs the laptop** | `atlas listen --capture-dump`, then `python scripts/bench_asr.py data/recordings/<session>` |
| 0 false wakes in 4 h idle | **needs the laptop** | leave `atlas listen` running; `atlas listen --status` prints the hits and the rate from `data/wake_log.jsonl` |
| ≤ 180 MB always-on | **needs the laptop** | Task Manager while `atlas listen` idles; the sandbox has no ONNX runtime to weigh |
| Cloud ASR p50 ≤ 1.2 s | **needs the laptop** | `python scripts/bench_asr.py <rec> --engines cloud` |
| Local Darija p50 ≤ 3.5 s | **needs the laptop + model** | same command, `--engines local:darija` |
| Everything is testable with no hardware | **done** | 401 tests; the mic, the models and the network are all injected |

### The laptop session, in order

```powershell
uv run python -m atlas audio devices      # is WASAPI there, which device is default
uv run python -m atlas audio test         # record one second, play it back
uv run python -m atlas listen --status    # wake engine, VAD backend, ASR mode, devices
uv run python -m atlas listen --ptt       # first end-to-end turn, no wake word needed
uv run python -m atlas listen --capture-dump   # the real thing, recorded

# then, with the recording in hand:
python scripts/bench_asr.py data/recordings/<session> --from-manifest --engines cloud
python scripts/bench_asr.py data/recordings/<session> --engines local:darija --repeat 3
```

Paste the resulting table back into this file, under "On the laptop".

### Not yet installed anywhere (and the honest failure)

`sherpa-onnx` (wake word + Silero VAD) and `faster-whisper` (local ASR) are
optional extras; they are not in the base install and not in CI. Without them
Atlas says so in one sentence and keeps working:

| Missing | Behaviour |
| --- | --- |
| `sounddevice` | `atlas audio devices` reports the extras line; nothing crashes |
| `sherpa-onnx` | wake falls back to the energy demo, VAD to the energy detector, both logged |
| `openwakeword` | same, with the package name in the message |
| `faster-whisper` | cloud-only; `local_first`/`local_only` report "no ASR engine available" |
| no API key | `--status` prints `cloud keys: none` instead of attempting a request |

## What L2 deliberately does not do

- **No speaker verification.** Voice identity is L4; until then, "restricted
  mode" cannot exist, and nothing here pretends it does.
- **No TTS.** `say()` receives deltas from L1's streaming reply and prints them;
  L3 replaces it with Piper/DarijaTTS. Half duplex is already enforced, so L3
  inherits a loop that cannot hear itself.
- **No always-on cloud streaming.** Audio goes out only after a wake word, only
  for the utterance, and only when `cloud_audio` is true.
- **No `atlas watch`/tray icon.** The orb and its cloud glyph arrive with the UI
  level; the loop already exposes `snapshot()` for it.

## Next (L3 — the mouth)

1. `PiperSynthesizer` (en-GB) and the DarijaTTS sidecar, behind L0's
   `SpeechSynthesizer` contract — the two planned classes in `engines.py`.
2. Sentence-chunked streaming into the audio device, with barge-in: if the owner
   speaks during playback, stop and listen (the loop already mutes, so barge-in
   is a coordination change, not a rewrite).
3. First-sound latency target from the plan: ≤ 700 ms from reply start.
4. `atlas say "..."` for a zero-brain voice test, and the chime that L2 already
   has a flag for (`dialogue.chime`) but no speaker to play it on.
