# L2 — Ears (wake word, VAD, Darija ASR)

**Goal:** hands-free. Atlas wakes on **"atlas"**, hears you in **Darija or British English**, and stops when you stop talking. Push-to-talk exists as a fallback and a test tool.
**Effort:** ~1.5 weeks. **Depends on:** L1. **Gate:** 10 hands-free commands in a row (5 Darija, 5 English) produce correct transcripts, 0 false wakes in a 4 h idle test, ≤ 180 MB for the always-on models.

---

## Deliverables

- `WakeWordEngine` implementation (`SherpaKwsEngine`, with `OpenWakeWordEngine` as an alternative) + **1.5 s pre-roll ring buffer**.
- `VadSegmenter` (Silero VAD via sherpa-onnx, WebRTC VAD as backstop) with configurable endpointing.
- `SpeechRecognizer` implementations: `CloudRecognizer` (Gemini native audio; Groq `whisper-large-v3-turbo`), `LocalWhisperRecognizer` (faster-whisper, Darija model), `SherpaOfflineRecognizer` (MoulSot/Qwen3-ASR → ONNX later).
- `LanguageRouter` audio path (spoken language ID / script detection + explicit switching).
- Half-duplex enforcement: mic muted while speaking (hard requirement, see master plan §7.7).
- `scripts/bench_asr.py` producing realtime-factor numbers for your voice and commands.

## Why sherpa-onnx first

One CPU runtime covers KWS + VAD + speaker ID + ASR + TTS, in C++, with ONNX models for Windows x64, no Python ML stack, no torch. Fewer moving parts, less RAM, one thing to keep warm. faster-whisper stays as the CTranslate2 path where its Darija/English quality wins.

---

## Steps

1. **Audio plumbing.** `sounddevice` + WASAPI: list devices, store names in `config.toml`, open a **16 kHz mono** input stream (downsample if the device insists on 48 kHz), 80 ms frames (1280 samples). Add `atlas audio devices` and a record/playback self-test.
2. **Frame bus.** One producer thread → `asyncio.Queue` of frames, fanned out to KWS, VAD and the recorder **without blocking**. Frames are `np.int16`. Add a `--capture-dump` flag that writes raw frames to disk so you can replay a real session in tests (this becomes your regression corpus).
3. **Wake word.**
   - Bootstrap: sherpa-onnx KWS with an English keyword model; alternative `OpenWakeWordEngine` with `hey_jarvis`. Both behind `WakeWordEngine`.
   - Custom word: train **"atlas"** with openWakeWord's synthetic-TTS notebook (free Colab), then a **verifier** on ~10 of your own recordings.
   - Threshold + 2-frame confirmation; log every detection with score to `wake_log.jsonl`.
   - Pre-roll: keep 1.5 s of audio; on detection, hand KWS-detection + pre-roll + subsequent frames to ASR.
4. **VAD / endpointing.** Silero VAD (sherpa-onnx) over the frames; utterance starts on first speech, ends after **500 ms** silence (config; 350 ms for snappy mode). Enforce a max utterance (12 s) and a min (300 ms) to kill coughs and door slams.
5. **ASR — cloud (Lean, default).**
   - `CloudRecognizer` with two backends: **Gemini native audio** (audio bytes in → text; best measured Darija accuracy) and **Groq whisper-large-v3-turbo** (fast, huge free quota).
   - Send 16 kHz PCM (WAV/PCM16), `language_hint="ar"` or `en`, plus the **Darija lexicon** as a prompt/hotword list (names, apps, places).
   - Timeout 6 s → fall to the other backend → fall to local. Log which one served.
6. **ASR — local (Bunker / privacy).**
   - `LocalWhisperRecognizer`: faster-whisper, `compute_type="int8"`, `cpu_threads=3`, `beam_size=1`, `condition_on_previous_text=False`, VAD off (we already did it).
   - Model choice: **Darija** → a Moroccan fine-tune in CTranslate2 format; **English** → `small.en` or `base.en`.
     *Converting a HF Darija Whisper to CTranslate2 is a scripted step — turn it into `atlas-convert` (L9 artifact) and publish it.*
   - `SherpaOfflineRecognizer`: sherpa-onnx offline recogniser for MoulSot/Qwen3-ASR once an ONNX export exists (or you export it in L9). Keeps Bunker mode inside one runtime.
7. **ASR selection policy** (`RecognizerFactory` + `ResourceLease`):
   `Darija → cloud if online & quota → else local Darija → else whisper-multilingual-small`
   `English → cloud if online → else small.en → else whisper-multilingual`
   Never load two ASR engines at once on 8 GB.
8. **Language routing from audio.** Script detection on the transcript (Arabic vs Latin) + explicit switch phrases + sticky language. Handle code-switching gracefully: if a Darija sentence contains English words, keep Darija as the reply language unless the user asked otherwise.
9. **Post-processing.** Punctuation restoration (sherpa-onnx punctuation model), Darija lexicon corrections, numbers normalisation for spoken output ("2015" → "alfayn w khamsach"), and a confidence score you actually store (used later for confirmations and for the mood engine).
10. **Interaction loop (FSM-driven).**
    `Wake → chime → record until VAD end → ASR → L1 brain → TTS (L3) → follow-up window 2.5 s`.
    Include a "listen again without wake word" grace period so *"Atlas… chno ljaw? …w mn be3d?"* feels natural.
11. **Push-to-talk + hotkeys** for testing: `Ctrl+Alt+Space` records; `Ctrl+Alt+M` mutes the mic; `Ctrl+Alt+S` stops speech.
12. **Benchmark and choose.** `scripts/bench_asr.py`: 20 recordings (10 Darija, 10 en-GB), each engine, report WER-ish (count of errors by eye is fine) and realtime factor; write results to `docs/levels/NOTES-L02.md`.
13. **Privacy plumbing.** Local KWS always; cloud ASR only after wake; a `cloud_audio` flag in `preferences.md`; when cloud ASR is used the orb shows a subtle cloud glyph. Never upload pre-roll beyond the utterance window.

## Classes

`AudioDeviceManager` · `FrameProducer` · `FrameBus` · `RingBuffer` · `WakeWordEngine` (ABC) · `SherpaKwsEngine` · `OpenWakeWordEngine` · `VadSegmenter` · `Utterance` · `SpeechRecognizer` (ABC) · `CloudRecognizer` · `LocalWhisperRecognizer` · `SherpaOfflineRecognizer` · `RecognizerFactory` · `LanguageId` · `DarijaLexicon` · `AsrPostProcessor` · `TurnRecorder` (dumps replayable sessions).

## Tests

- `FakeMic` replays WAV fixtures → wake → utterance → transcript, with zero hardware in CI.
- Ring buffer: 1.5 s pre-roll never truncates the first word (assert against a fixture where the wake word abuts speech).
- VAD: synthetic silence/speech patterns → correct start/end; min/max utterance bounds.
- Recogniser contract suite: all three implementations return `Transcript{text, lang, confidence, engine, ms}`.
- Language routing: audio fixtures in Darija, English, and mixed → expected language.
- Half-duplex: while synthesising, no frames are consumed for ASR (assert on events).
- Replay regression: `--capture-dump` session → identical transcript across runs (determinism check).

## Done when

- 10 hands-free commands (5 Darija / 5 English) transcript correctly; the failures are logged honestly in `NOTES-L02.md`.
- 4 h idle: 0 false wakes, idle RAM ≤ 180 MB for KWS+VAD, CPU ≤ 5 % of one core.
- Cloud ASR p50 ≤ 1.2 s; local Darija p50 ≤ 3.5 s (Lean vs Bunker).
- Pull the Wi-Fi cable: Atlas keeps listening and answers via the local engine, saying it's offline.

## Pitfalls

- ❌ 48 kHz stereo capture "because the device does it" → convert at the edge, models want 16 kHz mono.
- ❌ Windows audio "enhancements" + AGC + exclusive mode → disable all three (they wreck VAD and WER).
- ❌ Reloading the ASR model per turn → keep it warm during a session, release on lease timeout only.
- ❌ Lowering the wake threshold to make testing easier → you'll ship a TV-activated assistant. Keep it high, accept repeats.
- ❌ Treating a transcript as ground truth for destructive actions → read back before acting (L7).
- ⚠️ Community Darija Whisper fine-tunes report roughly ~50 % WER in the wild — your lexicon file and cloud-primary policy are the real accuracy strategy, not a model swap.
- ⚠️ Teams/Zoom/OBS grab the mic. Handle the error, tell the user which app holds it, and retry (L10 watchdog).
