"""Atlas audio — the ears and the mouth.

Layering, bottom to top:

* `frames`      — 80 ms frames, ring buffer with pre-roll, fan-out bus, producer thread
* `devices`     — what the machine actually has (sounddevice probing, honest reports)
* `capture`     — WAV read/write, session recording, the microphone self-test
* `vad`         — Silero (sherpa-onnx) or energy fallback, endpointing into utterances
* `wake`        — "atlas": sherpa-onnx KWS primary, openWakeWord alternative, hit log
* `asr`         — cloud first (Gemini / Groq), faster-whisper Darija fallback, one at a time
* `postprocess` — language detection, dictation numbers, the owner-editable lexicon
* `loop`        — the FSM-driven turn: wake → listen → transcribe → reply → follow-up
* `ptt`         — push-to-talk, terminal or hotkey, for when "atlas" is not polite
* `speaker`     — L4: sherpa-onnx voice embeddings, the enrolment session, the guard wiring
* `prosody`     — mood → rate/energy/expressiveness, with pitch deliberately absent
* `cache`       — SQLite LRU of synthesised clips, keyed by text+voice+rate+engine
* `tts`         — the voices: Piper (en-GB + Arabic), the Darija sidecar, SAPI
* `playback`    — PCM to the sound card, fades, metering, and the half-duplex gate
* `speech`      — sentence streaming and `Mouth`, the one object that speaks

Every level L0–L4 now has real implementations; there is no "planned engine" stub
left in this package, and `atlas doctor` reports each piece's true state instead.

Nothing here needs a microphone to be tested: every engine takes an injected
reader, detector or model factory, so the whole level runs from WAV files.
"""

from atlas_audio.asr import (
    AsrAttempt,
    CloudRecognizer,
    LocalWhisperRecognizer,
    RecognizerFactory,
    RecognizerPolicy,
    SherpaOfflineRecognizer,
    build_recognizers,
    wav_header,
)
from atlas_audio.cache import (
    CachedClip,
    CacheStats,
    TtsCache,
    cache_key,
)
from atlas_audio.capture import (
    RecordedTurn,
    SelfTestResult,
    Session,
    TurnRecorder,
    WavFile,
    load_session,
    pcm_to_wav_bytes,
    platform_notes,
    self_test,
    silence,
    speech,
    tone,
)
from atlas_audio.devices import (
    AudioStackReport,
    DeviceInfo,
    check_audio_stack,
    list_devices,
    pick_device,
)
from atlas_audio.frames import (
    FRAME_BYTES,
    FRAME_MS,
    FRAME_SAMPLES,
    SAMPLE_RATE,
    Frame,
    FrameBus,
    FramePacket,
    FrameProducer,
    RingBuffer,
    downmix_and_resample,
    frame_from_bytes,
    frame_to_bytes,
    frames_to_pcm,
    is_silence,
    new_frame,
    pcm_to_frames,
    rms,
    to_numpy,
)
from atlas_audio.loop import LoopConfig, LoopState, Turn, VoiceLoop
from atlas_audio.playback import (
    AudioPlayer,
    DuckingController,
    MicGate,
    NullWriter,
    PlaybackResult,
    SoundDeviceWriter,
    peak_level,
)
from atlas_audio.postprocess import (
    AsrPostProcessor,
    detect_language,
    is_arabic,
    learn_correction,
    load_lexicon,
    normalise_for_match,
    number_to_words,
    save_lexicon,
)
from atlas_audio.prosody import CALM, Prosody, ProsodyDirector, in_quiet_hours
from atlas_audio.ptt import HOTKEYS, HotkeyListener, PushToTalk
from atlas_audio.speaker import (
    ENROL_PROMPTS,
    EnrollmentSession,
    EnrollmentStep,
    SherpaSpeakerVerifier,
    build_identity,
    build_verifier,
    speaker_status,
    speech_ms,
    trim_silence,
)
from atlas_audio.speech import (
    CANNED_LINES,
    InterruptPolicy,
    Mouth,
    MouthConfig,
    SentenceSplitter,
    SentenceStreamer,
    SpokenSentence,
)
from atlas_audio.tts import (
    DarijaTtsSidecarSynthesizer,
    PiperSynthesizer,
    SapiSynthesizer,
    SidecarConfig,
    SidecarProcess,
    VoiceConfig,
    VoiceSynthesizer,
    build_synthesizer,
    build_voice_chain,
    engine_ready,
    http_health,
    tts_status,
    voice_problems,
    wav_bytes,
)
from atlas_audio.vad import (
    EnergyVad,
    SegmenterConfig,
    SileroVad,
    Utterance,
    VadSegmenter,
    utterances_from_pcm,
)
from atlas_audio.wake import (
    EnergyWakeEngine,
    OpenWakeWordEngine,
    SherpaKwsEngine,
    WakeHit,
    WakeLog,
    build_wake_engine,
    wake_engine_status,
)

__all__ = [
    "CALM",
    "CANNED_LINES",
    "ENROL_PROMPTS",
    "FRAME_BYTES",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "HOTKEYS",
    "SAMPLE_RATE",
    "AsrAttempt",
    "AsrPostProcessor",
    "AudioPlayer",
    "AudioStackReport",
    "CacheStats",
    "CachedClip",
    "CloudRecognizer",
    "DarijaTtsSidecarSynthesizer",
    "DeviceInfo",
    "DuckingController",
    "EnergyVad",
    "EnergyWakeEngine",
    "EnrollmentSession",
    "EnrollmentStep",
    "Frame",
    "FrameBus",
    "FramePacket",
    "FrameProducer",
    "HotkeyListener",
    "InterruptPolicy",
    "LocalWhisperRecognizer",
    "LoopConfig",
    "LoopState",
    "MicGate",
    "Mouth",
    "MouthConfig",
    "NullWriter",
    "OpenWakeWordEngine",
    "PiperSynthesizer",
    "PlaybackResult",
    "Prosody",
    "ProsodyDirector",
    "PushToTalk",
    "RecognizerFactory",
    "RecognizerPolicy",
    "RecordedTurn",
    "RingBuffer",
    "SapiSynthesizer",
    "SegmenterConfig",
    "SelfTestResult",
    "SentenceSplitter",
    "SentenceStreamer",
    "Session",
    "SherpaKwsEngine",
    "SherpaOfflineRecognizer",
    "SherpaSpeakerVerifier",
    "SidecarConfig",
    "SidecarProcess",
    "SileroVad",
    "SoundDeviceWriter",
    "SpokenSentence",
    "TtsCache",
    "Turn",
    "TurnRecorder",
    "Utterance",
    "VadSegmenter",
    "VoiceConfig",
    "VoiceLoop",
    "VoiceSynthesizer",
    "WakeHit",
    "WakeLog",
    "WavFile",
    "build_identity",
    "build_recognizers",
    "build_synthesizer",
    "build_verifier",
    "build_voice_chain",
    "build_wake_engine",
    "cache_key",
    "check_audio_stack",
    "detect_language",
    "downmix_and_resample",
    "engine_ready",
    "frame_from_bytes",
    "frame_to_bytes",
    "frames_to_pcm",
    "http_health",
    "in_quiet_hours",
    "is_arabic",
    "is_silence",
    "learn_correction",
    "list_devices",
    "load_lexicon",
    "load_session",
    "new_frame",
    "normalise_for_match",
    "number_to_words",
    "pcm_to_frames",
    "pcm_to_wav_bytes",
    "peak_level",
    "pick_device",
    "platform_notes",
    "rms",
    "save_lexicon",
    "self_test",
    "silence",
    "speaker_status",
    "speech",
    "speech_ms",
    "to_numpy",
    "tone",
    "trim_silence",
    "tts_status",
    "utterances_from_pcm",
    "voice_problems",
    "wake_engine_status",
    "wav_bytes",
    "wav_header",
]
