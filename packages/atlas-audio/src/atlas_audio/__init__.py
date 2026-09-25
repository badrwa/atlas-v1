"""Atlas audio — the ears.

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
* `engines`     — the L3/L4 slots that are still deliberately unimplemented

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
from atlas_audio.engines import DarijaTtsSidecar, PiperSynthesizer, SherpaSpeakerVerifier
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
from atlas_audio.ptt import HOTKEYS, HotkeyListener, PushToTalk
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
    "FRAME_BYTES",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "HOTKEYS",
    "SAMPLE_RATE",
    "AsrAttempt",
    "AsrPostProcessor",
    "AudioStackReport",
    "CloudRecognizer",
    "DarijaTtsSidecar",
    "DeviceInfo",
    "EnergyVad",
    "EnergyWakeEngine",
    "Frame",
    "FrameBus",
    "FramePacket",
    "FrameProducer",
    "HotkeyListener",
    "LocalWhisperRecognizer",
    "LoopConfig",
    "LoopState",
    "OpenWakeWordEngine",
    "PiperSynthesizer",
    "PushToTalk",
    "RecognizerFactory",
    "RecognizerPolicy",
    "RecordedTurn",
    "RingBuffer",
    "SegmenterConfig",
    "SelfTestResult",
    "Session",
    "SherpaKwsEngine",
    "SherpaOfflineRecognizer",
    "SherpaSpeakerVerifier",
    "SileroVad",
    "Turn",
    "TurnRecorder",
    "Utterance",
    "VadSegmenter",
    "VoiceLoop",
    "WakeHit",
    "WakeLog",
    "WavFile",
    "build_recognizers",
    "build_wake_engine",
    "check_audio_stack",
    "detect_language",
    "downmix_and_resample",
    "frame_from_bytes",
    "frame_to_bytes",
    "frames_to_pcm",
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
    "pick_device",
    "platform_notes",
    "rms",
    "save_lexicon",
    "self_test",
    "silence",
    "speech",
    "to_numpy",
    "tone",
    "utterances_from_pcm",
    "wake_engine_status",
    "wav_header",
]
