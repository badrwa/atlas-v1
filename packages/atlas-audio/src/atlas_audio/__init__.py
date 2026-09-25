"""Atlas audio — device plumbing now, engines in L2/L3."""

from atlas_audio.devices import (
    AudioStackReport,
    DeviceInfo,
    check_audio_stack,
    list_devices,
    pick_device,
)
from atlas_audio.engines import (
    CloudRecognizer,
    DarijaTtsSidecar,
    LocalWhisperRecognizer,
    PiperSynthesizer,
    SherpaKws,
    SherpaSpeakerVerifier,
    SileroVad,
)

__all__ = [
    "AudioStackReport",
    "CloudRecognizer",
    "DarijaTtsSidecar",
    "DeviceInfo",
    "LocalWhisperRecognizer",
    "PiperSynthesizer",
    "SherpaKws",
    "SherpaSpeakerVerifier",
    "SileroVad",
    "check_audio_stack",
    "list_devices",
    "pick_device",
]
