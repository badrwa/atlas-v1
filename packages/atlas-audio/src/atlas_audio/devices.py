"""Audio device plumbing — the part of L2 that can be built and tested without a mic.

Everything here is defensive: the plan's whole point is that a missing
`sounddevice` (or a laptop with no mic) must produce a *clear sentence*, never a
traceback at 2 a.m.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DeviceInfo:
    index: int
    name: str
    channels: int
    sample_rate: int
    kind: str  # "input" | "output"
    is_default: bool = False

    def label(self) -> str:
        default = " (default)" if self.is_default else ""
        return f"[{self.index}] {self.name} — {self.channels}ch @ {self.sample_rate}Hz{default}"


@dataclass
class AudioStackReport:
    """What the doctor prints — honest about what is missing and what to do."""

    sounddevice_available: bool = False
    numpy_available: bool = False
    inputs: list[DeviceInfo] = field(default_factory=list)
    outputs: list[DeviceInfo] = field(default_factory=list)
    host_apis: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.inputs) and bool(self.outputs) and not self.problems

    def summary(self) -> str:
        if not self.sounddevice_available:
            return "sounddevice not installed — run: pip install 'atlas-audio[audio]' (L2)"
        if not self.inputs:
            return "no input device found — check Settings → Privacy → Microphone"
        return f"{len(self.inputs)} input / {len(self.outputs)} output device(s)"


def _import_sounddevice():
    try:
        import sounddevice  # type: ignore[import-not-found]

        return sounddevice
    except (ImportError, OSError):
        return None


def list_devices() -> tuple[list[DeviceInfo], list[DeviceInfo], list[str]]:
    """Return (inputs, outputs, host_apis). Empty lists when PortAudio is absent."""
    sounddevice = _import_sounddevice()
    if sounddevice is None:
        return [], [], []

    try:
        raw = sounddevice.query_devices()
        host_apis = [api["name"] for api in sounddevice.query_hostapis()]
        default_in, default_out = sounddevice.default.device
    except Exception:
        return [], [], []

    inputs: list[DeviceInfo] = []
    outputs: list[DeviceInfo] = []
    for index, device in enumerate(raw):
        info = DeviceInfo(
            index=index,
            name=str(device["name"]),
            channels=int(device["max_input_channels"] or device["max_output_channels"]),
            sample_rate=int(device["default_samplerate"]),
            kind="input" if device["max_input_channels"] else "output",
            is_default=index in (default_in, default_out),
        )
        (inputs if info.kind == "input" else outputs).append(info)
    return inputs, outputs, host_apis


def check_audio_stack(required_sample_rate: int = 16_000) -> AudioStackReport:
    """Diagnostics for `atlas doctor`."""
    report = AudioStackReport()
    report.sounddevice_available = _import_sounddevice() is not None
    try:
        import numpy  # noqa: F401

        report.numpy_available = True
    except ImportError:
        report.problems.append("numpy missing — install 'atlas-audio[audio]'")

    report.inputs, report.outputs, report.host_apis = list_devices()

    if report.sounddevice_available and not report.inputs:
        report.problems.append("no microphone detected")
    if report.sounddevice_available and not report.outputs:
        report.problems.append("no speaker detected")
    if report.inputs and not any(
        device.sample_rate >= 16_000 for device in report.inputs
    ):
        report.problems.append(
            f"no device reports {required_sample_rate} Hz+ — resampling will be needed"
        )
    return report


def pick_device(devices: list[DeviceInfo], wanted: str) -> DeviceInfo | None:
    """Resolve a configured device *name* (survives index changes across reboots)."""
    if not wanted:
        return next((device for device in devices if device.is_default), None)
    needle = wanted.strip().lower()
    for device in devices:
        if needle == device.name.strip().lower():
            return device
    for device in devices:  # substring match, e.g. "microphone array"
        if needle in device.name.lower():
            return device
    return None


__all__ = [
    "AudioStackReport",
    "DeviceInfo",
    "check_audio_stack",
    "list_devices",
    "pick_device",
]
