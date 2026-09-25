"""`atlas audio …` and `atlas listen` — the L2 commands, run for real.

CI has no microphone and no cloud keys, which is exactly the point: these tests
prove that a machine without audio hardware, without sherpa-onnx and without a
Gemini key still gets a *useful sentence* instead of a traceback.  The replay
path is exercised end to end, because that is the path the laptop will use.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from pathlib import Path

import pytest

from atlas.cli import main

REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG = """
[app]
profile = "lean"
language = "ar-MA"

[asr]
mode = "cloud_first"
cloud_audio = true

[mind]
provider_order = ["deadlocal"]

[[providers]]
name = "deadlocal"
kind = "openai_compatible"
model = "none"
base_url = "http://127.0.0.1:9/v1"
timeout_s = 1.0
"""


@pytest.fixture
def vc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Callable[..., tuple[int, str]]]:
    """A throwaway workspace with a config, and a runner that captures stdout."""
    (tmp_path / "config.toml").write_text(CONFIG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_CONFIG", str(tmp_path / "config.toml"))

    def run(*argv: str) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(list(argv))
        return code, buffer.getvalue()

    return tmp_path, run


# ── audio devices / test ─────────────────────────────────────────────
def test_audio_devices_reports_the_missing_stack_instead_of_crashing(vc) -> None:
    code, out = vc[1]("audio", "devices")
    assert "Audio devices" in out
    assert "sounddevice" in out
    try:
        import sounddevice  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, OSError):
        assert code == 1, "without the audio extra, `audio devices` must exit non-zero"
        assert "atlas-audio[audio]" in out
    else:  # pragma: no cover - only where the extra is installed
        assert code in (0, 1)


def test_audio_test_tells_you_what_to_install(vc) -> None:
    code, out = vc[1]("audio", "test", "--seconds", "0.05")
    assert "Microphone self-test" in out
    try:
        import sounddevice  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, OSError):
        assert code == 1
        assert "atlas-audio[audio]" in out
    else:  # pragma: no cover
        assert code in (0, 1)


# ── listen --status ──────────────────────────────────────────────────
def test_listen_status_explains_every_ear(vc) -> None:
    code, out = vc[1]("listen", "--status")
    assert code == 0
    assert "sherpa-kws" in out and "openwakeword" in out and "energy-wake" in out
    assert "asr mode: cloud_first" in out
    assert "wake log:" in out  # the false-wake number the gate needs
    # No key in this environment: it says so instead of pretending.
    assert "cloud keys: none" in out


def test_listen_status_reflects_the_cloud_audio_switch(vc, tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        CONFIG.replace("cloud_audio = true", "cloud_audio = false"), encoding="utf-8"
    )
    code, out = vc[1]("listen", "--status")
    assert code == 0
    assert "asr mode: local_only" in out  # the flag wins over the mode


# ── replay ───────────────────────────────────────────────────────────
def _record_a_session(workspace: Path) -> Path:
    """Write a session dump the way `atlas listen --capture-dump` would."""
    import asyncio
    import sys

    sys.path.insert(0, str(REPO_ROOT / "packages" / "atlas-audio" / "src"))
    from atlas_audio import TurnRecorder, VadSegmenter, VoiceLoop
    from atlas_audio.capture import silence, speech
    from atlas_audio.loop import LoopConfig
    from atlas_audio.wake import EnergyWakeEngine
    from atlas_core.contracts import ResourceCost, Transcript

    class Recognizer:
        name = "replay-asr"

        async def transcribe(self, audio, *, language="unknown", sample_rate=16000):
            return Transcript(
                text="chno ljaw", language="ar-MA", confidence=0.9, engine="replay-asr"
            )

        async def load(self) -> None: ...

        async def unload(self) -> None: ...

        def is_loaded(self) -> bool:
            return True

        def cost_hint(self) -> ResourceCost:
            return ResourceCost(ram_mb=1)

    async def run() -> Path:
        recorder = TurnRecorder(workspace / "data" / "recordings", session="demo")
        loop = VoiceLoop(
            wake=EnergyWakeEngine(confirm_frames=1),
            segmenter=VadSegmenter(),
            recognizer=Recognizer(),
            recorder=recorder,
            config=LoopConfig(),
        )
        loop.arm()
        audio = silence(4) + speech(4) + silence(8) + speech(14) + silence(8)
        await loop.feed_frames(audio)
        manifest = recorder.flush()
        assert manifest is not None, "the session manifest is what replay walks"
        return manifest

    return asyncio.run(run()).parent


def test_audio_replay_runs_a_recording_through_the_chain(vc, monkeypatch) -> None:
    """One stub — the recogniser — and everything else is the real chain."""
    workspace, run = vc
    session = _record_a_session(workspace)
    _stub_recognizer(monkeypatch)
    code, out = run("audio", "replay", str(session))
    assert "Replay" in out
    assert code == 0, out
    assert "chno ljaw" in out  # the synthesised recogniser echoed the clip
    assert "recorded vs replayed" in out  # and it compares against the manifest


def _stub_recognizer(monkeypatch) -> None:
    """Replace the ASR engine everywhere the CLI builds one.

    The wiring under test is "session → frames → loop → output", not the cloud
    client, and CI has neither keys nor models.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "packages" / "atlas-audio" / "src"))
    from atlas_audio import RecognizerFactory
    from atlas_audio.asr import RecognizerPolicy
    from atlas_core.contracts import ResourceCost, SpeechRecognizer, Transcript

    class StubRecognizer(SpeechRecognizer):
        """A recogniser that is honest about being a stub — and typed as one."""

        name = "stub-asr"

        async def transcribe(self, audio: bytes, *, language="unknown", sample_rate: int = 16000):
            return Transcript(text="chno ljaw", language="ar-MA", confidence=0.9, engine="stub-asr")

        async def load(self) -> None: ...

        async def unload(self) -> None: ...

        def is_loaded(self) -> bool:
            return True

        def cost_hint(self) -> ResourceCost:
            return ResourceCost(ram_mb=1)

    import atlas_audio

    def stub_build_recognizers(config, *, lexicon_prompt=None):
        stub = StubRecognizer()
        return {
            "cloud": stub,
            "local": {},
            "processor": atlas_audio.AsrPostProcessor(),
            "factory": RecognizerFactory(cloud=stub, policy=RecognizerPolicy(mode="cloud_only")),
        }

    monkeypatch.setattr(atlas_audio, "build_recognizers", stub_build_recognizers)


def test_audio_replay_on_a_wav_file_works_too(vc, monkeypatch) -> None:
    workspace, run = vc
    import sys

    sys.path.insert(0, str(REPO_ROOT / "packages" / "atlas-audio" / "src"))
    from array import array

    from atlas_audio import WavFile
    from atlas_audio.capture import speech
    from atlas_audio.frames import frames_to_pcm

    wav = WavFile(array("h", list(array("h", frames_to_pcm(speech(6))))))
    path = wav.write(workspace / "one.wav")
    _stub_recognizer(monkeypatch)
    code, out = run("audio", "replay", str(path))
    assert "Replay" in out
    # No wake word in the clip: the honest answer is "nothing transcribed".
    assert code == 1
    assert "nothing transcribed" in out


def test_audio_replay_rejects_a_missing_file(vc) -> None:
    code, out = vc[1]("audio", "replay", "nope/does-not-exist")
    assert code == 1
    assert "no such file" in out


def test_audio_rejects_an_unknown_action(vc) -> None:
    with pytest.raises(SystemExit):  # argparse rejects it before we run anything
        vc[1]("audio", "sing")
