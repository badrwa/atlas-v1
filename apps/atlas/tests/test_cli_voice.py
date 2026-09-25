"""`atlas say` and `atlas voice` — the L3 commands, run for real.

CI has no speakers, no Piper model and no Darija sidecar.  These tests prove that
the commands still do something *true* in that world: `say --out` writes a real
WAV through an injected voice, `voice` reports the chain it would use, and a
machine where every engine fails gets a sentence instead of a traceback.
"""

from __future__ import annotations

import contextlib
import io
import wave
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from atlas.cli import main
from atlas_core.contracts import AudioChunk, LanguageTag

CONFIG = """
[app]
profile = "lean"
language = "ar-MA"

[tts]
max_sentences = 3
cache_path = "data/tts-cache.sqlite3"

[mind]
provider_order = ["deadlocal"]

[[providers]]
name = "deadlocal"
kind = "openai_compatible"
model = "none"
base_url = "http://127.0.0.1:9/v1"
timeout_s = 1.0
"""


class StubVoice:
    """A voice with the engine's shape, minus the neural network."""

    name = "stub:darija"
    language: LanguageTag = "ar-MA"
    supports_prosody = True
    ram_mb = 1
    cold_start_s = 0.0

    def __init__(self, *, fail: bool = False, rate: int = 22050, seconds: float = 0.3) -> None:
        self.fail = fail
        self.rate = rate
        self.seconds = seconds
        self.texts: list[str] = []

    async def load(self) -> None:
        return None

    async def unload(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def cost_hint(self):
        from atlas_core.contracts import ResourceCost

        return ResourceCost(ram_mb=1, cold_start_s=0.0)

    def synthesize(
        self, text: str, *, voice: str = "", language: LanguageTag = "ar-MA"
    ) -> Iterator[AudioChunk]:
        yield from self.speak(text, voice=voice, language=language, prosody=None)

    def speak(self, text: str, *, voice: str = "", language: LanguageTag = "ar-MA", prosody=None):
        if self.fail:
            raise RuntimeError("stub is broken")
        self.texts.append(text)
        samples = int(self.rate * self.seconds)
        yield AudioChunk(pcm=b"\x11\x00" * samples, sample_rate=self.rate, final=False)
        yield AudioChunk(pcm=b"", sample_rate=self.rate, final=True)


@pytest.fixture
def vc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Callable[..., tuple[int, str]]]:
    (tmp_path / "config.toml").write_text(CONFIG, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_CONFIG", str(tmp_path / "config.toml"))

    def run(*argv: str) -> tuple[int, str]:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(list(argv))
        return code, buffer.getvalue()

    return tmp_path, run


def use_voice(monkeypatch: pytest.MonkeyPatch, *voices: StubVoice) -> None:
    """Inject the chain where `Mouth.from_config` will look for it."""
    import atlas_audio.speech as speech

    monkeypatch.setattr(speech, "build_voice_chain", lambda *a, **k: list(voices))


# ── say ──────────────────────────────────────────────────────────────
def test_say_writes_a_real_wav_through_the_pipeline(vc, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, run = vc
    use_voice(monkeypatch, StubVoice())

    code, out = run("say", "Salam. Chno ljaw?", "--out", "out.wav", "--language", "ar-MA")

    assert code == 0, out
    target = workspace / "out.wav"
    assert target.exists()
    with wave.open(str(target), "rb") as handle:
        assert handle.getframerate() == 16000, "the pipeline's canonical rate"
        assert handle.getnchannels() == 1
        assert handle.getnframes() > 0
    assert "first audio" in out
    assert "Salam." in out


def test_say_splits_what_it_says_into_sentences(vc, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, run = vc
    voice = StubVoice()
    use_voice(monkeypatch, voice)

    code, _out = run("say", "Salam. Safi.", "--out", "out.wav")
    assert code == 0
    assert voice.texts == ["Salam.", "Safi."]
    assert (workspace / "out.wav").exists()


def test_say_says_so_when_no_engine_can_speak(vc, monkeypatch: pytest.MonkeyPatch) -> None:
    _, run = vc
    use_voice(monkeypatch, StubVoice(fail=True), StubVoice(fail=True))

    code, out = run("say", "Salam.", "--out", "out.wav")
    assert code == 1
    assert "no voice" in out
    assert "atlas voice" in out


def test_say_falls_through_to_the_second_voice_mid_answer(vc, monkeypatch: pytest.MonkeyPatch) -> None:
    _workspace, run = vc
    working = StubVoice()
    use_voice(monkeypatch, StubVoice(fail=True), working)

    code, out = run("say", "Salam. Safi.", "--out", "out.wav")
    assert code == 0, out
    assert working.texts == ["Salam.", "Safi."]


def test_say_needs_something_to_say(vc) -> None:
    _, run = vc
    code, out = run("say", "")
    assert code == 1
    assert "nothing to say" in out


def test_say_reads_stdin_when_asked(monkeypatch: pytest.MonkeyPatch, vc) -> None:
    workspace, run = vc
    use_voice(monkeypatch, StubVoice())
    monkeypatch.setattr("sys.stdin", io.StringIO("Salam men stdin."))

    code, _ = run("say", "-", "--out", "out.wav")
    assert code == 0
    assert (workspace / "out.wav").exists()


# ── voice ────────────────────────────────────────────────────────────
def test_voice_reports_the_chain_and_the_missing_pieces(vc) -> None:
    _, run = vc
    code, out = run("voice")
    assert code == 0
    assert "piper · en_GB-alan-medium" in out
    assert "darija-tts sidecar" in out
    assert "ar-MA:" in out and "en-GB:" in out
    # On a machine with nothing installed, the fallback must be stated, not hidden.
    assert "not installed yet" in out or "unavailable" in out


def test_voice_cache_reports_and_clears_the_cache(vc) -> None:
    _, run = vc
    code, out = run("voice", "cache")
    assert code == 0
    assert "clips" in out

    code, out = run("voice", "cache", "--clear")
    assert code == 0
    assert "cleared" in out


def test_voice_test_speaks_a_line_to_a_file(vc, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace, run = vc
    use_voice(monkeypatch, StubVoice())
    code, out = run("voice", "test", "Salam.", "--out", "test.wav", "--language", "ar-MA")
    assert code == 0, out
    assert (workspace / "test.wav").exists()


def test_voice_warm_fills_the_cache_and_then_has_nothing_to_do(
    vc, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run = vc
    voice = StubVoice()
    use_voice(monkeypatch, voice)

    code, out = run("voice", "warm")
    assert code == 0, out
    assert "warmed" in out
    warmed = len(voice.texts)
    assert warmed >= 5, "the canned lines went through the voice"

    code, out = run("voice", "warm")
    assert code == 0
    assert len(voice.texts) == warmed, "second run is all cache hits"
    assert "0 line(s)" in out


def test_voice_warm_says_so_when_no_engine_can_speak(vc, monkeypatch: pytest.MonkeyPatch) -> None:
    _, run = vc
    use_voice(monkeypatch, StubVoice(fail=True))
    code, out = run("voice", "warm")
    assert code == 1
    assert "nothing warmed" in out


# ── the loop's half duplex ───────────────────────────────────────────
def test_the_mute_is_reference_counted_so_two_holders_cannot_open_the_mic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reply holds the mic shut, and so does the mouth's gate.  If the gate
    released first, the microphone would open while Atlas was still speaking."""
    from atlas_audio.loop import LoopConfig, VoiceLoop
    from atlas_audio.playback import MicGate
    from atlas_audio.vad import VadSegmenter
    from atlas_core.contracts import (
        Detection,
        FeedFrame,
        ResourceCost,
        SpeechRecognizer,
        Transcript,
        WakeWordEngine,
    )

    class SilentWake(WakeWordEngine):
        name = "silent"
        keywords: tuple[str, ...] = ()

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        def is_healthy(self) -> bool:
            return True

        def feed(self, frame: FeedFrame) -> Detection:
            return Detection()

        def is_loaded(self) -> bool:
            return True

        def cost_hint(self) -> ResourceCost:
            return ResourceCost(ram_mb=0, cold_start_s=0.0)

        async def load(self) -> None:
            return None

        async def unload(self) -> None:
            return None

    class NeverRecognizer(SpeechRecognizer):
        name = "never"

        async def transcribe(
            self, audio: bytes, *, language: LanguageTag = "ar-MA", sample_rate: int = 16000
        ) -> Transcript:
            raise AssertionError("no frame is fed in this test")

        def is_loaded(self) -> bool:
            return True

        def cost_hint(self) -> ResourceCost:
            return ResourceCost(ram_mb=0, cold_start_s=0.0)

        async def load(self) -> None:
            return None

        async def unload(self) -> None:
            return None

    monkeypatch.chdir(tmp_path)
    loop = VoiceLoop(
        wake=SilentWake(),
        segmenter=VadSegmenter(),
        recognizer=NeverRecognizer(),
        config=LoopConfig(),
    )
    loop.arm()
    gate = MicGate(on_close=loop.mute, on_open=loop.unmute)

    loop.mute()  # the reply is being streamed
    gate.close()  # the mouth is playing the first sentence
    gate.open()  # playback of the last sentence finished
    assert loop.mic_muted is True, "still speaking: the mic stays shut"
    loop.unmute()
    assert loop.mic_muted is False, "now it may hear the follow-up"
