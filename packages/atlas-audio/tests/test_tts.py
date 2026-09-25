"""L3 engines: Piper, the Darija sidecar, SAPI — and the policy that picks one.

No model file, no network, no sound card.  What is worth testing here is not the
neural network (the user's ears judge that — see NOTES-L03) but everything the
code decided: the chunk-shape adapter, the fallback order per language, the
sidecar's start-and-give-up logic, the cache's LRU, and the WAV round trip that
turns a proprietary engine's output into the pipeline's PCM.
"""

from __future__ import annotations

from array import array
from pathlib import Path

import pytest

from atlas_audio.cache import TtsCache, cache_key
from atlas_audio.capture import pcm_to_wav_bytes
from atlas_audio.playback import (
    AudioPlayer,
    DuckingController,
    MicGate,
    NullWriter,
    SoundDeviceWriter,
    apply_gain,
    fade,
    peak_level,
)
from atlas_audio.prosody import Prosody
from atlas_audio.tts import (
    DarijaTtsSidecarSynthesizer,
    PiperSynthesizer,
    SapiSynthesizer,
    SidecarConfig,
    SidecarProcess,
    VoiceConfig,
    _sapi_rate,
    build_synthesizer,
    build_voice_chain,
    engine_ready,
    http_health,
    tts_status,
    voice_problems,
    wav_bytes,
)
from atlas_core.contracts import AudioChunk
from atlas_core.errors import Unsupported


# ── doubles ──────────────────────────────────────────────────────────
class FakeVoice:
    """Whatever Piper hands back this year — `AudioChunk`-like today."""

    def __init__(self, chunk: object, *, error: Exception | None = None) -> None:
        self.chunk = chunk
        self.error = error
        self.config = type("Config", (), {"sample_rate": 22050})()
        self.kwargs: list[dict[str, float]] = []

    def synthesize(self, text: str, **kwargs):
        self.kwargs.append(kwargs)
        if self.error is not None:
            raise self.error
        yield self.chunk


class TupleVoice(FakeVoice):
    """The old shape: a plain `(bytes, rate)` tuple, no attributes."""

    def synthesize(self, text: str, **kwargs):
        self.kwargs.append(kwargs)
        yield self.chunk


class FakeProcess:
    def __init__(self, code: int | None = None) -> None:
        self.code = code
        self.terminated = False

    def poll(self) -> int | None:
        return self.code

    def terminate(self) -> None:
        self.terminated = True


def wav_of(seconds: float = 0.2, *, rate: int = 22050) -> bytes:
    samples = array("h", [1200] * int(rate * seconds))
    return pcm_to_wav_bytes(samples.tobytes(), sample_rate=rate)


# ── Piper ────────────────────────────────────────────────────────────
def test_piper_speaks_through_whatever_chunk_shape_it_returns():
    engine = PiperSynthesizer("models/tts/x.onnx", voice="x", voice_loader=lambda *a: None)
    engine._model = FakeVoice(
        type("Chunk", (), {"audio_int16_bytes": b"\x01\x00" * 100, "sample_rate": 22050})()
    )
    chunks = list(engine.speak("Salam."))
    assert chunks[-1].final is True
    assert chunks[0].sample_rate == 22050
    assert len(chunks[0].pcm) == 200


def test_piper_accepts_the_older_tuple_chunk():
    engine = PiperSynthesizer("models/tts/x.onnx", voice="x", voice_loader=lambda *a: None)
    engine._model = TupleVoice((b"\x02\x00" * 50, 16000))
    chunks = list(engine.speak("Salam."))
    assert chunks[0].sample_rate == 16000
    assert len(chunks[0].pcm) == 100


def test_piper_passes_prosody_through_and_survives_a_version_that_cannot():
    engine = PiperSynthesizer("models/tts/x.onnx", voice="x", voice_loader=lambda *a: None)
    voice = FakeVoice(b"a")
    engine._model = voice
    list(engine.speak("Salam.", prosody=Prosody(rate=0.9, expressiveness=1.2)))
    assert voice.kwargs[0]["length_scale"] == pytest.approx(1 / 0.9, abs=0.001)

    def refuses(text: str, **kwargs):
        if kwargs:
            raise TypeError("synthesize() takes no keyword arguments")
        yield b"a"

    engine._model = FakeVoice(b"a")
    engine._model.synthesize = refuses  # type: ignore[method-assign]
    assert list(engine.speak("Salam.", prosody=Prosody(rate=0.9)))
    assert engine.supports_prosody is False, "say so once, then stop trying"


def test_piper_refuses_to_pretend_when_the_model_is_missing(tmp_path: Path):
    engine = PiperSynthesizer(tmp_path / "nope.onnx", voice="en_GB-alan-medium")
    assert engine.available() is False
    assert "en_GB-alan-medium.onnx" in engine.missing()
    with pytest.raises(Unsupported):
        list(engine.speak("hello"))


def test_piper_uses_two_names_for_two_voices_so_the_lease_cannot_confuse_them():
    english = PiperSynthesizer("models/tts/a.onnx", voice="en_GB-alan-medium")
    arabic = PiperSynthesizer("models/tts/b.onnx", voice="ar_JO-kareem-medium")
    assert english.name != arabic.name
    assert english.name.startswith("piper:")


# ── the Darija sidecar ───────────────────────────────────────────────
def test_the_sidecar_synthesizer_turns_a_wav_response_into_chunks():
    posted: list[dict[str, object]] = []

    def poster(url: str, payload: dict[str, object]) -> bytes:
        posted.append({"url": url, **payload})
        return wav_of(0.1, rate=24000)

    engine = DarijaTtsSidecarSynthesizer(poster=poster)
    chunks = list(engine.speak("Salam, chno ljaw?", prosody=Prosody(rate=0.95)))
    assert chunks[0].sample_rate == 24000
    assert len(chunks[0].pcm) == 4800
    assert posted[0]["text"] == "Salam, chno ljaw?"
    assert posted[0]["rate"] == 0.95
    assert engine.last_ms >= 0


def test_the_sidecar_synthesizer_refuses_an_empty_response():
    engine = DarijaTtsSidecarSynthesizer(poster=lambda url, payload: b"")
    with pytest.raises(Unsupported, match="no audio"):
        list(engine.speak("Salam."))


async def test_the_sidecar_is_started_on_demand_and_reported_ready():
    started: list[list[str]] = []
    health = {"up": False}

    def runner(argv, **kwargs):
        started.append(list(argv))
        health["up"] = True  # "it came up" — the poll then sees health
        return FakeProcess()

    process = SidecarProcess(
        SidecarConfig(url="http://127.0.0.1:9", python="python", timeout_s=2.0),
        runner=runner,
        health=lambda url: health["up"],
    )
    assert await process.ensure_started() is True
    assert started and started[0][0] == "python"
    assert process.running() is True
    process.stop()
    assert process.process is None


async def test_a_sidecar_that_never_answers_health_gives_up_instead_of_hanging():
    ticks = {"n": 0}

    def health(url: str) -> bool:
        ticks["n"] += 1
        return False

    process = SidecarProcess(
        SidecarConfig(url="http://127.0.0.1:9", python="python", timeout_s=0.3),
        runner=lambda argv, **kwargs: FakeProcess(),
        health=health,
    )
    assert await process.ensure_started() is False
    assert ticks["n"] > 1, "it polls rather than returning on the first miss"


async def test_a_sidecar_that_exits_immediately_is_not_waited_for():
    process = SidecarProcess(
        SidecarConfig(url="http://127.0.0.1:9", python="python", timeout_s=5.0),
        runner=lambda argv, **kwargs: FakeProcess(code=1),
        health=lambda url: False,
    )
    assert await process.ensure_started() is False
    assert process.running() is False


async def test_autostart_off_means_a_missing_sidecar_is_reported_not_started():
    calls: list[object] = []
    process = SidecarProcess(
        SidecarConfig(url="http://127.0.0.1:9", autostart=False),
        runner=lambda *a, **k: calls.append(a),
        health=lambda url: False,
    )
    assert await process.ensure_started() is False
    assert calls == []


async def test_a_sidecar_with_no_interpreter_is_never_spawned():
    process = SidecarProcess(
        SidecarConfig(url="http://127.0.0.1:9", python="/nope/python", timeout_s=0.2),
        runner=lambda *a, **k: pytest.fail("must not spawn without an interpreter"),
        health=lambda url: False,
    )
    assert await process.ensure_started() is False


def test_http_health_says_no_instead_of_raising():
    assert http_health("http://127.0.0.1:9", timeout_s=0.1) is False


# ── SAPI ─────────────────────────────────────────────────────────────
def test_sapi_maps_a_rate_multiplier_onto_its_minus_ten_to_ten_scale():
    assert _sapi_rate(1.0) == 0
    assert _sapi_rate(0.5) == -10
    assert _sapi_rate(2.0) == 10
    assert _sapi_rate(0.9) == -2


def test_sapi_is_unavailable_off_windows_and_says_why():
    engine = SapiSynthesizer(platform="linux")
    assert engine.available() is False
    with pytest.raises(Unsupported, match="Windows"):
        list(engine.speak("hello"))


def test_sapi_passes_the_text_through_a_temp_file(tmp_path: Path, monkeypatch):
    """Never through the command line: text is model output, not shell input."""
    captured: dict[str, str] = {}
    engine = SapiSynthesizer(platform="win32")

    def runner(script: str) -> None:
        captured["script"] = script
        out = Path(script.split("'")[0] if False else _temp_out(script))
        out.write_bytes(wav_of(0.05, rate=22050))

    monkeypatch.setattr("atlas_audio.tts._run_powershell", runner)
    engine._runner = runner
    assert engine.available() is True
    chunks = list(engine.speak("salam"))
    assert chunks[0].sample_rate == 22050
    assert "ReadAllText" in captured["script"]


def _temp_out(script: str) -> str:
    marker = "SetOutputToWaveFile('"
    start = script.index(marker) + len(marker)
    return script[start : script.index("'", start)]


# ── the policy ───────────────────────────────────────────────────────
def test_the_chain_leads_with_the_voice_that_matches_the_language():
    assert [e.name for e in build_voice_chain(None, language="en-GB")] == [
        "piper:en_GB-alan-medium",
        "sapi",
    ]
    assert [e.name for e in build_voice_chain(None, language="ar-MA")] == [
        "darija-tts",
        "piper:ar_JO-kareem-medium",
        "sapi",
    ]


def test_an_explicit_engine_leads_the_chain_and_keeps_a_fallback():
    chain = build_voice_chain(None, language="ar-MA", engine="piper_arabic")
    assert chain[0].name == "piper:ar_JO-kareem-medium"
    assert chain[-1].name == "sapi"
    assert len(chain) == 3


def test_an_unknown_engine_name_is_a_config_error_not_a_silent_fallback():
    with pytest.raises(ValueError, match="unknown tts engine"):
        build_voice_chain(None, language="en-GB", engine="elevenlabs")


def test_nothing_downloaded_means_none_of_the_local_engines_is_ready(tmp_path: Path):
    class Tts:
        models_dir = str(tmp_path)
        english_engine = "piper"
        english_voice = "en_GB-alan-medium"
        darija_engine = "piper_arabic"
        arabic_voice = "ar_JO-kareem-medium"

    class Config:
        tts = Tts()

    chain = build_voice_chain(Config(), language="en-GB")
    assert engine_ready(chain[0], Config()) is False
    # `build_synthesizer` returns the primary anyway: the chain is the answer,
    # and the Mouth walks it at the first sentence.
    chosen = build_synthesizer(Config(), language="en-GB")
    assert chosen is not None and chosen.name == chain[0].name


def test_voice_problems_explains_the_missing_sidecar_and_the_missing_model(tmp_path: Path):
    class Tts:
        models_dir = str(tmp_path)
        sidecar_python = str(tmp_path / "nope" / "python.exe")
        sidecar_url = "http://127.0.0.1:9"

    class Config:
        tts = Tts()

    problems = voice_problems(Config(), language="ar-MA")
    assert len(problems) == 3  # sidecar, arabic piper, sapi
    assert "vendor/darija-tts/README.md" in problems[0]
    assert "ar_JO-kareem-medium.onnx" in problems[1]


def test_the_status_table_names_every_engine():
    rows = dict(tts_status(None))
    assert "darija-tts sidecar" in rows
    assert any(key.startswith("piper · en_GB-alan-medium") for key in rows)
    assert "sapi" in rows


def test_voice_config_reads_the_tts_section_and_maps_languages():
    class Tts:
        english_engine = "sapi"
        english_voice = "en_GB-alan-medium"
        darija_engine = "piper_arabic"
        darija_voice = "darija"
        models_dir = "models/tts"

    class Config:
        tts = Tts()

    voices = VoiceConfig.from_config(Config())
    assert voices.for_language("en-GB") == ("sapi", "en_GB-alan-medium")
    assert voices.for_language("ar-MA") == ("piper_arabic", "darija")
    assert voices.piper_path("x").endswith("models/tts/x.onnx")


def test_wav_bytes_collects_a_synthesis_into_one_header():
    chunks = iter(
        [
            AudioChunk(pcm=b"\x01\x00" * 10, sample_rate=16000),
            AudioChunk(pcm=b"", sample_rate=16000, final=True),
        ]
    )
    data, rate = wav_bytes(chunks)
    assert rate == 16000
    assert data[:4] == b"RIFF"


# ── the cache ────────────────────────────────────────────────────────
def test_the_cache_key_is_stable_across_processes_and_sensitive_to_the_voice():
    first = cache_key("Salam.", voice="a", language="ar-MA", engine="piper", rate=1.0)
    assert first == cache_key("  salam.  ", voice="a", language="ar-MA", engine="piper", rate=1.0)
    assert first != cache_key("Salam.", voice="b", language="ar-MA", engine="piper", rate=1.0)
    assert first != cache_key("Salam.", voice="a", language="ar-MA", engine="piper", rate=1.05)


def test_the_cache_round_trips_a_clip_and_counts_its_uses():
    cache = TtsCache(":memory:")
    key = cache_key("Salam.")
    cache.put(key, b"\x00\x01" * 100, sample_rate=22050, engine="piper", text="Salam.")

    clip = cache.get(key)
    assert clip is not None
    assert clip.sample_rate == 22050
    assert len(clip.pcm) == 200
    assert cache.get("missing") is None
    assert cache.stats().hits == 1
    assert cache.stats().misses == 1
    assert cache.stats().hit_rate == 0.5


def test_the_cache_evicts_the_least_recently_used_clip_to_stay_inside_its_cap():
    cache = TtsCache(":memory:", max_mb=0.0015)  # 1.5 kB: two 600-byte clips fit
    clip = b"\x00" * 600
    cache.put("old", clip, sample_rate=16000)
    cache.put("new", clip, sample_rate=16000)
    assert cache.stats().entries == 2

    cache.get("new")  # "old" is now the least recently used
    cache.put("newest", clip, sample_rate=16000)
    assert cache.stats().entries == 2, "one clip evicted, not the whole batch"
    assert cache.get("old") is None
    assert cache.get("newest") is not None


def test_checking_for_a_clip_does_not_count_as_a_hit_or_a_miss():
    """Warming the cache at boot is not a user listening."""
    cache = TtsCache(":memory:")
    assert cache.has(cache_key("Salam.")) is False
    assert cache.stats().hits == 0
    assert cache.stats().misses == 0

    cache.put(cache_key("Salam."), b"\x00" * 10, sample_rate=16000)
    assert cache.has(cache_key("Salam.")) is True
    assert cache.stats().hits == 0


def test_warming_reports_exactly_which_keys_are_missing():
    cache = TtsCache(":memory:")
    cache.put(cache_key("Safi."), b"\x00" * 10, sample_rate=16000)
    keys = [cache_key(text) for text in ("Safi.", "Salam.", "Daba.")]
    assert cache.missing(keys) == [cache_key("Salam."), cache_key("Daba.")]


def test_the_cache_can_be_cleared_and_survives_being_reopened(tmp_path: Path):
    path = tmp_path / "cache.sqlite3"
    first = TtsCache(path)
    first.put(cache_key("Salam."), b"\x00" * 8, sample_rate=16000)
    assert first.clear() == 1
    assert first.stats().entries == 0

    second = TtsCache(path)
    second.put(cache_key("Safi."), b"\x00" * 8, sample_rate=16000)
    reopened = TtsCache(path)
    assert reopened.stats().entries == 1
    assert reopened.get(cache_key("Safi.")) is not None


# ── playback ─────────────────────────────────────────────────────────
def test_the_fade_kills_the_click_without_eating_a_short_reply():
    long_clip = array("h", [10000] * 1600)
    faded = fade(long_clip, frames=160)
    assert faded[0] == 0
    assert faded[-1] < 100
    assert faded[800] == 10000

    short_clip = array("h", [10000] * 100)
    assert fade(short_clip, frames=160) is short_clip, "too short to fade, leave it alone"


def test_gain_clips_instead_of_wrapping_around():
    samples = array("h", [30000, -30000, 100])
    doubled = apply_gain(samples, 2.0)
    assert list(doubled) == [32767, -32768, 200]
    assert apply_gain(array("h", [1, 2]), 1.0) == array("h", [1, 2])


def test_peak_level_is_a_fraction_of_full_scale():
    assert peak_level(array("h", [])) == 0.0
    assert peak_level(array("h", [0, 32767])) == pytest.approx(1.0, abs=0.001)
    assert peak_level(array("h", [0, 16384])) == pytest.approx(0.5, abs=0.001)


def test_ducking_multiplies_the_base_gain_and_can_be_restored():
    ducking = DuckingController(base_gain=0.8, duck_factor=0.5)
    assert ducking.gain == pytest.approx(0.8)
    ducking.duck()
    assert ducking.gain == pytest.approx(0.4)
    ducking.restore()
    assert ducking.gain == pytest.approx(0.8)
    assert DuckingController(base_gain=9).base_gain == 2.0, "clamped"


def test_the_player_meters_what_it_plays_not_what_it_captures():
    levels: list[float] = []
    player = AudioPlayer(NullWriter(), level_hz=50, on_level=levels.append)
    player.play(b"\x00\x40" * 16000, sample_rate=16000)  # one second at ~0.5
    assert levels, "the orb needs levels even with no sound card"
    assert max(levels) == pytest.approx(0.5, abs=0.01)
    assert len(levels) >= 40, "~50 levels per second of audio"


def test_the_gate_is_idempotent_and_always_reopens():
    events: list[str] = []
    gate = MicGate(on_close=lambda: events.append("shut"), on_open=lambda: events.append("open"))
    gate.close()
    gate.close()
    assert events == ["shut"]
    with gate.guarded():
        assert gate.muted is True
    assert gate.muted is False
    assert events == ["shut", "open"]


def test_playing_a_tone_is_the_wake_chime_and_needs_no_file():
    writer = NullWriter()
    player = AudioPlayer(writer)
    result = player.play_tones(2, freq=880)
    assert result.seconds > 0.1
    assert writer.bytes_written == result.bytes_written


def test_the_sound_device_writer_reports_a_missing_backend_instead_of_crashing():
    """CI and a fresh Windows box both land here; the answer must be a bool."""
    assert isinstance(SoundDeviceWriter.available(), bool)
