"""Speech recognition: cloud first, local fallback, one engine warm at a time.

No network and no models: the cloud backends are driven through an
`httpx.MockTransport` and faster-whisper through an injected `model_factory`.
That is the same seam production uses, so these tests exercise the real code
paths — requests included.
"""

from __future__ import annotations

import asyncio
import json
import struct

import httpx
import pytest

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
from atlas_core.errors import ProviderUnavailable, RateLimited, Unsupported
from atlas_core.resources import ResourceLease


def pcm(ms: int = 400) -> bytes:
    return b"\x10\x20" * (16_000 * ms // 1000)


def client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── the WAV container ────────────────────────────────────────────────
def test_wav_header_is_a_real_wav_file():
    data = pcm(100)
    header = wav_header(data)
    assert header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    assert struct.unpack("<I", header[4:8])[0] == len(header) - 8
    assert struct.unpack("<H", header[20:22])[0] == 1  # PCM
    assert struct.unpack("<H", header[22:24])[0] == 1  # mono
    assert struct.unpack("<I", header[24:28])[0] == 16_000
    assert struct.unpack("<I", header[28:32])[0] == 16_000 * 2  # byte rate
    assert struct.unpack("<H", header[34:36])[0] == 16  # bits per sample
    assert struct.unpack("<I", header[40:44])[0] == len(data)  # data chunk size


# ── cloud ────────────────────────────────────────────────────────────
async def test_gemini_transcribes_inline_audio_and_reports_the_engine():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        body = json.loads(request.content)
        seen["payload"] = body
        return httpx.Response(
            200, json={"candidates": [{"content": {"parts": [{"text": "  chno ljaw  "}]}}]}
        )

    recognizer = CloudRecognizer(
        gemini_key="gk", client=client(handler), hotwords=lambda: "Zineb, Obsidian"
    )
    transcript = await recognizer.transcribe(pcm(), language="ar-MA")

    assert transcript.text == "chno ljaw"
    assert transcript.language == "ar-MA"
    assert transcript.engine == "cloud:gemini"
    assert transcript.duration_ms > 0
    assert "gemini-2.5-flash:generateContent" in seen["url"]
    assert seen["key"] == "gk"

    parts = seen["payload"]["contents"][0]["parts"]
    assert parts[0]["text"].startswith("Transcribe this audio")
    assert "Zineb, Obsidian" in parts[0]["text"]  # the lexicon as a hotword prompt
    assert parts[1]["inline_data"]["mime_type"] == "audio/wav"
    import base64

    sent = base64.b64decode(parts[1]["inline_data"]["data"])
    assert sent[:4] == b"RIFF" and len(sent) > len(pcm())
    assert seen["payload"]["generationConfig"]["temperature"] == 0.0


async def test_groq_is_used_when_gemini_is_unauthorised():
    def handler(request: httpx.Request) -> httpx.Response:
        if "generativelanguage" in str(request.url):
            return httpx.Response(403, json={"error": "bad key"})
        assert str(request.url).endswith("/openai/v1/audio/transcriptions")
        assert "multipart/form-data" in request.headers["content-type"]
        assert b'language' in request.content and b"ar" in request.content
        return httpx.Response(200, json={"text": "salam, chno ljaw", "segments": [{"no_speech_prob": 0.1}]})

    recognizer = CloudRecognizer(gemini_key="gk", groq_key="gk2", client=client(handler))
    transcript = await recognizer.transcribe(pcm(), language="ar-MA")

    assert transcript.text == "salam, chno ljaw"
    assert transcript.engine == "cloud:groq"
    assert transcript.confidence == pytest.approx(0.9)
    assert [attempt.engine for attempt in recognizer.attempts] == ["gemini", "groq"]
    assert [attempt.ok for attempt in recognizer.attempts] == [False, True]
    assert recognizer.attempts[0].as_dict()["ms"] >= 0.0


async def test_rate_limited_is_not_a_licence_to_keep_hammering():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(429, json={"error": "quota"})

    recognizer = CloudRecognizer(gemini_key="gk", groq_key="gk2", client=client(handler))
    with pytest.raises(RateLimited) as excinfo:
        await recognizer.transcribe(pcm(), language="en-GB")
    assert len(calls) == 2  # gemini then groq, then it stops
    # All backends rate-limited is a *quota* answer, not a generic failure: the
    # router shows the right message and the factory reaches for a local model.
    assert "gemini" in str(excinfo.value) and "groq" in str(excinfo.value)


async def test_every_cloud_backend_down_is_one_honest_error():
    recognizer = CloudRecognizer(
        gemini_key="gk", groq_key="gk2", client=client(lambda _: httpx.Response(500))
    )
    with pytest.raises(ProviderUnavailable) as excinfo:
        await recognizer.transcribe(pcm())
    assert "gemini" in str(excinfo.value) and "groq" in str(excinfo.value)


async def test_no_key_at_all_says_so_instead_of_posting_nothing():
    recognizer = CloudRecognizer(client=client(lambda _: httpx.Response(200, json={})))
    with pytest.raises(ProviderUnavailable, match="none configured"):
        await recognizer.transcribe(pcm())
    assert recognizer.backends == []
    assert (await recognizer.health()).ok is False


async def test_offline_flag_short_circuits_before_any_request():
    hits = 0

    def handler(_: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        nonlocal hits
        hits += 1
        return httpx.Response(200, json={})

    recognizer = CloudRecognizer(gemini_key="gk", client=client(handler), offline_check=lambda: True)
    with pytest.raises(ProviderUnavailable, match="offline"):
        await recognizer.transcribe(pcm())
    assert hits == 0
    assert (await recognizer.health()).ok is False


async def test_the_cloud_recognizer_is_an_engine_and_a_recognizer():
    recognizer = CloudRecognizer(gemini_key="gk", client=client(lambda _: httpx.Response(200, json={})))
    assert recognizer.is_loaded() is False
    await recognizer.load()
    assert recognizer.is_loaded() is True
    await recognizer.load()  # idempotent
    assert recognizer.cost_hint().ram_mb < 50  # cloud costs no local RAM worth speaking of
    await recognizer.aclose()


# ── local (faster-whisper) ───────────────────────────────────────────
class FakeSegment:
    def __init__(self, text: str, avg_logprob: float = -0.1) -> None:
        self.text = text
        self.avg_logprob = avg_logprob


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel — same call, no CPU spend."""

    def __init__(self, segments, language: str = "ar") -> None:
        self.segments = segments
        self.language = language
        self.calls: list[dict] = []

    def transcribe(self, samples, **kwargs):
        self.calls.append({"samples": samples, **kwargs})
        info = type("Info", (), {"language": self.language, "language_probability": 0.9})()
        return iter(self.segments), info


async def test_local_whisper_decodes_off_the_event_loop_with_the_plan_s_settings():
    model = FakeWhisperModel([FakeSegment(" daba ", -0.1), FakeSegment("3ndi chwiya l3ba", -0.3)])
    recognizer = LocalWhisperRecognizer(
        "models/whisper-darija-ct2", model_factory=lambda: model, key="darija"
    )
    transcript = await recognizer.transcribe(pcm(500), language="ar-MA")

    assert transcript.text == "daba 3ndi chwiya l3ba"
    assert transcript.language == "ar-MA"
    assert transcript.confidence == pytest.approx(0.8)  # 1 + mean(-0.1, -0.3)
    assert transcript.engine == "local:whisper-darija-ct2"
    assert recognizer.name == "whisper:darija"  # unique identity for the lease

    call = model.calls[0]
    assert call["language"] == "ar"
    assert call["beam_size"] == 1
    assert call["condition_on_previous_text"] is False
    assert call["vad_filter"] is False  # we already segmented
    assert len(call["samples"]) == 16_000 * 500 // 1000  # 500 ms of audio, as float32
    assert call["samples"].dtype.name == "float32"


async def test_local_whisper_caps_a_runaway_buffer_and_clamps_confidence():
    model = FakeWhisperModel([FakeSegment("long", -5.0)], language="en")
    recognizer = LocalWhisperRecognizer("small.en", model_factory=lambda: model, language="en-GB", key="english")
    transcript = await recognizer.transcribe(pcm(60_000), language="en-GB")

    assert len(model.calls[0]["samples"]) == 16_000 * 30  # 30 s hard cap
    assert transcript.confidence == 0.0  # clamped, never negative
    assert transcript.language == "en-GB"  # an explicit hint wins over the model

    other = FakeWhisperModel([FakeSegment("x", 0.5)], language="en")
    lenient = LocalWhisperRecognizer("m", model_factory=lambda: other, key="m")
    unknown = await lenient.transcribe(pcm(), language="unknown")
    assert unknown.confidence == 1.0  # logprob 0.5 → 1.5, clamped
    assert unknown.language == "en-GB"  # whisper's own verdict, mapped to ours


async def test_local_whisper_without_the_library_explains_the_install():
    recognizer = LocalWhisperRecognizer("models/whisper-darija-ct2", key="darija")
    try:
        import faster_whisper  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        with pytest.raises(Unsupported, match="faster-whisper is not installed"):
            await recognizer.load()
        assert (await recognizer.health()).ok is False
    else:  # pragma: no cover - only where the extra is installed
        health = await recognizer.health()
        assert health.ok is True


async def test_local_whisper_reports_a_missing_model_file(monkeypatch):
    """With the library present but the model absent, the error says which path."""
    import sys
    import types

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    recognizer = LocalWhisperRecognizer("models/nope-ct2", key="nope")
    with pytest.raises(Unsupported, match="model not found at models/nope-ct2"):
        await recognizer.load()
    # A Hugging Face repo id is *not* a missing local path: the library fetches it.
    from atlas_audio.asr import _is_missing_local_model

    assert _is_missing_local_model("ychafiqui/whisper-small-darija") is False
    assert _is_missing_local_model("small.en") is False


def test_sherpa_offline_recognizer_is_an_honest_stub():
    recognizer = SherpaOfflineRecognizer()
    assert recognizer.missing  # what is missing is named, not hidden
    with pytest.raises(Unsupported):
        asyncio.run(recognizer.transcribe(pcm()))


# ── the factory: one engine on 8 GB, cloud as the safety net ─────────
def _local(name: str, key: str) -> LocalWhisperRecognizer:
    return LocalWhisperRecognizer(
        f"models/{key}", model_factory=lambda: FakeWhisperModel([FakeSegment("ok")]), key=key
    )


async def test_acquiring_a_second_local_engine_evicts_the_first():
    darija, english = _local("darija", "darija"), _local("english", "english")
    factory = RecognizerFactory(
        local_models={"local:darija": darija, "local:english": english},
        policy=RecognizerPolicy(mode="local_only"),
    )
    await factory.choose("ar-MA")
    assert darija.is_loaded() is True
    await factory.choose("en-GB")
    assert english.is_loaded() is True
    assert darija.is_loaded() is False  # 8 GB cannot hold both
    resident = factory.snapshot()["resident_mb"]
    assert isinstance(resident, int) and resident <= 1200


async def test_a_lease_denial_falls_back_instead_of_failing():
    tight = ResourceLease(free_ram_mb=lambda: 1000, ram_floor_mb=400)
    recognizer = LocalWhisperRecognizer(
        "models/big", model_factory=lambda: FakeWhisperModel([FakeSegment("x")]), ram_mb=1200, key="big"
    )
    factory = RecognizerFactory(
        local_models={"local:darija": recognizer},
        lease=tight,
        policy=RecognizerPolicy(mode="local_only"),
    )
    with pytest.raises(Unsupported, match="no ASR engine available"):
        await factory.choose("ar-MA")
    assert factory.denials and "1200 MB" in factory.denials[0]


async def test_the_cloud_is_skipped_when_the_wire_is_down():
    cloud = CloudRecognizer(gemini_key="gk", client=client(lambda _: httpx.Response(200, json={})))
    factory = RecognizerFactory(cloud=cloud, online=lambda: False)
    with pytest.raises(Unsupported) as excinfo:
        await factory.choose("ar-MA")
    assert "offline" in str(excinfo.value)


async def test_transcribe_falls_back_to_local_when_the_cloud_fails_mid_sentence():
    cloud = CloudRecognizer(gemini_key="gk", client=client(lambda _: httpx.Response(500)))
    local = LocalWhisperRecognizer("models/darija", model_factory=lambda: FakeWhisperModel([FakeSegment("fallback")]), key="darija")
    factory = RecognizerFactory(cloud=cloud, local_models={"local:darija": local})
    transcript = await factory.transcribe(pcm(), language="ar-MA")
    assert transcript.text == "fallback"
    assert factory.last_choice == "local:darija"
    assert any("cloud" in denial for denial in factory.denials)
    await factory.aclose()


async def test_preference_order_matches_the_plan():
    policy = RecognizerPolicy()
    assert policy.order("ar-MA") == ["cloud", "local:darija", "local:multilingual"]
    assert policy.order("en-GB") == ["cloud", "local:english", "local:multilingual"]
    assert RecognizerPolicy(mode="local_only").order("ar-MA") == ["local:darija", "local:multilingual"]


# ── built from config ────────────────────────────────────────────────
def test_build_recognizers_from_the_real_config_object():
    from atlas_core.config import AppConfig

    config = AppConfig.model_validate(
        {
            "providers": [
                {"name": "gemini", "kind": "gemini", "model": "gemini-2.5-flash-lite", "api_key_env": "GEMINI_API_KEY"},
                {
                    "name": "groq",
                    "kind": "openai_compatible",
                    "model": "whisper-large-v3-turbo",
                    "api_key_env": "GROQ_API_KEY",
                    "base_url": "https://api.groq.com/openai/v1",
                },
            ],
            "asr": {"darija_model": "models/whisper-darija-ct2", "english_model": "small.en"},
        }
    )
    built = build_recognizers(config)
    assert isinstance(built["cloud"], CloudRecognizer)
    assert set(built["local"]) == {"local:darija", "local:english", "local:multilingual"}
    assert built["factory"].policy.order("ar-MA")[0] == "cloud"
    assert built["cloud"].order == ("gemini", "groq")  # from asr.cloud_order
    assert {recognizer.name for recognizer in built["local"].values()} == {
        "whisper:darija",
        "whisper:english",
        "whisper:multilingual",
    }


def test_asr_attempt_is_serialisable_for_the_timing_log():
    attempt = AsrAttempt("gemini", False, 812.5, "rate limited")
    assert attempt.as_dict() == {"engine": "gemini", "ok": False, "ms": 812.5, "detail": "rate limited"}


def test_a_mode_the_config_declares_but_nobody_implements_is_rejected():
    with pytest.raises(ValueError, match="unknown ASR mode"):
        RecognizerPolicy(mode="telepathy")


def test_cloud_only_never_touches_a_local_model():
    policy = RecognizerPolicy(mode="cloud_only")
    assert policy.order("ar-MA") == ["cloud"]
    assert policy.order("en-GB") == ["cloud"]
    factory = RecognizerFactory(local_models={"local:darija": _local("d", "darija")}, policy=policy)
    with pytest.raises(Unsupported):
        asyncio.run(factory.choose("ar-MA"))
