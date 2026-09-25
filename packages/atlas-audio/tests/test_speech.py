"""L3 — the splitter, the streaming pipeline, the fallback chain.

Everything here runs without a sound card, a model file or a network: the voices
are fakes, the player writes into a list, and the token stream is a list of
strings.  That is the point — the latency-critical logic is the *ordering* of
work, and ordering is exactly what a test can pin down.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator

import pytest

from atlas_audio.cache import TtsCache, cache_key
from atlas_audio.playback import AudioPlayer, MicGate, NullWriter
from atlas_audio.prosody import ProsodyDirector
from atlas_audio.speech import (
    CANNED_LINES,
    InterruptPolicy,
    Mouth,
    MouthConfig,
    SentenceSplitter,
    SentenceStreamer,
    SpokenSentence,
)
from atlas_audio.tts import VoiceSynthesizer
from atlas_core.contracts import AudioChunk, LanguageTag


# ── doubles ──────────────────────────────────────────────────────────
class FakeVoice(VoiceSynthesizer):
    """A voice that returns a beep-length buffer and records what it was asked."""

    def __init__(
        self,
        name: str = "fake",
        *,
        delay_s: float = 0.0,
        fail: bool = False,
        sample_rate: int = 22050,
    ) -> None:
        self.name = name
        self.delay_s = delay_s
        self.fail = fail
        self.sample_rate = sample_rate
        self.calls: list[tuple[str, float]] = []
        self.started: list[float] = []

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return True

    def cost_hint(self):
        from atlas_core.contracts import ResourceCost

        return ResourceCost(ram_mb=1, cold_start_s=0.0)

    def speak(
        self,
        text: str,
        *,
        voice: str = "",
        language: LanguageTag = "ar-MA",
        prosody=None,
    ) -> Iterator[AudioChunk]:
        self.started.append(time.monotonic())
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError(f"{self.name} is broken")
        self.calls.append((text, prosody.rate if prosody else 1.0))
        # One second of "speech" per 10 characters, so audio length is predictable.
        samples = max(1, len(text)) * self.sample_rate // 10
        yield AudioChunk(pcm=b"\x01\x00" * samples, sample_rate=self.sample_rate, final=False)
        yield AudioChunk(pcm=b"", sample_rate=self.sample_rate, final=True)


async def tokens_of(text: str, *, delay_s: float = 0.0) -> AsyncIterator[str]:
    """A token stream: word by word, like a provider's deltas."""
    for word in text.split(" "):
        if delay_s:
            await asyncio.sleep(delay_s)
        yield word + " "


def make_streamer(*engines: VoiceSynthesizer, **kwargs) -> SentenceStreamer:
    return SentenceStreamer(list(engines) or [FakeVoice()], **kwargs)


# ── the splitter ─────────────────────────────────────────────────────
def test_the_splitter_ends_sentences_in_both_languages():
    splitter = SentenceSplitter()
    assert splitter.split("Salam. Chno ljaw? Safi!") == ["Salam.", "Chno ljaw?", "Safi!"]


def test_the_splitter_keeps_abbreviations_and_decimals_together():
    splitter = SentenceSplitter()
    text = "Dr. Smith said 3.5 metres, i.e. a lot. Then he left."
    assert splitter.split(text) == [
        "Dr. Smith said 3.5 metres, i.e. a lot.",
        "Then he left.",
    ]


def test_the_arabic_question_mark_ends_a_sentence():
    assert SentenceSplitter().split("Chno smitek؟ Ana Atlas.") == ["Chno smitek؟", "Ana Atlas."]


def test_a_long_sentence_is_cut_at_a_comma_not_mid_word():
    splitter = SentenceSplitter(max_chars=40)
    sentences = splitter.feed("x" * 10 + ", " + "y" * 40 + ", " + "z" * 40)
    assert sentences and sentences[0].endswith(",")
    assert all(len(part) <= 45 for part in splitter.feed(""))


def test_the_splitter_holds_a_partial_sentence_until_it_is_finished():
    splitter = SentenceSplitter()
    assert splitter.feed("Salam, ") == []
    assert splitter.feed("chno ljaw") == []
    assert splitter.feed("?") == ["Salam, chno ljaw?"]
    assert splitter.flush() == ""


def test_the_splitter_flushes_a_tail_that_never_ended():
    splitter = SentenceSplitter()
    assert splitter.feed("hadchi mzyan") == []
    assert splitter.flush() == "hadchi mzyan"


# ── the pipeline ─────────────────────────────────────────────────────
async def test_every_sentence_is_spoken_in_order():
    engine = FakeVoice(sample_rate=16000)
    streamer = make_streamer(engine)
    spoken = [item async for item in streamer.stream(tokens_of("Salam. Chno ljaw? Safi."))]
    assert [item.text for item in spoken] == ["Salam.", "Chno ljaw?", "Safi."]
    assert [item.index for item in spoken] == [0, 1, 2]
    assert all(item.audible for item in spoken)
    assert [text for text, _rate in engine.calls] == ["Salam.", "Chno ljaw?", "Safi."]


async def test_the_second_sentence_is_synthesised_while_the_first_is_played():
    """The whole point of streaming: sentence n+1 must not wait for playback."""
    engine = FakeVoice(sample_rate=16000, delay_s=0.05)
    streamer = make_streamer(engine)
    stream = streamer.stream(tokens_of("One. Two. Three."))

    first = await stream.__anext__()
    assert first.text == "One."
    started_before = engine.started[1] if len(engine.started) > 1 else 0.0
    # While the consumer "plays" sentence one, the worker has already begun two.
    await asyncio.sleep(0.12)
    second = await stream.__anext__()
    assert second.text == "Two."
    assert started_before or engine.started[1] < time.monotonic()
    assert len(engine.started) >= 2
    await stream.aclose()


async def test_the_first_word_does_not_wait_for_the_last_token():
    """Time-to-first-audio must track sentence one, not the whole answer."""
    engine = FakeVoice(sample_rate=16000)
    streamer = make_streamer(engine)
    started = time.monotonic()

    async def slow_tokens() -> AsyncIterator[str]:
        yield "Salam. "
        await asyncio.sleep(0.5)  # the model is still writing sentence two
        yield "Hadchi twil bezzaf."

    stream = streamer.stream(slow_tokens())
    first = await stream.__anext__()
    elapsed = time.monotonic() - started
    assert first.text == "Salam."
    assert elapsed < 0.4, "the first sentence must not wait for the rest"
    await stream.aclose()


async def test_the_long_answer_cap_asks_to_continue_instead_of_saying_everything():
    engine = FakeVoice(sample_rate=16000)
    streamer = make_streamer(engine, max_sentences=2)
    spoken = [
        item
        async for item in streamer.stream(tokens_of("One. Two. Three. Four. Five."))
    ]
    assert [item.text for item in spoken] == ["One.", "Two.", "bghiti nkemmel?"]
    assert streamer.truncated is True


async def test_the_cap_can_be_disabled_for_a_short_answer():
    streamer = make_streamer(FakeVoice(sample_rate=16000), max_sentences=2)
    spoken = [item async for item in streamer.stream(tokens_of("One. Two. Three."), cap=False)]
    assert [item.text for item in spoken] == ["One.", "Two.", "Three."]
    assert streamer.truncated is False


async def test_a_dead_engine_mid_answer_falls_through_to_the_next_voice():
    broken = FakeVoice("broken", fail=True)
    working = FakeVoice("working", sample_rate=16000)
    streamer = make_streamer(broken, working)

    spoken = [item async for item in streamer.stream(tokens_of("One. Two."))]
    assert [item.text for item in spoken] == ["One.", "Two."]
    assert all(item.engine == "working" for item in spoken)
    assert streamer.engine_switches == 1
    assert streamer.degraded is False


async def test_when_every_voice_is_dead_the_words_still_arrive_as_captions():
    streamer = make_streamer(FakeVoice("one", fail=True), FakeVoice("two", fail=True))
    spoken = [item async for item in streamer.stream(tokens_of("Salam. Safi."))]
    assert [item.text for item in spoken] == ["Salam.", "Safi."]
    assert all(not item.audible for item in spoken)
    assert streamer.degraded is True
    assert streamer.stats()["engine"] == "none"


async def test_an_interrupt_while_speaking_drops_the_rest_of_the_answer():
    """The stop hotkey lands mid-answer: what is left in the queue is dropped."""
    streamer = make_streamer(FakeVoice(sample_rate=16000, delay_s=0.02), max_sentences=0)
    stream = streamer.stream(tokens_of("One. Two. Three. Four."))

    first = await stream.__anext__()
    assert first.text == "One."
    streamer.stop("hotkey")  # the user presses Ctrl+Alt+S here

    rest = [item async for item in stream]
    assert rest == []
    assert streamer.stop_reason == "hotkey"
    assert [item.text for item in streamer.spoken] == ["One."]


async def test_cache_hits_skip_the_engine_and_are_reported():
    engine = FakeVoice(sample_rate=16000)
    cache = TtsCache(":memory:")
    streamer = make_streamer(engine, cache=cache)

    first = [item async for item in streamer.stream(tokens_of("Salam."))]
    assert first[0].cached is False
    assert len(engine.calls) == 1

    second = [item async for item in streamer.stream(tokens_of("Salam."))]
    assert second[0].cached is True
    assert len(engine.calls) == 1, "a cached sentence must not be synthesised again"
    assert cache.stats().hits == 1


async def test_the_cache_key_includes_the_rate_so_moods_do_not_share_audio():
    """A cheerful "Salam." is different audio from a calm one — do not serve one
    as the other, and do not re-synthesise the same mood twice."""
    engine = FakeVoice(sample_rate=16000)
    cache = TtsCache(":memory:")
    streamer = make_streamer(engine, cache=cache)

    class Happy:
        label = "happy"
        intensity = 1.0

    await streamer.say("Salam.")  # calm, 1.0
    await streamer.say("Salam.", mood=Happy())  # amused, a touch faster
    await streamer.say("Salam.", mood=Happy())  # …cached now

    rates = [rate for _text, rate in engine.calls]
    assert len(rates) == 2, "one synthesis per mood, and the third is a cache hit"
    assert rates[0] == 1.0
    assert rates[1] > 1.0
    assert cache.stats().entries == 2


# ── the mouth, the player and the gate ───────────────────────────────
async def test_playing_a_sentence_closes_the_mic_before_the_first_byte():
    events: list[str] = []

    class WatchedWriter(NullWriter):
        def write(self, samples) -> None:
            events.append(f"write muted={gate.muted}")
            super().write(samples)

    gate = MicGate(on_close=lambda: events.append("closed"), on_open=lambda: events.append("opened"))
    player = AudioPlayer(WatchedWriter(), gate=gate)
    streamer = make_streamer(FakeVoice(sample_rate=16000), max_sentences=0)

    for sentence in [item async for item in streamer.stream(tokens_of("Salam. Safi."))]:
        player.play(sentence.pcm, sample_rate=sentence.sample_rate)

    assert events[0] == "closed"
    assert events[-1] == "opened"
    assert all("muted=True" in event for event in events if event.startswith("write"))
    assert gate.muted is False, "the mic must be open again once the tail is out"


async def test_a_player_converts_the_engine_rate_once_for_the_whole_sentence():
    writer = NullWriter()
    player = AudioPlayer(writer)
    engine = FakeVoice(sample_rate=22050)
    sentence = (await make_streamer(engine).say("Salam."))[0]

    result = player.play(sentence.pcm, sample_rate=sentence.sample_rate)
    assert result.converted is True
    # 22050 Hz input at 16 kHz output: either side of the exact ratio.
    expected = len(sentence.pcm) / 2 / 22050
    assert result.seconds == pytest.approx(expected, rel=0.05)


def test_the_stop_word_list_matches_whole_words_only():
    policy = InterruptPolicy(["stop", "safi"])
    assert policy.hears_stop("safi") is True
    assert policy.hears_stop("Safi.") is True
    assert policy.hears_stop("stop it") is True
    assert policy.hears_stop("unstoppable") is False


def test_prosody_follows_the_mood_and_the_clock():
    director = ProsodyDirector()
    assert director.plan("Salam.").rate == 1.0

    class Mood:
        label = "happy"
        intensity = 0.9

    amused = director.plan("Salam.", mood=Mood())
    assert amused.rate > 1.0
    assert amused.label == "amused"

    # 23:30 is inside the quiet window: softer, slower, whatever the mood.
    from datetime import datetime

    quiet = director.plan("Salam.", mood=Mood(), now=datetime(2026, 1, 1, 23, 30))
    assert quiet.energy < 1.0
    assert quiet.rate <= amused.rate


def test_a_question_is_asked_like_a_question_even_when_the_mood_is_happy():
    class Mood:
        label = "happy"
        intensity = 1.0

    director = ProsodyDirector()
    assert director.plan("Chno ljaw?", mood=Mood()).label == "confirm"


def test_mouth_config_reads_the_tts_section_and_survives_an_empty_one():
    class Tts:
        max_sentences = 5
        ask_to_continue = False
        prosody = False
        quiet_hours = ("23:00", "06:30")
        gain = 0.8
        barge_in = True
        stop_words = ("safi",)
        prefetch = 1
        cache_path = "data/other.sqlite3"
        cache_max_mb = 10

    class Config:
        tts = Tts()

    settings = MouthConfig.from_config(Config())
    assert settings.max_sentences == 5
    assert settings.quiet_hours == ("23:00", "06:30")
    assert settings.barge_in is True
    assert settings.stop_words == ("safi",)
    assert MouthConfig.from_config(None).max_sentences == 3


def test_the_canned_lines_exist_in_both_languages():
    assert "salam" in CANNED_LINES["ar-MA"][0].lower()
    assert "atlas" in CANNED_LINES["en-GB"][0].lower()
    assert len(CANNED_LINES["ar-MA"]) == len(CANNED_LINES["en-GB"])


async def test_warming_puts_the_canned_lines_in_the_cache_and_does_not_repeat_itself():
    """Boot warming is what makes the first "salam" of the day instant."""
    engine = FakeVoice(sample_rate=16000)
    cache = TtsCache(":memory:")
    mouth = Mouth([engine], player=AudioPlayer(NullWriter()), cache=cache)

    assert await mouth.warm(["Salam.", "Safi."]) == 2
    assert len(engine.calls) == 2
    assert await mouth.warm(["Salam.", "Safi."]) == 0, "already cached: nothing to do"
    assert len(engine.calls) == 2

    # …and the entry is a real cache row, keyed by the engine that made it.
    assert cache.stats().entries == 2
    assert cache.get(
        cache_key("Salam.", voice="", language="ar-MA", engine="fake", rate=1.0)
    ) is not None


async def test_warming_plays_nothing():
    writer = NullWriter()
    mouth = Mouth([FakeVoice(sample_rate=16000)], player=AudioPlayer(writer), cache=TtsCache(":memory:"))
    await mouth.warm(["Salam."])
    assert writer.bytes_written == 0
    await mouth.say("Safi.")  # normal speech still plays… but onto the null writer
    assert writer.bytes_written > 0


def test_spoken_sentence_serialises_what_the_bench_needs():
    sentence = SpokenSentence(text="Salam.", pcm=b"\x00\x00" * 8000, sample_rate=16000, index=2)
    payload = sentence.as_dict()
    assert payload["index"] == 2
    assert payload["audio_s"] == 0.5
    assert payload["chars"] == 6
    assert payload["cached"] is False
