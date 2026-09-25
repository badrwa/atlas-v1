"""Sentence streaming: the single biggest perceived-latency win in the project.

Speaking an answer only after the model has finished writing it wastes the one
thing a language model is good at: producing text *incrementally*.  Atlas instead
splits the token stream into sentences and starts synthesising the first one while
the second is still being written — so the first word arrives in the time it takes
to say one sentence, not three.

The pipeline has three stages, each its own task, each connected by a **bounded**
queue:

    tokens ──▶ splitter ──▶ [ sentences ] ──▶ synthesizer ──▶ [ audio ] ──▶ playback

Bounded at two sentences in each stage on purpose: the queues are the memory
budget.  A long answer cannot sit in RAM as audio on an 8 GB laptop, and
backpressure from the speakers is exactly what should slow the model down.

Why a background pump instead of a plain generator: a generator suspends while
the consumer plays, so sentence *n+1* would only be synthesised after sentence
*n* had finished playing — the whole point undone.  The pump keeps running.

The splitter is abbreviation-aware in both languages, because "Dr." and "e.g."
are not the end of a sentence, and `3.5` is not two sentences.  Darija uses the
same `.`/`!`/`?` (plus `؟`) with `،` as a soft break for long sentences.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from atlas_audio.cache import TtsCache, cache_key
from atlas_audio.playback import AudioPlayer, MicGate, SoundDeviceWriter
from atlas_audio.prosody import Prosody, ProsodyDirector
from atlas_audio.tts import VoiceSynthesizer, build_voice_chain
from atlas_core.contracts import LanguageTag

log = logging.getLogger(__name__)

#: Sentence endings, in both scripts.  `؟` is the Arabic question mark and `…`
#: the ellipsis; a Darija sentence ends with any of them.
TERMINATORS = ".!?؟…"

#: Closing punctuation that may follow a terminator ("safi." then `"`).
CLOSERS = "\"'»”’)]}"

#: Words whose trailing dot is part of the word, not the sentence.
ABBREVIATIONS = frozenset(
    {
        "dr", "mr", "mrs", "ms", "prof", "st", "vs", "etc", "no", "fig", "al",
        "e.g", "i.e", "a.m", "p.m", "approx", "min", "max", "sec", "hr", "kg",
        "km", "cm", "mm", "cf", "bgh", "d", "j", "av", "bd", "ch", "3la", "wal",
    }
)

#: Soft breaks — used to cut a sentence that is too long to wait for.
SOFT_BREAKS = "،,;:—–"

#: The continuation prompt, in the language the plan chose for it.
CONTINUE_QUESTION = "bghiti nkemmel?"

#: Lines Atlas says over and over.  Warming these at boot (in a background
#: thread, below normal priority) turns the first "salam" of the day from a
#: synthesis into a cache read — and they are the lines a user judges speed by.
CANNED_LINES: dict[str, tuple[str, ...]] = {
    "ar-MA": (
        "Salam, ana Atlas.",
        "Safi.",
        "Daba.",
        "Wakha, ghadi n3awnek.",
        "Ma fhemtsh, 3awed 3afak.",
        CONTINUE_QUESTION,
    ),
    "en-GB": (
        "Hello, this is Atlas.",
        "Right.",
        "In a moment.",
        "Of course, I will help with that.",
        "I did not catch that — say it again?",
        "Shall I go on?",
    ),
}


class SentenceSplitter:
    """Incremental, abbreviation-safe sentence splitting.

    `feed()` takes whatever arrived and returns the sentences that are now
    complete; `flush()` returns the tail when the stream ends mid-sentence.  The
    buffer is the *only* state, which is what makes it testable with golden cases.
    """

    def __init__(self, *, max_chars: int = 220, min_chars: int = 2) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.buffer = ""

    def reset(self) -> None:
        self.buffer = ""

    def feed(self, delta: str) -> list[str]:
        self.buffer += delta
        out: list[str] = []
        while (boundary := self._boundary(self.buffer)) is not None:
            sentence = self.buffer[:boundary].strip()
            self.buffer = self.buffer[boundary:].lstrip()
            if len(sentence) >= self.min_chars:
                out.append(sentence)
        if len(self.buffer) > self.max_chars and (cut := self._soft_cut(self.buffer)):
            head = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:].lstrip()
            if head:
                out.append(head)
        return out

    def flush(self) -> str:
        """Everything left, as one last sentence (may be empty)."""
        tail = self.buffer.strip()
        self.buffer = ""
        return tail if len(tail) >= self.min_chars else ""

    def split(self, text: str) -> list[str]:
        """Whole-string convenience: feed + flush."""
        out = self.feed(text)
        if tail := self.flush():
            out.append(tail)
        return out

    # ── the rules ────────────────────────────────────────────────────
    def _boundary(self, text: str) -> int | None:
        """Index just past the first sentence-ending terminator, or None."""
        for index, char in enumerate(text):
            if char not in TERMINATORS:
                continue
            following = text[index + 1] if index + 1 < len(text) else ""
            closing = not following or following.isspace() or following in CLOSERS
            # "3.5" mid-token and "safi.9" are not sentence endings: a dot inside
            # a word only ends a sentence if what follows could start one.
            if not closing and (char == "." or following not in TERMINATORS):
                continue
            if char == "." and self._dot_is_internal(text, index):
                continue
            return index + 1
        return None

    def _dot_is_internal(self, text: str, index: int) -> bool:
        """A dot that belongs to a number, an abbreviation or an initial."""
        previous = text[index - 1] if index else ""
        following = text[index + 1] if index + 1 < len(text) else ""
        if following == ".":  # the middle of an ellipsis or a leader ".."
            return True
        if previous == ".":  # the last dot of "..." *is* an ending
            return False
        if previous.isdigit() and following.isdigit():
            return True  # 3.5, 1.000
        word = self._word_before(text, index)
        if word in ABBREVIATIONS:
            return True
        return len(word) == 1 and word.isalpha()  # initials: "B. A."

    @staticmethod
    def _word_before(text: str, index: int) -> str:
        end = index
        start = end
        while start > 0 and (text[start - 1].isalnum() or text[start - 1] in ".-'"):
            start -= 1
        return text[start:end].strip(".-'").lower()

    def _soft_cut(self, text: str) -> int | None:
        """Cut a too-long sentence at the best break before `max_chars`."""
        window = text[: self.max_chars]
        for index in range(len(window) - 1, 0, -1):
            if window[index] in SOFT_BREAKS:
                return index + 1
        if (space := window.rfind(" ")) > 0:
            return space + 1
        return None


@dataclass(slots=True)
class SpokenSentence:
    """One sentence, its audio, and what producing it cost."""

    text: str
    pcm: bytes = b""
    sample_rate: int = 16000
    index: int = 0
    prosody: Prosody = field(default_factory=Prosody)
    engine: str = ""
    cached: bool = False
    synth_ms: float = 0.0
    played_ms: float = 0.0

    @property
    def audible(self) -> bool:
        return bool(self.pcm)

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "chars": len(self.text),
            "engine": self.engine,
            "cached": self.cached,
            "synth_ms": round(self.synth_ms, 1),
            "audio_s": round(len(self.pcm) / 2 / max(1, self.sample_rate), 3),
            "prosody": self.prosody.as_dict(),
        }


class InterruptPolicy:
    """How "stop" reaches the mouth.

    Three routes, one flag: the stop hotkey (always), a stop word heard *while*
    the mic is open between sentences, and the long-answer cap.  Voice barge-in
    over Atlas's own speech needs echo cancellation that this laptop does not
    have, so it is opt-in (`[tts] barge_in`) and off by default — an assistant
    that stops mid-word because the television said "stop" is worse than one you
    interrupt with a key.
    """

    def __init__(self, words: Iterable[str] = (), *, enabled: bool = True) -> None:
        self.words = tuple(word.strip().lower() for word in words if word.strip())
        self.enabled = enabled
        self.triggered = False
        self.reason = ""

    def stop(self, reason: str = "hotkey") -> None:
        self.triggered = True
        self.reason = reason

    def reset(self) -> None:
        self.triggered = False
        self.reason = ""

    def hears_stop(self, text: str) -> bool:
        """Is this utterance a stop word?  Whole-word match, not substring."""
        if not self.enabled or not self.words:
            return False
        words = {word.strip(".,!?؛،").lower() for word in text.split()}
        return any(word in words for word in self.words)


class SentenceStreamer:
    """Token stream in, audible sentences out, synthesised ahead of playback."""

    def __init__(
        self,
        engines: VoiceSynthesizer | Iterable[VoiceSynthesizer],
        *,
        cache: TtsCache | None = None,
        prosody: ProsodyDirector | None = None,
        splitter: SentenceSplitter | None = None,
        interrupt: InterruptPolicy | None = None,
        max_chars: int = 220,
        max_sentences: int = 3,
        ask_to_continue: bool = True,
        continuation_question: str = CONTINUE_QUESTION,
        prefetch: int = 2,
        sample_rate: int = 16000,
    ) -> None:
        self.engines: list[VoiceSynthesizer] = (
            [engines] if isinstance(engines, VoiceSynthesizer) else list(engines)
        )
        if not self.engines:
            raise ValueError("SentenceStreamer needs at least one voice")
        self.cache = cache
        self.prosody = prosody or ProsodyDirector()
        self.splitter = splitter or SentenceSplitter(max_chars=max_chars)
        self.interrupt = interrupt or InterruptPolicy()
        self.max_sentences = max_sentences
        self.ask_to_continue = ask_to_continue
        self.continuation_question = continuation_question
        self.prefetch = max(1, prefetch)
        self.sample_rate = sample_rate

        # ── per-stream state, reset in `stream()` ──
        self.mood: Any = None
        self.voice = ""
        self.language: LanguageTag = "ar-MA"
        self.engine_index = 0
        self.engine_switches = 0
        self.spoken: list[SpokenSentence] = []
        self.truncated = False
        self.stop_reason = ""
        self.first_audio_ms = 0.0
        self.degraded = False

    # ── public API ───────────────────────────────────────────────────
    async def stream(
        self,
        tokens: AsyncIterator[str],
        *,
        voice: str = "",
        language: LanguageTag = "ar-MA",
        mood: Any = None,
        cap: bool = True,
    ) -> AsyncGenerator[SpokenSentence, None]:
        """Speak a token stream, starting as soon as the first sentence exists."""
        self._reset()
        self.mood = mood
        self.voice = voice
        self.language = language
        self._sentences: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self.prefetch)
        self._audio: asyncio.Queue[SpokenSentence | None] = asyncio.Queue(maxsize=self.prefetch)
        started = time.monotonic()

        filler = asyncio.create_task(self._split_tokens(tokens, cap=cap))
        speaker = asyncio.create_task(self._synthesize_all())
        try:
            while True:
                item = await self._audio.get()
                if item is None:
                    break
                if self.interrupt.triggered:
                    self.stop_reason = self.stop_reason or self.interrupt.reason or "interrupted"
                    break
                if not self.first_audio_ms:
                    self.first_audio_ms = (time.monotonic() - started) * 1000
                self.spoken.append(item)
                yield item
        finally:
            for task in (filler, speaker):
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    async def say(
        self, text: str, *, voice: str = "", language: LanguageTag = "ar-MA", mood: Any = None
    ) -> list[SpokenSentence]:
        """One-shot: speak a whole string (used by `atlas say`, chimes and L5)."""

        async def one_shot() -> AsyncIterator[str]:
            yield text

        return [sentence async for sentence in self.stream(one_shot(), voice=voice, language=language, mood=mood, cap=False)]

    def key_for(self, text: str, prosody: Prosody) -> str:
        """The cache key for *this* sentence as it will be spoken.

        Prosody is part of it because a cheerful sentence is different audio,
        and the voice/engine are part of it because two engines are two voices.
        One definition, used by synthesis and by cache warming alike.
        """
        return cache_key(
            text,
            voice=self.voice,
            language=str(self.language),
            engine=self.engine.name,
            rate=prosody.rate,
        )

    def stop(self, reason: str = "stop") -> None:
        self.interrupt.stop(reason)

    @property
    def engine(self) -> VoiceSynthesizer:
        """The engine currently in charge — clamped, never an IndexError.

        `engine_index` legitimately walks off the end when every voice has failed,
        and a stats call must not be the thing that crashes a conversation.
        """
        return self.engines[min(self.engine_index, len(self.engines) - 1)]

    def stats(self) -> dict[str, object]:
        return {
            "sentences": len(self.spoken),
            "truncated": self.truncated,
            "stop_reason": self.stop_reason,
            "engine": "none" if self.degraded else self.engine.name,
            "engine_switches": self.engine_switches,
            "degraded": self.degraded,
            "first_audio_ms": round(self.first_audio_ms, 1),
            "cache_hits": sum(1 for s in self.spoken if s.cached),
            "audio_s": round(sum(len(s.pcm) / 2 / max(1, s.sample_rate) for s in self.spoken), 2),
        }

    # ── pipeline ─────────────────────────────────────────────────────
    def _reset(self) -> None:
        self.spoken = []
        self.truncated = False
        self.stop_reason = ""
        self.first_audio_ms = 0.0
        self.degraded = False
        self.engine_index = 0
        self.engine_switches = 0
        self.splitter.reset()
        self.interrupt.reset()

    async def _split_tokens(self, tokens: AsyncIterator[str], *, cap: bool) -> None:
        """Stage 1: tokens → complete sentences (bounded queue)."""
        count = 0
        limit = self.max_sentences if cap else 0
        try:
            async for delta in tokens:
                if not delta:
                    continue
                for sentence in self.splitter.feed(delta):
                    if limit and count >= limit:
                        self.truncated = True
                        break
                    await self._sentences.put(sentence)
                    count += 1
                if self.truncated or self.interrupt.triggered:
                    break
            tail = self.splitter.flush() if not (self.truncated or self.interrupt.triggered) else ""
            if tail and not (limit and count >= limit):
                await self._sentences.put(tail)
            elif tail:
                self.truncated = True
            if self.truncated and self.ask_to_continue:
                self.splitter.reset()
                await self._sentences.put(self.continuation_question)
        finally:
            # The sentinel always goes in, even on cancellation, or the speaker
            # task would wait forever on an empty queue.
            with suppress(asyncio.CancelledError):
                await self._sentences.put(None)

    async def _synthesize_all(self) -> None:
        """Stage 2: sentences → audio (in a worker thread; CPU-bound by nature)."""
        index = 0
        try:
            while (text := await self._sentences.get()) is not None:
                prosody = self.prosody.plan(text, mood=self.mood)
                started = time.monotonic()
                pcm, rate, cached, engine_name = await asyncio.to_thread(self._render, text, prosody)
                spoken = SpokenSentence(
                    text=text,
                    pcm=pcm,
                    sample_rate=rate,
                    index=index,
                    prosody=prosody,
                    engine=engine_name,
                    cached=cached,
                    synth_ms=(time.monotonic() - started) * 1000,
                )
                index += 1
                await self._audio.put(spoken)
        finally:
            with suppress(asyncio.CancelledError):
                await self._audio.put(None)

    # ── synthesis, with the fallback chain ───────────────────────────
    def _render(self, text: str, prosody: Prosody) -> tuple[bytes, int, bool, str]:
        """Synthesise one sentence, falling forward through the engine chain.

        A dead engine mid-answer must not shorten the answer (the plan's rule):
        the *same sentence* is retried on the next voice, and only when every
        voice has failed is the sentence returned empty — captions, no audio.
        """
        key = ""
        if self.cache is not None:
            key = self.key_for(text, prosody)
            if (clip := self.cache.get(key)) is not None:
                return clip.pcm, clip.sample_rate, True, self.engine.name

        while self.engine_index < len(self.engines):
            engine = self.engine
            try:
                pcm, rate = _collect(
                    engine, text, voice=self.voice, language=self.language, prosody=prosody
                )
            except Exception as exc:
                log.warning("tts_engine_failed engine=%s error=%s", engine.name, exc)
                self.engine_switches += 1
                self.engine_index += 1
                continue
            if not pcm:
                log.warning("tts_empty_audio engine=%s", engine.name)
                self.engine_switches += 1
                self.engine_index += 1
                continue
            if self.cache is not None and key:
                self.cache.put(key, pcm, sample_rate=rate, engine=engine.name, voice=self.voice, text=text)
            return pcm, rate, False, engine.name

        # Every voice is gone: the words still reach the user, as text.
        self.degraded = True
        log.error("tts_all_engines_failed text=%r", text[:60])
        return b"", self.sample_rate, False, "none"


def _collect(
    engine: VoiceSynthesizer,
    text: str,
    *,
    voice: str,
    language: LanguageTag,
    prosody: Prosody,
) -> tuple[bytes, int]:
    """Run one synthesis to completion; the one place chunks become bytes."""
    pcm = bytearray()
    rate = 16000
    for chunk in engine.speak(text, voice=voice, language=language, prosody=prosody):
        if chunk.final and not chunk.pcm:
            rate = chunk.sample_rate or rate
            continue
        if chunk.pcm:
            pcm.extend(chunk.pcm)
            rate = chunk.sample_rate or rate
    return bytes(pcm), rate


def _as_tag(value: Any) -> LanguageTag:
    """Coerce config text to the `LanguageTag` literal — one place, one rule."""
    text = str(value or "ar-MA")
    return "en-GB" if text.startswith("en") else ("unknown" if text == "unknown" else "ar-MA")


@dataclass(slots=True)
class MouthConfig:
    """Everything `[tts]` says about *how* Atlas speaks, in one object.

    Duck-typed off the config like the rest of the audio package: no import of
    `atlas-core.config` here, so a test can hand this a dict-ish stand-in.
    """

    max_sentences: int = 3
    ask_to_continue: bool = True
    prosody: bool = True
    quiet_hours: tuple[str, str] = ("22:30", "07:00")
    gain: float = 1.0
    barge_in: bool = False
    stop_words: tuple[str, ...] = ("stop", "safi", "skut")
    prefetch: int = 2
    cache: bool = True
    cache_path: str = "data/tts-cache.sqlite3"
    cache_max_mb: float = 300.0
    continuation_question: str = CONTINUE_QUESTION

    @classmethod
    def from_config(cls, config: Any = None) -> MouthConfig:
        tts = getattr(config, "tts", None)
        if tts is None:
            return cls()
        hours = list(getattr(tts, "quiet_hours", []) or [])
        words = tuple(str(word) for word in getattr(tts, "stop_words", []) or [])
        return cls(
            max_sentences=int(getattr(tts, "max_sentences", 3) or 0),
            ask_to_continue=bool(getattr(tts, "ask_to_continue", True)),
            prosody=bool(getattr(tts, "prosody", True)),
            quiet_hours=(hours[0], hours[1]) if len(hours) >= 2 else ("22:30", "07:00"),
            gain=float(getattr(tts, "gain", 1.0) or 1.0),
            barge_in=bool(getattr(tts, "barge_in", False)),
            stop_words=words or ("stop", "safi", "skut"),
            prefetch=max(1, int(getattr(tts, "prefetch", 2) or 2)),
            cache=bool(getattr(tts, "cache_path", "")),
            cache_path=str(
                getattr(tts, "cache_path", "data/tts-cache.sqlite3") or "data/tts-cache.sqlite3"
            ),
            cache_max_mb=float(getattr(tts, "cache_max_mb", 300) or 300),
        )


class Mouth:
    """Atlas's voice: the pipeline, the player and the microphone gate together.

    The CLI talks to this and nothing else, which is why the L2 loop did not have
    to change to gain a voice — it still hands over a token stream and gets text
    back.
    """

    def __init__(
        self,
        engines: VoiceSynthesizer | Iterable[VoiceSynthesizer] | None = None,
        *,
        player: AudioPlayer | None = None,
        cache: TtsCache | None = None,
        prosody: ProsodyDirector | None = None,
        interrupt: InterruptPolicy | None = None,
        max_sentences: int = 3,
        ask_to_continue: bool = True,
        language: LanguageTag = "ar-MA",
        voice: str = "",
        engine: str | None = None,
        config: Any = None,
    ) -> None:
        if engines is None:
            engines = build_voice_chain(config, language=language, engine=engine)
        self.language = language
        self.voice = voice
        self.player = player or AudioPlayer()
        self.interrupt = interrupt or InterruptPolicy()
        self.streamer = SentenceStreamer(
            engines,
            cache=cache,
            prosody=prosody,
            interrupt=self.interrupt,
            max_sentences=max_sentences,
            ask_to_continue=ask_to_continue,
        )
        self.spoken_text: list[str] = []
        self.audio_s = 0.0

    @classmethod
    def from_config(
        cls,
        config: Any = None,
        *,
        player: AudioPlayer | None = None,
        gate: MicGate | None = None,
        cache: TtsCache | None = None,
        language: str | None = None,
        voice: str = "",
        engine: str | None = None,
        playing: bool = True,
    ) -> Mouth:
        """Build the whole mouth from `[tts]`, with every piece replaceable.

        `playing=False` swaps in a `NullWriter` — synthesis without a sound card,
        which is what `--out file.wav`, the benchmark and CI all use.
        """
        from atlas_audio.playback import NullWriter

        settings = MouthConfig.from_config(config)
        if cache is None and settings.cache:
            cache = TtsCache(settings.cache_path, max_mb=settings.cache_max_mb)
        if player is None:
            writer = SoundDeviceWriter() if playing else NullWriter()
            player = AudioPlayer(
                writer,
                gain=settings.gain,
                gate=gate,
                on_level=None,
            )
        return cls(
            None,
            player=player,
            cache=cache,
            prosody=ProsodyDirector(enabled=settings.prosody, quiet_hours=settings.quiet_hours),
            # Stop *words* are recognised only when barge-in is on; the stop
            # hotkey works regardless (it calls `stop()` directly).
            interrupt=InterruptPolicy(settings.stop_words, enabled=settings.barge_in),
            max_sentences=settings.max_sentences,
            ask_to_continue=settings.ask_to_continue,
            language=_as_tag(language or getattr(getattr(config, "app", None), "language", "ar-MA")),
            voice=voice,
            engine=engine,
            config=config,
        )

    @property
    def cache(self) -> TtsCache | None:
        return self.streamer.cache

    @property
    def engine_name(self) -> str:
        return self.streamer.engine.name

    async def speak(
        self,
        tokens: AsyncIterator[str],
        *,
        language: LanguageTag | None = None,
        mood: Any = None,
        cap: bool = True,
    ) -> AsyncIterator[str]:
        """Speak a token stream and yield each sentence's text as it is played."""
        player = self.player
        player.reset()
        stream = self.streamer.stream(
            tokens, voice=self.voice, language=language or self.language, mood=mood, cap=cap
        )
        try:
            async for spoken in stream:
                if spoken.audible:
                    result = await asyncio.to_thread(
                        player.play, spoken.pcm, sample_rate=spoken.sample_rate, gain=spoken.prosody.energy
                    )
                    spoken.played_ms = result.seconds * 1000
                    self.audio_s += result.seconds
                else:
                    log.warning("speaking_text_only sentence=%s", spoken.index)
                self.spoken_text.append(spoken.text)
                yield spoken.text
        finally:
            with suppress(Exception):
                await stream.aclose()

    async def say(
        self, text: str, *, language: LanguageTag | None = None, mood: Any = None, play: bool = True
    ) -> list[SpokenSentence]:
        """Speak one whole string — greetings, chimes, error notices.

        `play=False` synthesises and returns the audio instead of writing it to
        the device: that is what `atlas say --out file.wav` and the benchmark use.
        """
        if not play:
            return await self.streamer.say(
                text, voice=self.voice, language=language or self.language, mood=mood
            )

        async def one_shot() -> AsyncIterator[str]:
            yield text

        async for _sentence in self.speak(one_shot(), language=language, mood=mood, cap=False):
            pass
        return list(self.streamer.spoken)

    async def warm(
        self, lines: Iterable[str] | None = None, *, language: LanguageTag | None = None
    ) -> int:
        """Pre-synthesise the standard lines into the cache.  Returns how many.

        Called at boot (L5) and by `atlas voice warm`.  It synthesises *without*
        playing, so it is safe to run while the user is talking to someone else —
        and it goes through the same streamer, so what lands in the cache is
        exactly what a real utterance would have produced.

        Async, like the rest of the pipeline: the caller decides where the thread
        goes.  The CLI wraps it in `asyncio.run`; L5 warms on a low-priority
        background thread so a cold start never blocks the first word.
        """
        language = language or self.language
        if self.streamer.cache is None:
            return 0
        wanted = tuple(lines) if lines is not None else CANNED_LINES.get(str(language), ())
        # The prosody a line will be spoken with decides its key, so ask the
        # director rather than assuming 1.0 — "bghiti nkemmel?" is a question.
        plans = {line: self.streamer.prosody.plan(line) for line in wanted}
        keys = {line: self.streamer.key_for(line, plan) for line, plan in plans.items()}
        missing = set(self.streamer.cache.missing(keys.values()))
        warmed = 0
        for line in wanted:
            if keys[line] not in missing:
                continue
            spoken = await self.streamer.say(line, voice=self.voice, language=language)
            warmed += 1 if any(item.audible for item in spoken) else 0
        log.info("tts_cache_warmed lines=%s of %s", warmed, len(wanted))
        return warmed

    def stop(self, reason: str = "stop") -> None:
        self.streamer.stop(reason)
        self.player.stop()

    def close(self) -> None:
        self.player.close()

    def stats(self) -> dict[str, object]:
        return {
            **self.streamer.stats(),
            "player": self.player.as_dict(),
            "gate_muted": self.player.gate.muted,
        }


__all__ = [
    "ABBREVIATIONS",
    "CANNED_LINES",
    "CONTINUE_QUESTION",
    "InterruptPolicy",
    "Mouth",
    "SentenceSplitter",
    "SentenceStreamer",
    "SpokenSentence",
]
