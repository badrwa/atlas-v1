"""The interaction loop: wake → listen → transcribe → reply → listen again.

This is where L2 stops being a pile of engines and becomes the thing the plan
describes: *"Atlas… chno ljaw?"* — one wake word, then a conversation that keeps
flowing for as long as you keep talking.

Four rules shape the whole file:

1. **Half duplex, always.** The microphone is muted while Atlas speaks. Open mic
   during TTS is the classic toy-assistant bug: it hears itself, transcribes
   itself, and answers itself.
2. **A follow-up window, not a new wake word.** After a reply, `followup_ms`
   (2.5 s) of grace where the next utterance starts on speech alone. Without it,
   every second sentence costs you "atlas". When the window expires the loop
   re-arms — otherwise the television gets a turn.
3. **A guard after the wake word.** KWS fires at the end of "atlas", but a few
   frames of the keyword itself (and of the chime) are still arriving. Those are
   dropped, so the command never starts with "-las".
4. **Identity is one gate per utterance, not a per-word guess.**  When a verifier
   and a guard are configured, every utterance is embedded once, compared to the
   enrolled profiles, and turned into a `Permissions` set *before* the reply is
   generated.  The loop does not enforce capability itself — it records the
   verdict on the `Turn` and exposes it as `loop.permissions`, so the caller (the
   CLI, the tool loop, the vault writer) gates on one answer from one place.
5. **The FSM does not own I/O.** `LoopState` says what is happening — the orb
   shows it — while this class performs the blocking work.  That separation is
   why the whole loop can be driven by a WAV file, a frame at a time, in tests.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from atlas_audio.capture import TurnRecorder
from atlas_audio.frames import FRAME_MS, Frame, FrameBus, FramePacket, rms
from atlas_audio.postprocess import AsrPostProcessor, detect_language
from atlas_audio.vad import Utterance, VadSegmenter
from atlas_audio.wake import WakeLog
from atlas_core.contracts import Detection, SpeechRecognizer, Transcript, WakeWordEngine
from atlas_core.events import AudioLevel, EventBus, SpeakerMatched, StateChanged
from atlas_core.identity import Permissions

log = logging.getLogger(__name__)


class LoopState(StrEnum):
    """What the loop is doing — the same vocabulary the orb shows.

    `StrEnum` rather than `(str, Enum)`: same JSON behaviour, without the
    multiple-inheritance footgun that ruff's UP042 flags.
    """

    IDLE = "idle"  # stopped, or never started
    WAKING = "waking"  # armed, waiting for "atlas"
    LISTENING = "listening"  # keyword heard, capturing the command
    THINKING = "thinking"  # transcribing + waiting for the brain
    SPEAKING = "speaking"  # microphone muted
    FOLLOWUP = "followup"  # post-reply grace window, no keyword needed
    STOPPED = "stopped"


@dataclass
class LoopConfig:
    """Timings straight from LEVEL-02: 2.5 s of grace, 160 ms of guard."""

    followup_ms: int = 2500
    wake_guard_ms: int = 160
    silence_ms: int = 500
    chime: bool = True

    @classmethod
    def from_config(
        cls,
        config: Any = None,
        *,
        followup_ms: int | None = None,
        snappy: bool = False,
    ) -> LoopConfig:
        """Read `[dialogue]`/`[vad]`; explicit arguments win (CLI flags).

        `snappy` is the only thing the CLI overrides often enough to earn a
        parameter of its own — it is the difference between "fast" and "polite".
        """
        dialogue = getattr(config, "dialogue", None)
        vad = getattr(config, "vad", None)
        return cls(
            followup_ms=_int(followup_ms or getattr(dialogue, "followup_ms", None), 2500),
            wake_guard_ms=_int(getattr(dialogue, "wake_guard_ms", None), 160),
            silence_ms=350 if snappy else _int(getattr(vad, "silence_ms", None), 500),
            chime=bool(getattr(dialogue, "chime", True)),
        )

    @property
    def guard_frames(self) -> int:
        return max(0, self.wake_guard_ms // FRAME_MS)

    @property
    def followup_frames(self) -> int:
        return max(1, self.followup_ms // FRAME_MS)


@dataclass
class Turn:
    """One complete exchange, as the rest of Atlas sees it."""

    index: int
    transcript: Transcript
    text: str = ""
    language: str = "unknown"
    wake: Detection | None = None
    utterance_ms: float = 0.0
    recorded: bool = False
    reply: str = ""
    #: Who said it (L4).  An empty name with `owner=True` is the single-user case.
    speaker: str = ""
    speaker_score: float = 0.0
    owner: bool = False
    restricted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "language": self.language,
            "confidence": round(self.transcript.confidence, 3),
            "engine": self.transcript.engine,
            "asr_ms": round(self.transcript.duration_ms, 1),
            "utterance_ms": round(self.utterance_ms, 1),
            "wake_score": round(self.wake.score, 3) if self.wake else 0.0,
            "speaker": self.speaker,
            "speaker_score": round(self.speaker_score, 3),
            "owner": self.owner,
            "restricted": self.restricted,
        }


def _int(value: Any, default: int) -> int:
    """Config values arrive as `Any` (a TOML section, or a dict in tests).

    This is the single place they become ints: one coercion function beats six
    `int(getattr(...))` calls that each guess a different default.
    """
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        log.warning("config_value_invalid value=%r — using %s", value, default)
        return default


class VoiceLoop:
    """Wake → listen → transcribe, driven by frames from any source.

    The reply itself is injected (`respond`): L1's `ChatSession` in production, a
    canned line in tests.  That keeps this class about *ears*, which is the level
    it belongs to — and it keeps the L2 tests offline, fast and deterministic.
    """

    def __init__(
        self,
        *,
        wake: WakeWordEngine,
        segmenter: VadSegmenter,
        recognizer: SpeechRecognizer | Any,
        respond: Callable[[str, str], AsyncIterator[str]] | None = None,
        say: Callable[[str], None] | None = None,
        processor: AsrPostProcessor | None = None,
        bus: FrameBus | None = None,
        recorder: TurnRecorder | None = None,
        wake_log: WakeLog | None = None,
        config: LoopConfig | None = None,
        verifier: Any = None,
        guard: Any = None,
        profiles: dict[str, Any] | None = None,
        speaker_log: Any = None,
        events: EventBus | None = None,
    ) -> None:
        self.wake = wake
        self.segmenter = segmenter
        self.recognizer = recognizer
        self.respond = respond
        self.say = say
        self.processor = processor or AsrPostProcessor()
        self.bus = bus
        self.recorder = recorder
        self.wake_log = wake_log
        self.config = config or LoopConfig()
        # L4 is optional on purpose: a loop with no verifier is the L2/L3 loop,
        # and every existing test keeps working without one.
        self.verifier = verifier
        self.guard = guard
        self.profiles = profiles or {}
        self.speaker_log = speaker_log
        #: The verdict for the utterance being handled.  The caller reads this to
        #: decide what memory and which tools the reply may see.
        self.permissions: Permissions = Permissions.owner_of("", reason="identity_off")
        self.identity_ms = 0.0
        #: The orb listens here (L5).  The loop publishes its *own* vocabulary;
        #: turning that into motion is the UI's job, not the ears'.
        self.events = events

        #: Annotated here, always written through `_set_state`, so no assignment
        #: can skip the notification.
        self.state: LoopState = LoopState.IDLE
        self.turns: list[Turn] = []
        self.mic_muted = False
        self._mute_depth = 0
        self.frames_seen = 0
        self.wake_hits = 0
        self.frames_dropped_after_wake = 0

        self._speaking = False
        self._guard = 0
        self._last_wake: Detection | None = None
        self._followup_deadline = 0

    # ── arming, half duplex ──────────────────────────────────────────
    def arm(self) -> None:
        """Start waiting for the wake word (the loop is not running until then)."""
        self._set_state(LoopState.WAKING)
        self._guard = 0
        self.segmenter.reset()

    def stop(self) -> None:
        self._set_state(LoopState.STOPPED)
        self.segmenter.reset()

    @property
    def listening(self) -> bool:
        return not self.mic_muted and self.state in {
            LoopState.WAKING,
            LoopState.LISTENING,
            LoopState.FOLLOWUP,
        }

    def mute(self) -> None:
        """Half duplex: no frames are acted on while Atlas is speaking.

        Reference counted, because two things now close the microphone at
        overlapping times — the reply itself, and the mouth's `MicGate` around
        playback.  A plain boolean would let whichever finished first reopen the
        mic while the other was still making noise, which is the exact bug that
        makes an assistant answer itself.
        """
        self._mute_depth += 1
        self.mic_muted = self._mute_depth > 0

    def unmute(self) -> None:
        """Release one hold; the mic opens when the last one is gone."""
        self._mute_depth = max(0, self._mute_depth - 1)
        self.mic_muted = self._mute_depth > 0

    # ── one frame at a time ──────────────────────────────────────────
    async def feed(self, packet: FramePacket) -> Turn | None:
        """The whole loop, one frame in. Returns a `Turn` when one completes.

        Frames are still *counted* while muted — the microphone is not physically
        off, we simply refuse to act on them, which is what stops Atlas from
        transcribing itself mid-sentence.
        """
        self.frames_seen += 1
        if self.recorder:
            self.recorder.frames_seen += 1
            # Recorded unconditionally, before the mute check: a capture dump is
            # what the microphone heard, not what Atlas decided to act on.
            self.recorder.push_frame(packet.frame)

        if self.state in {LoopState.IDLE, LoopState.STOPPED, LoopState.THINKING}:
            return None
        if self.mic_muted or self._speaking:
            return None

        if self._guard > 0:  # the tail of the keyword, and the chime
            self._guard -= 1
            self.frames_dropped_after_wake += 1
            return None

        if self.state == LoopState.WAKING:
            return await self._handle_wake(packet)

        if self.state == LoopState.FOLLOWUP and self.frames_seen > self._followup_deadline:
            log.info("followup_expired frames=%s", self.frames_seen)
            self.arm()  # back to needing "atlas"
            return None

        return await self._handle_listening(packet)

    async def _handle_wake(self, packet: FramePacket) -> Turn | None:
        detection = self.wake.feed(packet.frame)
        if not detection.hit:
            return None
        self.wake_hits += 1
        self._last_wake = detection
        log.info("wake keyword=%s score=%.2f", detection.keyword, detection.score)
        self._set_state(LoopState.LISTENING)
        self._guard = self.config.guard_frames
        # The keyword's own audio must not become the first word of the command.
        self.segmenter.reset()
        return None

    async def _handle_listening(self, packet: FramePacket) -> Turn | None:
        self._emit_level(packet.frame)
        utterance = self.segmenter.feed(packet)
        if utterance is None:
            return None
        return await self._transcribe(utterance, wake=self._last_wake)

    async def _transcribe(self, utterance: Utterance, *, wake: Detection | None = None) -> Turn | None:
        self._set_state(LoopState.THINKING)
        try:
            transcript = await self.recognizer.transcribe(utterance.pcm, language="unknown")
        except Exception as exc:
            log.warning("asr_failed error=%s", exc)
            self._set_state(LoopState.WAKING)
            return None

        text = self.processor.process(transcript.text, confidence=transcript.confidence)
        language = transcript.language
        if language == "unknown":
            language = detect_language(text)
        if not text.strip():
            log.info("empty_transcript engine=%s", transcript.engine)
            self._set_state(LoopState.WAKING if wake is None else LoopState.FOLLOWUP)
            return None

        permissions = await self._identify(utterance)
        turn = Turn(
            index=len(self.turns),
            transcript=transcript,
            text=text,
            language=str(language),
            wake=wake,
            utterance_ms=utterance.duration_ms,
            speaker=permissions.speaker,
            speaker_score=permissions.score,
            owner=permissions.owner,
            restricted=permissions.restricted,
        )
        self.turns.append(turn)
        if self.recorder:
            self.recorder.add_turn(
                utterance.pcm,
                transcript=text,
                language=str(language),
                confidence=transcript.confidence,
                engine=transcript.engine,
                reason=utterance.reason,
            )
            turn.recorded = True

        self.permissions = permissions
        if permissions.speaker or permissions.restricted:
            self._emit(
                SpeakerMatched(
                    name=permissions.speaker,
                    score=permissions.score,
                    owner=permissions.owner,
                )
            )
        if self.respond is not None:
            await self._reply(turn)
        else:
            self._enter_followup()
        return turn

    # ── state, and telling the orb about it ──────────────────────────
    def _set_state(self, state: LoopState) -> None:
        """One writer for `self.state`, so the UI cannot miss a transition."""
        previous, self.state = self.state, state
        if previous != state:
            self._emit(StateChanged(state=state.value, previous=previous.value))

    def _emit(self, event: Any) -> None:
        """Fire-and-forget publish: the loop never waits on a watcher.

        `EventBus.emit` already owns that policy (a task when a loop is running,
        a direct publish when there is none); a second implementation here would
        be a second thing to get wrong, and a slow subscriber must never add
        latency to a turn.
        """
        if self.events is not None:
            self.events.emit(event)

    def _emit_level(self, frame: Any) -> None:
        """Microphone loudness, for the orb's listening rings."""
        if self.events is not None:
            self._emit(AudioLevel(level=rms(frame)))

    async def _identify(self, utterance: Utterance) -> Permissions:
        """Who is speaking?  One embedding, one comparison, one verdict.

        Three outcomes, and the log names them:

        * no guard configured → the pre-L4 world: owner capabilities, `no_guard`;
        * a verifier that cannot run (no model, no microphone) → single user again,
          *logged* as `verifier_unavailable`, because Atlas must never let a broken
          gate pass silently for an open one;
        * everything working → the score decides, and a stranger gets general
          conversation and nothing else.
        """
        if self.guard is None:
            return Permissions.owner_of("", reason="no_guard")

        verifier = self.verifier
        available = bool(
            verifier is not None and getattr(verifier, "available", lambda: False)()
        )
        if not available:
            return self.guard.unavailable(
                reason="verifier_unavailable", utterance_ms=utterance.duration_ms
            )

        started = time.perf_counter()
        embedding = await asyncio.to_thread(verifier.embed_or_none, utterance.pcm)
        elapsed = (time.perf_counter() - started) * 1000
        self.identity_ms = elapsed
        if embedding is None:
            # Too short to identify: not a rejection, but not a licence either.
            return self.guard.unavailable(
                reason="embed_too_short", utterance_ms=utterance.duration_ms
            )
        return self.guard.verify(
            embedding, utterance_ms=utterance.duration_ms, elapsed_ms=elapsed
        )

    async def _reply(self, turn: Turn) -> None:
        """Speak, with the mic muted for the whole of it (half duplex)."""
        assert self.respond is not None
        self._speaking = True
        self.mute()
        self._set_state(LoopState.SPEAKING)
        chunks: list[str] = []
        try:
            async for delta in self.respond(turn.text, turn.language):
                chunks.append(delta)
                if self.say:
                    self.say(delta)
        except Exception as exc:
            log.warning("reply_failed error=%s", exc)
        finally:
            self._speaking = False
            self.unmute()
        turn.reply = "".join(chunks)
        if not turn.reply:
            log.warning("reply_empty turn=%s", turn.index)
        self._enter_followup()
        log.info(
            "turn_complete index=%s chars=%s asr_ms=%.0f utterance_ms=%.0f",
            turn.index,
            len(turn.reply),
            turn.transcript.duration_ms,
            turn.utterance_ms,
        )

    def _enter_followup(self) -> None:
        self._set_state(LoopState.FOLLOWUP)
        self._followup_deadline = self.frames_seen + self.config.followup_frames
        self.segmenter.reset()

    # ── replay and push-to-talk ──────────────────────────────────────
    async def feed_frames(self, frames: Iterable[Frame], *, source: str = "file") -> list[Turn]:
        """Drive the loop from a list of frames — no bus, no threads, no timing."""
        found: list[Turn] = []
        for index, frame in enumerate(frames):
            packet = FramePacket(frame=frame, index=index, at_ms=index * FRAME_MS, source=source)
            if turn := await self.feed(packet):
                found.append(turn)
        # A recording that ends mid-sentence still has something to say.
        tail = self.segmenter.flush() if self.state in {LoopState.LISTENING, LoopState.FOLLOWUP} else None
        if tail is not None and (turn := await self._transcribe(tail, wake=self._last_wake)):
            found.append(turn)
        return found

    async def push_to_talk(self, frames: Iterable[Frame]) -> Turn | None:
        """Skip the wake word: the session was started by a key, not a keyword."""
        self._set_state(LoopState.LISTENING)
        self._guard = 0
        self._last_wake = None
        self.segmenter.reset()
        turns = await self.feed_frames(frames, source="ptt")
        return turns[-1] if turns else None

    # ── reporting ────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "muted": self.mic_muted,
            "frames": self.frames_seen,
            "turns": len(self.turns),
            "wake_hits": self.wake_hits,
            "guard_dropped": self.frames_dropped_after_wake,
            "wake_engine": self.wake.name,
            "asr": getattr(self.recognizer, "last_choice", "") or getattr(self.recognizer, "name", ""),
            "speaker": self.permissions.speaker,
            "capabilities": sorted(str(capability) for capability in self.permissions.capabilities),
            "identity_ms": round(self.identity_ms, 1),
        }

    def summary(self) -> str:
        if not self.turns:
            return "no turns yet"
        return "\n".join(
            f"{turn.index + 1}. [{turn.language} {turn.transcript.confidence:.2f}] {turn.text}"
            for turn in self.turns
        )


__all__ = ["LoopConfig", "LoopState", "Turn", "VoiceLoop"]
