"""ChatSession — one text turn, streamed, timed and evented.

This is the L1 brain.  L2 wraps it with microphone input, L3 with a voice, L7
with hands; the shape of a turn never changes, which is the whole point of
building the brain before the ears.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from atlas_core.config import AppConfig
from atlas_core.contracts import (
    LanguageTag,
    LlmRequest,
    Message,
    Role,
    TextDelta,
)
from atlas_core.events import EventBus, ReplyFinished, TokenDelta
from atlas_core.timings import TimingRecord, TimingRecorder
from atlas_mind.context import ContextBuilder
from atlas_mind.envelope import (
    REPLY_SCHEMA,
    EnvelopeError,
    ReplyEnvelope,
    envelope_instructions,
    parse_reply,
    repair_prompt,
)
from atlas_mind.language import LanguageRouter
from atlas_mind.mood import MoodEngine, MoodView
from atlas_mind.persona import Persona
from atlas_mind.router import ProviderRouter

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    text: str = ""
    language: str = "ar-MA"
    provider: str = ""
    ttft_ms: float | None = None
    total_ms: float = 0.0
    context: dict[str, int] = field(default_factory=dict)
    emotion: str = ""
    mood: dict[str, object] = field(default_factory=dict)
    structured: bool = False
    repaired: bool = False
    followup: bool = False


def reply_envelope_from_text(raw: str, language: LanguageTag) -> ReplyEnvelope:
    """Salvage an envelope from messy output: strip the JSON, keep the words.

    Used when both the first answer and the repair failed validation — and as the
    wrap for plain-text turns, so callers never have to know which path ran.
    """
    from atlas_mind.envelope import _strip_to_json  # local: one definition, not two

    text = raw.strip()
    if text.startswith("{"):
        try:
            payload = json.loads(_strip_to_json(text))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            spoken = payload.get("reply") or payload.get("text") or ""
            if isinstance(spoken, str) and spoken.strip():
                text = spoken.strip()
        else:
            text = ""
    return ReplyEnvelope(reply=text, language=language)


@dataclass
class EnvelopeOutcome:
    """What came back from a structured turn, and whether to believe it."""

    envelope: ReplyEnvelope
    repaired: bool = False
    from_model: bool = False


class ChatSession:
    """Holds the conversation and turns one user utterance into one reply."""

    def __init__(
        self,
        config: AppConfig,
        router: ProviderRouter,
        *,
        persona: Persona | None = None,
        language_router: LanguageRouter | None = None,
        context_builder: ContextBuilder | None = None,
        timings: TimingRecorder | None = None,
        events: EventBus | None = None,
        mood: MoodEngine | None = None,
        structured: bool = False,
    ) -> None:
        self.config = config
        self.router = router
        self.persona = persona or Persona(config)
        self.mood = mood or MoodEngine()
        # Structured turns cost time-to-first-token, so they are opt-in: the
        # fast streaming path is what makes Atlas feel like a conversation.
        self.structured = structured
        self.language = language_router or LanguageRouter(
            default=config.app.language,  # type: ignore[arg-type]
            secondary=config.app.secondary_language,  # type: ignore[arg-type]
        )
        self.context = context_builder or ContextBuilder(
            budget_tokens=config.mind.context_token_budget,
            memory_budget=config.mind.memory_token_budget,
            config_temperature=config.mind.temperature,
            max_output_tokens=config.mind.max_output_tokens,
        )
        self.timings = timings
        self.events = events
        self.history: list[Message] = []
        self.turns = 0
        self.last: TurnResult | None = None

    # ── state ────────────────────────────────────────────────────────
    @property
    def current_language(self) -> LanguageTag:
        return self.language.current

    def reset(self) -> None:
        self.history.clear()
        self.turns = 0
        self.language.reset()
        self.mood.reset()

    # ── the turn ─────────────────────────────────────────────────────
    async def send(
        self,
        text: str,
        *,
        memory: str = "",
        mood: MoodView | None = None,
        structured: bool | None = None,
        audience: Any = None,
    ) -> TurnResult:
        """Run one turn; returns the full reply (see `stream` for token-by-token)."""
        result = TurnResult()
        async for _delta in self.stream(
            text,
            memory=memory,
            mood=mood,
            structured=structured,
            audience=audience,
            _result=result,
        ):
            pass  # `stream` owns accumulation; send() just drives it to completion
        return result

    async def stream(
        self,
        text: str,
        *,
        memory: str = "",
        mood: MoodView | None = None,
        structured: bool | None = None,
        audience: Any = None,
        _result: TurnResult | None = None,
    ) -> AsyncIterator[str]:
        """Yield reply deltas as they arrive (this is what TTS consumes in L3).

        `audience` is L4's identity verdict as the *brain* sees it (a name, and
        whether this is the owner).  It changes the prompt, never the data: the
        caller is still responsible for passing `memory=""` when the speaker may
        not read it — two independent guards, on purpose.
        """
        started = perf_counter()
        language = self.language.route(text)
        self.mood.observe_owner(text)
        result = _result if _result is not None else TurnResult()
        result.language = language
        use_envelope = self.structured if structured is None else structured
        result.structured = use_envelope

        system = self.persona.system_prompt(
            language=language,
            mood=mood or self.mood.view,
            memory=memory,
            audience=audience,
        )
        request: LlmRequest = self.context.build(
            system=system,
            user_text=text,
            history=self.history,
            json_schema=REPLY_SCHEMA if use_envelope else None,
            language=language,
            # Memory travels *inside the system prompt* (persona renders it and
            # `ContextBuilder` budgets the rest of the window around it).  Passing
            # it here as well would send it twice and double-charge the budget.
            memory="",
        )
        if use_envelope and request.system:
            # The envelope instructions live with the schema, not in persona.jinja:
            # the schema and the words describing it must change together.
            request = request.model_copy(
                update={"system": request.system + "\n\n" + envelope_instructions(language)}
            )
        result.context = dict(self.context.last_build)

        first_delta_at: float | None = None

        def note_first_delta() -> None:
            nonlocal first_delta_at
            if first_delta_at is None:
                first_delta_at = (perf_counter() - started) * 1000
                result.ttft_ms = first_delta_at

        try:
            if use_envelope:
                raw = await self._collect(request)
                outcome = await self._envelope_or_repair(request, raw, language)
                result.repaired = outcome.repaired
                result.text = outcome.envelope.reply
                result.language = outcome.envelope.language
                if outcome.from_model:
                    # Only trust metadata the model actually produced: a salvaged
                    # reply has no right to claim an emotion it never reported.
                    result.emotion = outcome.envelope.emotion
                    result.followup = outcome.envelope.followup
                    self.mood.observe_envelope(outcome.envelope)
                else:
                    self.mood.observe_reply(result.text)
                note_first_delta()
                await self._publish(TokenDelta(text=outcome.envelope.reply))
                yield outcome.envelope.reply
            else:
                async for event in self.router.stream(request):
                    if isinstance(event, TextDelta) and event.text:
                        note_first_delta()
                        result.text += event.text
                        await self._publish(TokenDelta(text=event.text))
                        yield event.text
                self.mood.observe_reply(result.text)
        finally:
            result.total_ms = (perf_counter() - started) * 1000
            result.provider = self.router.last_provider
            result.mood = self.mood.view.as_dict()

        self._remember(text, result.text)
        self.turns += 1
        self.last = result
        self._record(result)
        await self._publish(
            ReplyFinished(text=result.text, language=language, provider=result.provider)
        )

    # ── structured turns ─────────────────────────────────────────────
    async def _collect(self, request: LlmRequest) -> str:
        """Buffer a structured reply — partial JSON is never worth showing."""
        chunks: list[str] = []
        async for event in self.router.stream(request):
            if isinstance(event, TextDelta) and event.text:
                chunks.append(event.text)
        return "".join(chunks)

    async def _envelope_or_repair(
        self, request: LlmRequest, raw: str, language: LanguageTag
    ) -> EnvelopeOutcome:
        """Parse, and on failure give the model exactly one chance to fix itself."""
        try:
            return EnvelopeOutcome(parse_reply(raw), repaired=False, from_model=True)
        except EnvelopeError as exc:
            problem = str(exc)
            log.info("envelope_invalid error=%s", problem)

        repair: LlmRequest = request.model_copy(
            update={
                "messages": [
                    *request.messages,
                    Message(role=Role.USER, content=repair_prompt(raw, problem)),
                ],
            }
        )
        try:
            second = await self._collect(repair)
            return EnvelopeOutcome(parse_reply(second), repaired=True, from_model=True)
        except EnvelopeError as exc2:
            # Losing metadata is fine; losing the answer is not. Speak the text.
            log.info("envelope_repair_failed error=%s", exc2)
            return EnvelopeOutcome(
                reply_envelope_from_text(raw, language), repaired=False, from_model=False
            )

    # ── bookkeeping ──────────────────────────────────────────────────
    def _remember(self, user_text: str, reply: str) -> None:
        self.history.append(Message(role=Role.USER, content=user_text))
        if reply:
            self.history.append(Message(role=Role.ASSISTANT, content=reply))
        # Keep the in-memory window small; the vault is the real long-term memory (L6).
        max_messages = self.context.max_turns * 2
        if len(self.history) > max_messages:
            del self.history[: len(self.history) - max_messages]

    def _record(self, result: TurnResult) -> None:
        if not self.timings:
            return
        self.timings.add(
            TimingRecord(
                turn=self.turns,
                ttft_ms=result.ttft_ms,
                total_ms=result.total_ms,
                provider=result.provider or "none",
                mode=self.config.app.profile,
                language=result.language,
                extra={"context": result.context},
            )
        )

    async def _publish(self, event) -> None:
        if self.events:
            await self.events.publish(event)

    def snapshot(self) -> dict[str, object]:
        return {
            "turns": self.turns,
            "language": self.language.current,
            "history": len(self.history),
            "providers": self.router.names,
            "last_provider": self.router.last_provider,
            "context": self.context.last_build,
        }


__all__ = ["ChatSession", "EnvelopeOutcome", "TurnResult", "reply_envelope_from_text"]
