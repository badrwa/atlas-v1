"""ChatSession — one text turn, streamed, timed and evented.

This is the L1 brain.  L2 wraps it with microphone input, L3 with a voice, L7
with hands; the shape of a turn never changes, which is the whole point of
building the brain before the ears.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from time import perf_counter

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
from atlas_mind.language import LanguageRouter
from atlas_mind.persona import MoodView, Persona
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
    ) -> None:
        self.config = config
        self.router = router
        self.persona = persona or Persona(config)
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

    # ── the turn ─────────────────────────────────────────────────────
    async def send(self, text: str, *, memory: str = "", mood: MoodView | None = None) -> TurnResult:
        """Run one turn; returns the full reply (see `stream` for token-by-token)."""
        result = TurnResult()
        async for _delta in self.stream(text, memory=memory, mood=mood, _result=result):
            pass  # `stream` owns accumulation; send() just drives it to completion
        return result

    async def stream(
        self,
        text: str,
        *,
        memory: str = "",
        mood: MoodView | None = None,
        _result: TurnResult | None = None,
    ) -> AsyncIterator[str]:
        """Yield reply deltas as they arrive (this is what TTS consumes in L3)."""
        started = perf_counter()
        language = self.language.route(text)
        result = _result if _result is not None else TurnResult()
        result.language = language

        system = self.persona.system_prompt(language=language, mood=mood, memory=memory)
        request: LlmRequest = self.context.build(
            system=system,
            user_text=text,
            history=self.history,
        )
        result.context = dict(self.context.last_build)

        first_delta_at: float | None = None
        try:
            async for event in self.router.stream(request):
                if isinstance(event, TextDelta) and event.text:
                    if first_delta_at is None:
                        first_delta_at = (perf_counter() - started) * 1000
                        result.ttft_ms = first_delta_at
                    result.text += event.text
                    await self._publish(TokenDelta(text=event.text))
                    yield event.text
        finally:
            result.total_ms = (perf_counter() - started) * 1000
            result.provider = self.router.last_provider

        self._remember(text, result.text)
        self.turns += 1
        self.last = result
        self._record(result)
        await self._publish(
            ReplyFinished(text=result.text, language=language, provider=result.provider)
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


__all__ = ["ChatSession", "TurnResult"]
