"""Gemini provider (native REST, no SDK).

Gemini is Atlas's *primary* brain for one specific reason: it is the strongest
available engine on Moroccan Darija — AtlasIA's own Darija pipeline picked it as
their transcription/annotation model — and its Flash/Flash-Lite free tier covers
a voice assistant's traffic many times over.

Wire format differences from the OpenAI world, all handled here:
    POST {base_url}/models/{model}:streamGenerateContent?alt=sse
    header `x-goog-api-key`
    body  {systemInstruction, contents[{role, parts[{text}]}], generationConfig}
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from atlas_core.contracts import (
    LlmEvent,
    LlmRequest,
    StreamEnd,
    TextDelta,
    ToolCallRequest,
    Usage,
)

from .http_base import HttpStreamingProvider

_ROLE_MAP = {"assistant": "model", "user": "user", "tool": "user"}


class GeminiProvider(HttpStreamingProvider):
    """Streaming Gemini client that speaks the same `LlmEvent` language as everyone else."""

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "x-goog-api-key": self.api_key,
            **self.config.extra_headers,
        }

    def _payload(self, request: LlmRequest) -> dict[str, Any]:
        contents: list[dict[str, Any]] = []
        for message in request.messages:
            role = _ROLE_MAP.get(message.role.value)
            if role is None:  # a system message mid-stream becomes plain user text
                contents.append({"role": "user", "parts": [{"text": message.content}]})
            else:
                contents.append({"role": role, "parts": [{"text": message.content}]})

        generation: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_output_tokens,
        }
        if request.json_schema:
            generation["responseMimeType"] = "application/json"
            generation["responseSchema"] = request.json_schema

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation,
        }
        if request.system:
            payload["systemInstruction"] = {"parts": [{"text": request.system}]}
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters or {"type": "object", "properties": {}},
                        }
                        for tool in request.tools
                    ]
                }
            ]
        return payload

    # ── streaming ────────────────────────────────────────────────────
    async def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]:
        url = (
            f"{self.config.base_url.rstrip('/')}/models/{self.model}:streamGenerateContent"
            "?alt=sse"
        )
        finish_reason = "stop"
        ended = False

        async for chunk in self._stream_chunks(url, self._payload(request)):
            for event in self._events_from_chunk(chunk):
                if isinstance(event, StreamEnd):
                    finish_reason = event.reason
                    ended = True
                yield event

        if not ended:
            yield StreamEnd(reason=finish_reason)

    def _events_from_chunk(self, chunk: dict[str, Any], state: Any = None) -> list[LlmEvent]:
        events: list[LlmEvent] = []
        if meta := chunk.get("usageMetadata"):
            events.append(
                Usage(
                    prompt_tokens=int(meta.get("promptTokenCount", 0)),
                    completion_tokens=int(meta.get("candidatesTokenCount", 0)),
                )
            )
        for candidate in chunk.get("candidates") or []:
            for part in (candidate.get("content") or {}).get("parts") or []:
                if text := part.get("text"):
                    events.append(TextDelta(text=text))
                if call := part.get("functionCall"):
                    events.append(
                        ToolCallRequest(
                            name=call.get("name", ""),
                            arguments=call.get("args") or {},
                        )
                    )
            if reason := candidate.get("finishReason"):
                events.append(StreamEnd(reason=str(reason).lower()))
        return events



__all__ = ["GeminiProvider"]
