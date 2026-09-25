"""One OpenAI-compatible provider for six backends.

Groq, OpenRouter, Together, HuggingFace's router, Ollama and `llama.cpp` all
speak the same wire protocol.  Writing one implementation and pointing it at
different `base_url`s is the plan's no-duplicated-code rule (R5) applied where
it matters most — provider code is exactly where duplication usually creeps in.

Streaming is SSE (`data: {...}` lines terminated by `data: [DONE]`).
"""

from __future__ import annotations

import json
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


class OpenAiCompatibleProvider(HttpStreamingProvider):
    """Any backend exposing `POST {base_url}/chat/completions` with SSE."""

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "atlas/0.1 (+https://github.com/)",
            **self.config.extra_headers,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        # OpenRouter likes to know the app; harmless elsewhere when omitted.
        if self.name == "openrouter":
            headers.setdefault("X-Title", "Atlas")
        return headers

    def _payload(self, request: LlmRequest) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if request.system:
            messages.append({"role": "system", "content": request.system})
        messages.extend(
            {"role": m.role.value, "content": m.content, **({"name": m.name} if m.name else {})}
            for m in request.messages
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": True,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters or {"type": "object", "properties": {}},
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = "auto"
        if request.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": request.json_schema},
            }
        return payload

    # ── streaming ────────────────────────────────────────────────────
    async def stream(self, request: LlmRequest) -> AsyncIterator[LlmEvent]:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason = "stop"
        ended = False

        async for chunk in self._stream_chunks(url, self._payload(request)):
            for event in self._events_from_chunk(chunk, tool_calls):
                if isinstance(event, StreamEnd):
                    finish_reason = event.reason
                    ended = True
                yield event

        for event in self._flush_tool_calls(tool_calls):
            yield event
        if not ended:
            yield StreamEnd(reason=finish_reason)

    def _events_from_chunk(
        self, chunk: dict[str, Any], tool_calls: dict[int, dict[str, Any]] | None = None
    ) -> list[LlmEvent]:
        tool_calls = tool_calls if tool_calls is not None else {}
        events: list[LlmEvent] = []
        if usage := chunk.get("usage"):
            events.append(
                Usage(
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                )
            )
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if text := delta.get("content"):
                events.append(TextDelta(text=text))
            for call in delta.get("tool_calls") or []:
                index = int(call.get("index", 0))
                slot = tool_calls.setdefault(index, {"name": "", "arguments": ""})
                if name := (call.get("function") or {}).get("name"):
                    slot["name"] = name
                if args := (call.get("function") or {}).get("arguments"):
                    slot["arguments"] += args
            if choice.get("finish_reason"):
                events.append(StreamEnd(reason=str(choice["finish_reason"])))
        return events

    @staticmethod
    def _flush_tool_calls(tool_calls: dict[int, dict[str, Any]]) -> list[LlmEvent]:
        """Streamed tool calls arrive in fragments; emit them once, complete."""
        events: list[LlmEvent] = []
        for index in sorted(tool_calls):
            slot = tool_calls[index]
            if not slot["name"]:
                continue
            try:
                arguments = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {"_raw": slot["arguments"]}
            events.append(ToolCallRequest(name=slot["name"], arguments=arguments))
        return events


__all__ = ["OpenAiCompatibleProvider"]
