"""One OpenAI-compatible provider for six backends.

Groq, OpenRouter, Together, HuggingFace's router, Ollama and `llama.cpp` all
speak the same wire protocol.  Writing one implementation and pointing it at
different `base_url`s is the plan's no-duplicated-code rule (R5) applied where
it matters most — provider code is exactly where duplication usually creeps in.

Streaming is SSE (`data: {...}` lines terminated by `data: [DONE]`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from atlas_core.config import ProviderConfig
from atlas_core.contracts import (
    HealthReport,
    LlmEvent,
    LlmProvider,
    LlmRequest,
    StreamEnd,
    TextDelta,
    ToolCallRequest,
    Usage,
)
from atlas_core.errors import AuthFailed, ProviderError, ProviderUnavailable, RateLimited

log = logging.getLogger(__name__)


class OpenAiCompatibleProvider(LlmProvider):
    """Any backend exposing `POST {base_url}/chat/completions` with SSE."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        api_key: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self.model = config.model
        self.supports_tools = config.supports_tools
        self.api_key = api_key
        self._client = client
        self._owns_client = client is None

    # ── plumbing ─────────────────────────────────────────────────────
    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout_s)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

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

        try:
            async with self.client.stream(
                "POST", url, headers=self._headers(), json=self._payload(request)
            ) as response:
                if response.status_code >= 400:
                    raise self._map_error(response.status_code, await response.aread())

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        log.debug("bad_sse_chunk provider=%s", self.name)
                        continue

                    for event in self._events_from_chunk(chunk, tool_calls):
                        if isinstance(event, StreamEnd):
                            finish_reason = event.reason
                            ended = True
                        yield event

        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(f"{self.name} timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{self.name} transport error: {exc}") from exc

        for event in self._flush_tool_calls(tool_calls):
            yield event
        if not ended:
            yield StreamEnd(reason=finish_reason)

    def _events_from_chunk(
        self, chunk: dict[str, Any], tool_calls: dict[int, dict[str, Any]]
    ) -> list[LlmEvent]:
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

    @staticmethod
    def _map_error(status: int, body: bytes) -> ProviderError:
        detail = body.decode("utf-8", "replace")[:300]
        if status in (401, 403):
            return AuthFailed(f"auth failed ({status}): {detail}")
        if status == 429:
            return RateLimited(f"rate limited: {detail}")
        if status >= 500:
            return ProviderUnavailable(f"upstream {status}: {detail}")
        return ProviderError(f"http {status}: {detail}")

    # ── health ───────────────────────────────────────────────────────
    async def health(self) -> HealthReport:
        url = f"{self.config.base_url.rstrip('/')}/models"
        try:
            response = await self.client.get(url, headers=self._headers(), timeout=5.0)
        except httpx.HTTPError as exc:
            return HealthReport(ok=False, detail=str(exc)[:120])
        ok = response.status_code < 400
        return HealthReport(ok=ok, detail=f"http {response.status_code}")


__all__ = ["OpenAiCompatibleProvider"]
