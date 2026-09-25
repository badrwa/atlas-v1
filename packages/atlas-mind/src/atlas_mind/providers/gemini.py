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

_ROLE_MAP = {"assistant": "model", "user": "user", "tool": "user"}


class GeminiProvider(LlmProvider):
    """Streaming Gemini client that speaks the same `LlmEvent` language as everyone else."""

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

        try:
            async with self.client.stream(
                "POST", url, headers=self._headers(), json=self._payload(request)
            ) as response:
                if response.status_code >= 400:
                    raise self._map_error(response.status_code, await response.aread())

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError:
                        log.debug("bad_sse_chunk provider=gemini")
                        continue
                    for event in self._events_from_chunk(chunk):
                        if isinstance(event, StreamEnd):
                            finish_reason = event.reason
                            ended = True
                        yield event

        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(f"gemini timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"gemini transport error: {exc}") from exc

        if not ended:
            yield StreamEnd(reason=finish_reason)

    def _events_from_chunk(self, chunk: dict[str, Any]) -> list[LlmEvent]:
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

    @staticmethod
    def _map_error(status: int, body: bytes) -> ProviderError:
        detail = body.decode("utf-8", "replace")[:300]
        if status in (401, 403):
            return AuthFailed(f"gemini auth failed ({status}): {detail}")
        if status == 429:
            return RateLimited(f"gemini rate limited: {detail}")
        if status >= 500:
            return ProviderUnavailable(f"gemini upstream {status}: {detail}")
        return ProviderError(f"gemini http {status}: {detail}")

    # ── health ───────────────────────────────────────────────────────
    async def health(self) -> HealthReport:
        url = f"{self.config.base_url.rstrip('/')}/models"
        try:
            response = await self.client.get(url, headers=self._headers(), timeout=6.0)
        except httpx.HTTPError as exc:
            return HealthReport(ok=False, detail=str(exc)[:120])
        return HealthReport(ok=response.status_code < 400, detail=f"http {response.status_code}")


__all__ = ["GeminiProvider"]
