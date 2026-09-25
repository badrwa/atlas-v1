"""Shared HTTP + SSE plumbing for every cloud provider.

Atlas talks to Gemini and to six OpenAI-compatible backends.  What they have in
common is not the payload — that genuinely differs — but the *policy*: how a
socket failure is reported, which HTTP status means "your key is wrong" versus
"slow down", how `data: {...}` frames are decoded, when a stream ended without
saying goodbye.  Written twice, those rules drift, and the drift shows up at the
worst time: when a quota runs out mid-sentence.

So they live here once.  Subclasses supply `_headers()`, `_payload()` and
`_events_from_chunk()`; everything else is inherited.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from atlas_core.config import ProviderConfig
from atlas_core.contracts import HealthReport, LlmEvent, LlmProvider, LlmRequest
from atlas_core.errors import AuthFailed, ProviderError, ProviderUnavailable, RateLimited
from atlas_core.http import OwnedHttpClient

log = logging.getLogger(__name__)


class HttpStreamingProvider(LlmProvider, OwnedHttpClient):
    """An LLM provider that streams server-sent events over HTTP."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        api_key: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        OwnedHttpClient.__init__(self, client, timeout_s=config.timeout_s)
        self.config = config
        self.name = config.name
        self.model = config.model
        self.supports_tools = config.supports_tools
        self.api_key = api_key

    # ── plumbing ─────────────────────────────────────────────────────
    async def health(self) -> HealthReport:
        """One cheap GET: did the credentials and the network both work?"""
        url = f"{self.config.base_url.rstrip('/')}/models"
        try:
            response = await self.client.get(url, headers=self._headers(), timeout=6.0)
        except httpx.HTTPError as exc:
            return HealthReport(ok=False, detail=str(exc)[:120])
        return HealthReport(ok=response.status_code < 400, detail=f"http {response.status_code}")

    # ── streaming ────────────────────────────────────────────────────
    async def _stream_chunks(self, url: str, payload: dict[str, Any]) -> AsyncIterator[dict]:
        """POST `payload` and yield decoded SSE frames until `[DONE]` or EOF."""
        try:
            async with self.client.stream(
                "POST", url, headers=self._headers(), json=payload
            ) as response:
                if response.status_code >= 400:
                    raise self._map_error(response.status_code, await response.aread())

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue
                    if raw == "[DONE]":
                        return
                    try:
                        yield json.loads(raw)
                    except json.JSONDecodeError:
                        log.debug("bad_sse_chunk provider=%s", self.name)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(f"{self.name} timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"{self.name} transport error: {exc}") from exc

    def _map_error(self, status: int, body: bytes) -> ProviderError:
        """HTTP status → typed Atlas error, so the router can react correctly."""
        detail = body.decode("utf-8", "replace")[:300]
        if status in (401, 403):
            return AuthFailed(f"{self.name} auth failed ({status}): {detail}")
        if status == 429:
            return RateLimited(f"{self.name} rate limited: {detail}")
        if status >= 500:
            return ProviderUnavailable(f"{self.name} upstream {status}: {detail}")
        return ProviderError(f"{self.name} http {status}: {detail}")

    # ── subclass hooks ───────────────────────────────────────────────
    def _headers(self) -> dict[str, str]:
        raise NotImplementedError

    def _payload(self, request: LlmRequest) -> dict[str, Any]:
        raise NotImplementedError

    def _events_from_chunk(self, chunk: dict[str, Any], state: Any = None) -> list[LlmEvent]:
        raise NotImplementedError


__all__ = ["HttpStreamingProvider"]
