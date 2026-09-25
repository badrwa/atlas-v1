"""One HTTP client, created once, closed by the side that made it.

The brain's providers and the ears' cloud ASR both need exactly this: an
`httpx.AsyncClient` created on first use, reused for the process's lifetime, and
closed only if we created it.  An injected client belongs to the caller — a test
transport, or a shared client in the app — and must survive us.

Two copies of this rule already meant one of them leaked a connection pool in
L1.  It lives here now.
"""

from __future__ import annotations

import httpx


class OwnedHttpClient:
    """Mixin: a lazily created, reference-counted-by-ownership HTTP client."""

    timeout_s: float = 20.0

    def __init__(self, client: httpx.AsyncClient | None = None, *, timeout_s: float = 20.0) -> None:
        self._client = client
        self._owns_client = client is None
        self.timeout_s = timeout_s

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s)
            self._owns_client = True
        return self._client

    async def aclose(self) -> None:
        """Close the client *we* made; never one that was handed to us."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None


__all__ = ["OwnedHttpClient"]
