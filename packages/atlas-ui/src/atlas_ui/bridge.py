"""The loopback bridge: `EventBus` → WebSocket → the orb.

Design constraints, in the order they matter on a 2-core laptop:

1. **The UI can never slow the conversation.**  The bridge reads a queue that
   `EventBus` fills with a drop policy, converts through `UiHub` (throttled), and
   pushes to bounded per-session deques.  Nothing here awaits the browser.
2. **It is loopback-only and token-gated.**  A local WebSocket that accepts any
   origin is a remote-control port: the URL carries a per-run random token, the
   socket checks it, and the HTTP server binds `127.0.0.1`.
3. **FastAPI is optional.**  `atlas-ui` imports without it (the hub and the theme
   are what the tests use); `atlas ui` says what to install instead of raising.

The WebSocket layer is intentionally thin — every decision it could make is
already made in `hub.py`, so there is exactly one place where "what does the orb
see" is defined.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import secrets
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from atlas_core.events import Event, EventBus
from atlas_ui.hub import UiHub, UiSession
from atlas_ui.protocol import PROTOCOL_VERSION, TOPIC_MESSAGES
from atlas_ui.theme import ThemeEngine

log = logging.getLogger(__name__)

# FastAPI is an optional extra, and these names must exist at *module* level.
# `from __future__ import annotations` makes `socket: WebSocket` a string that
# FastAPI resolves in this module's globals; importing it lazily inside the route
# turned the websocket parameter into a required query parameter (close code
# 1008, `missing query param 'socket'`) — the bug this comment exists to prevent.
FastAPI: Any = None
WebSocket: Any = None
WebSocketDisconnect: Any = RuntimeError
FileResponse: Any = None
HTMLResponse: Any = None
JSONResponse: Any = None
FASTAPI_ERROR = ""
try:  # pragma: no branch - present on any machine with the [orb] extra
    from fastapi import FastAPI as _FastAPI
    from fastapi import WebSocket as _WebSocket
    from fastapi import WebSocketDisconnect as _WebSocketDisconnect
    from fastapi.responses import FileResponse as _FileResponse
    from fastapi.responses import HTMLResponse as _HTMLResponse
    from fastapi.responses import JSONResponse as _JSONResponse

    FastAPI, WebSocket, WebSocketDisconnect = _FastAPI, _WebSocket, _WebSocketDisconnect
    FileResponse, HTMLResponse, JSONResponse = _FileResponse, _HTMLResponse, _JSONResponse
except ImportError as exc:  # pragma: no cover - a bare install
    FASTAPI_ERROR = str(exc)

#: How often the hub is ticked while the bridge runs (caption flush, stale clear).
TICK_HZ = 20.0
#: Topics the orb cares about — subscribing narrowly keeps the queue quiet.
#: What the bridge listens for.  Derived from the protocol's translation table
#: rather than typed out again: a topic the protocol can describe but nobody
#: subscribed to is an event the orb never receives (`system.error` was exactly
#: that for one commit — caught by driving a live socket, not by a unit test).
UI_TOPICS: tuple[str, ...] = tuple(sorted(TOPIC_MESSAGES))


def orb_dir() -> Path:
    """Where the HTML lives — inside the installed package, not the repo."""
    return Path(__file__).resolve().parent / "orb"


#: Where a running bridge parks its URL so the orb can attach to it from another
#: process.  Two processes on purpose: the conversation keeps running if the
#: window dies, and WebView2 owns its own crash domain.
ENDPOINT_FILE = "ui-endpoint.json"

#: An endpoint older than this is treated as debris from a previous boot.
ENDPOINT_MAX_AGE_S = 12 * 3600


def new_token() -> str:
    return secrets.token_urlsafe(18)


def endpoint_path(root: str | Path = "data") -> Path:
    return Path(root) / ENDPOINT_FILE


def read_endpoint(root: str | Path = "data") -> dict[str, Any] | None:
    """The bridge currently advertised on this machine, or None.

    The caller probes `/health` before trusting it: a pid can be recycled, a file
    can survive a crash, and only the server itself can say it is still there.
    """
    path = endpoint_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not payload.get("url"):
        return None
    started = float(payload.get("started_at") or 0)
    if started and time.time() - started > ENDPOINT_MAX_AGE_S:
        return None
    return payload


def probe_endpoint(url: str, *, timeout: float = 1.5) -> dict[str, Any] | None:
    """Ask a live bridge who it is.  None means "nobody is listening"."""
    import urllib.error
    import urllib.request

    health = url.split("?", 1)[0].rstrip("/")
    if not health.endswith("/health"):
        health = f"{health}/health"
    try:
        with urllib.request.urlopen(health, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return payload if isinstance(payload, dict) else None


def bridge_available() -> bool:
    """Can this machine serve the orb at all?"""
    return _module_present("fastapi") and _module_present("uvicorn")


def bridge_missing() -> str:
    if not _module_present("fastapi"):
        return "fastapi is not installed — pip install 'atlas-ui[orb]'"
    if not _module_present("uvicorn"):
        return "uvicorn is not installed — pip install 'atlas-ui[orb]'"
    return ""


def _module_present(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


#: Loopback by default.  A bridge that answers the network is a remote control.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


class EventBridge:
    """Runs the bus → hub pump and (optionally) the HTTP server around it."""

    def __init__(
        self,
        bus: EventBus | None = None,
        *,
        hub: UiHub | None = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        token: str = "",
        version: str = str(PROTOCOL_VERSION),
    ) -> None:
        self.bus = bus or EventBus()
        self.hub = hub or UiHub(theme=ThemeEngine())
        self.host = host
        self.port = port
        self.token = token or new_token()
        self.version = version
        self._queue: asyncio.Queue[Event] | None = None
        self._pump: asyncio.Task[None] | None = None
        self._tick: asyncio.Task[None] | None = None
        self._server: Any = None
        self.running = False

    # ── the pump ─────────────────────────────────────────────────────
    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._queue = self.bus.subscribe(*UI_TOPICS)
        self._pump = asyncio.create_task(self._run_pump(), name="atlas-ui-pump")
        self._tick = asyncio.create_task(self._run_tick(), name="atlas-ui-tick")

    async def stop(self) -> None:
        self.running = False
        for task in (self._pump, self._tick):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        self._pump = self._tick = None

    async def _run_pump(self) -> None:
        assert self._queue is not None
        while self.running:
            event = await self._queue.get()
            with contextlib.suppress(Exception):  # never let a bad event kill the UI
                self.hub.handle(event)

    async def _run_tick(self) -> None:
        interval = 1.0 / TICK_HZ
        while self.running:
            await asyncio.sleep(interval)
            with contextlib.suppress(Exception):
                self.hub.tick()

    # ── sessions ─────────────────────────────────────────────────────
    def session(self, name: str = "") -> UiSession:
        return self.hub.connect(name)

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws?token={self.token}"

    def http_url(self) -> str:
        return f"http://{self.host}:{self.port}/?token={self.token}"

    def attach_url(self) -> str:
        """The URL the orb process opens: token included, loopback only."""
        return self.http_url()

    def write_endpoint(self, root: str | Path = "data") -> Path:
        """Advertise this bridge so another process can attach to it."""
        path = endpoint_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "url": self.attach_url(),
                    "pid": os.getpid(),
                    "started_at": time.time(),
                    "protocol": self.version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def clear_endpoint(self, root: str | Path = "data", *, only_if_mine: bool = True) -> None:
        """Remove the advertisement — but never somebody else's."""
        path = endpoint_path(root)
        try:
            if only_if_mine:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if int(payload.get("pid") or 0) != os.getpid():
                    return
            path.unlink(missing_ok=True)
        except (OSError, ValueError):  # a corrupt file is debris: remove it
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)

    # ── the app ──────────────────────────────────────────────────────
    def create_app(self) -> Any:
        """Build the FastAPI app.  Raises ImportError with the install hint."""
        if FastAPI is None:  # pragma: no cover - exercised in the CLI test
            raise ImportError(bridge_missing() or FASTAPI_ERROR)

        app = FastAPI(title="Atlas orb bridge", docs_url=None, redoc_url=None, openapi_url=None)
        hub = self.hub

        @app.get("/")
        async def index(token: str = "", opaque: int = 0) -> Any:
            if token != self.token:
                return JSONResponse({"error": "bad token"}, status_code=403)
            page = orb_dir() / "index.html"
            if not page.is_file():
                return JSONResponse({"error": "orb assets missing"}, status_code=500)
            # The token and the theme are substituted, not templated: one HTML
            # file, two realities (transparent orb, or an opaque card on drivers
            # where a transparent frameless window flickers).
            html = page.read_text(encoding="utf-8")
            html = html.replace("__TOKEN__", self.token)
            html = html.replace(
                "theme-transparent", "theme-opaque" if opaque else "theme-transparent"
            )
            return HTMLResponse(html)

        @app.get("/orb/{name}")
        async def asset(name: str, token: str = "") -> Any:
            if token != self.token:
                return JSONResponse({"error": "bad token"}, status_code=403)
            path = (orb_dir() / name).resolve()
            if not path.is_file() or orb_dir().resolve() not in path.parents:
                return JSONResponse({"error": "not found"}, status_code=404)
            return FileResponse(path)

        @app.get("/health")
        async def health() -> Any:
            return {
                "ok": True,
                "sessions": hub.session_count(),
                "state": hub.state,
                "stats": hub.stats.as_dict(),
            }

        @app.websocket("/ws")
        async def websocket(socket: WebSocket, token: str = "") -> None:
            if token != self.token:
                await socket.close(code=1008)
                return
            await socket.accept()
            session = hub.connect(name=str(socket.client))
            try:
                for message in hub.snapshot():
                    await socket.send_text(message.model_dump_json())
                while True:
                    for message in session.drain():
                        await socket.send_text(message.model_dump_json())
                    await asyncio.sleep(1.0 / TICK_HZ)
            except (WebSocketDisconnect, RuntimeError):
                pass
            finally:
                session.close()
                log.info("ui_session_closed sessions=%d", hub.session_count())

        return app

    # ── serving ──────────────────────────────────────────────────────
    async def serve(
        self, *, advertise: bool = True, root: str | Path = "data"
    ) -> None:  # pragma: no cover - needs uvicorn at runtime
        """Run the HTTP server until cancelled (uvicorn, loopback only).

        `advertise=True` writes `data/ui-endpoint.json` for the first two seconds
        of the run: that file is how `atlas ui run` finds a conversation that is
        already going on and puts a face on it.
        """
        import uvicorn

        config = uvicorn.Config(
            self.create_app(),
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        if advertise:
            self.write_endpoint(root)
        try:
            await self._server.serve()
        finally:
            if advertise:
                self.clear_endpoint(root)

    def stats(self) -> dict[str, Any]:
        return {
            "url": self.http_url(),
            "sessions": self.hub.session_count(),
            "state": self.hub.state,
            "running": self.running,
            **self.hub.stats.as_dict(),
        }


def status_rows(root: str | Path = "data") -> list[tuple[str, str]]:
    """`atlas ui status` / doctor rows: what would run, honestly."""
    rows: list[tuple[str, str]] = []
    rows.append(("bridge", "ready" if bridge_available() else bridge_missing()))
    advertised = read_endpoint(root)
    if advertised:
        rows.append(("attached", f"conversation running (pid {advertised.get('pid', '?')})"))
    module = orb_dir() / "index.html"
    rows.append(
        ("orb", f"{module.parent} ({'present' if module.is_file() else 'missing'})"),
    )
    from atlas_ui.window import (
        hotkey_bindings,
        hotkeys_available,
        hotkeys_missing,
        tray_available,
        tray_missing,
        window_available,
        window_missing,
    )

    rows.append(("window", "pywebview ready (WebView2)" if window_available() else window_missing()))
    rows.append(("tray", "pystray ready" if tray_available() else tray_missing()))
    rows.append(("hotkeys", ", ".join(hotkey_bindings()) if hotkeys_available() else hotkeys_missing()))
    return rows


def topics() -> Iterable[str]:
    return UI_TOPICS


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "TICK_HZ",
    "UI_TOPICS",
    "EventBridge",
    "bridge_available",
    "bridge_missing",
    "new_token",
    "orb_dir",
    "status_rows",
    "topics",
]
