"""`atlas ui` — the face, and the honest report when the face cannot open.

CI has no display, no WebView2 and no tray.  That is not a limitation here: it is
the state the L5 gate cares about most ("core unaffected if the UI dies"), so
these tests run exactly the machine the plan is worried about.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, cast

import pytest

from atlas.cli import main
from atlas_ui.assets import check_assets

CONFIG = """
[app]
profile = "lean"
language = "ar-MA"
"""


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text(CONFIG, encoding="utf-8")
    yield tmp_path


def run(*args: str) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main(["--config", "config.toml", *args])
        except SystemExit as exc:
            code = int(exc.code or 0)
    return code, out.getvalue() + err.getvalue()


def test_ui_status_names_every_piece_and_how_to_install_it(workspace: Path) -> None:
    code, text = run("ui", "status")
    assert code == 0
    assert "orb assets" in text, "the asset report runs before anything else"
    for piece in ("bridge", "orb", "window", "tray", "hotkeys"):
        assert piece in text, piece
    # On a bare machine each missing extra explains itself rather than crashing.
    assert "pip install" in text or "ready" in text


def test_ui_protocol_still_emits_typescript_and_can_write_it(workspace: Path) -> None:
    del workspace
    code, text = run("ui", "protocol")
    assert code == 0
    assert "export interface StateMessage" in text

    target = Path("protocol.ts")
    code, _ = run("ui", "protocol", "--out", str(target))
    assert code == 0
    assert target.read_text(encoding="utf-8").startswith("// AUTO-GENERATED")

    # The L0 alias is the same implementation, not a second one.
    out, alias = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(alias):
        assert main(["--config", "config.toml", "ui-protocol"]) == 0
    assert "export interface StateMessage" in out.getvalue()


def test_ui_serve_says_what_is_missing_when_fastapi_is_absent(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare install must get the install line, not a traceback."""
    import atlas_ui.bridge as bridge_module

    monkeypatch.setattr(bridge_module, "bridge_available", lambda: False)
    monkeypatch.setattr(
        bridge_module, "bridge_missing", lambda: "fastapi is not installed — pip install 'atlas-ui[orb]'"
    )
    code, text = run("ui", "serve")
    assert code == 1
    assert "pip install 'atlas-ui[orb]'" in text


@pytest.mark.skipif(
    not __import__("atlas_ui.bridge", fromlist=["bridge_available"]).bridge_available(),
    reason="needs fastapi + uvicorn",
)
def test_ui_serve_runs_the_bridge_and_stops(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`atlas ui serve` really serves: /health answers, the page renders, it stops."""
    import asyncio
    import urllib.request

    import atlas_ui.bridge as bridge_module
    from atlas.cli import cmd_ui

    seen: dict[str, object] = {}

    async def serve_briefly(self) -> None:
        import uvicorn

        server = uvicorn.Server(
            uvicorn.Config(
                self.create_app(), host=self.host, port=self.port, log_level="error", access_log=False
            )
        )
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.05)

        def probe() -> None:
            with urllib.request.urlopen(f"http://{self.host}:{self.port}/health") as response:
                seen["health"] = json.loads(response.read())
            with urllib.request.urlopen(self.http_url()) as response:
                seen["page"] = response.read().decode("utf-8")

        await asyncio.to_thread(probe)
        server.should_exit = True
        await task

    monkeypatch.setattr(bridge_module.EventBridge, "serve", serve_briefly)

    class Args:
        config = "config.toml"
        ui_action = "serve"
        opaque = False
        out = ""
        host = "127.0.0.1"
        port = 8791
        demo = True

    from atlas.console import Console

    code = cmd_ui(cast("Any", Args()), Console(color=False))
    assert code == 0
    health = seen["health"]
    assert health["ok"] is True  # type: ignore[index]
    assert "<canvas" in str(seen["page"]), "the orb page is served with its assets"
    assert "__TOKEN__" not in str(seen["page"])


def test_a_second_ui_run_refuses_to_open_a_twin(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single instance is a lock file — and the second launch says so politely.

    The pid in the lock has to belong to a *live foreign process*: our own pid is
    treated as stale on purpose (a crashed run must not lock the orb out).
    """
    import subprocess
    import sys

    from atlas.cli import cmd_ui
    from atlas_ui.window import InstanceLock

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock = InstanceLock()  # data/ui.lock in this cwd — what `atlas ui` checks
        lock.path.parent.mkdir(parents=True, exist_ok=True)
        lock.path.write_text(json.dumps({"pid": child.pid}), encoding="utf-8")

        class Args:
            config = "config.toml"
            ui_action = "run"
            opaque = False
            no_attach = False
            out = ""
            host = ""
            port = 8765
            demo = False

        out, err = io.StringIO(), io.StringIO()
        from atlas.console import Console

        with redirect_stdout(out), redirect_stderr(err):
            code = cmd_ui(cast("Any", Args()), Console(color=False))
        assert code == 1
        assert "already running" in (out.getvalue() + err.getvalue())
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_ui_run_attaches_to_a_live_conversation(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`atlas listen --ui` publishes a URL; `atlas ui run` puts a face on it.

    The window is a viewer: it must not start a second, mute bridge when somebody
    else is already talking.  Here the endpoint is a fake HTTP server, so this is
    the whole handshake without WebView2.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    import atlas.cli as cli
    from atlas.console import Console
    from atlas_ui.bridge import endpoint_path

    class Health(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"ok": True, "sessions": 0, "state": "listening"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        endpoint_path(Path("data")).parent.mkdir(parents=True, exist_ok=True)
        endpoint_path(Path("data")).write_text(
            json.dumps(
                {
                    "url": f"http://127.0.0.1:{server.server_port}/?token=abc",
                    "pid": 4242,
                    "started_at": __import__("time").time(),
                    "protocol": 1,
                }
            ),
            encoding="utf-8",
        )

        opened: dict[str, str] = {}

        class FakeWindow:
            def __init__(self, url: str, **kwargs: object) -> None:
                opened["url"] = url
                self.state = type("S", (), {"width": 220, "height": 220})()
                self.restarts = 0

            def supervise(self) -> tuple[bool, str]:
                return False, "no WebView2 here"  # headless: run the backend inline

            def save_position(self) -> None:
                pass

            def close(self) -> None:
                pass

        import atlas_ui.window as window_module

        # `cmd_ui` imports the name *inside* the function, so the patch goes on
        # the module it imports from.
        monkeypatch.setattr(window_module, "OrbWindow", FakeWindow)

        class Args:
            config = "config.toml"
            ui_action = "run"
            opaque = False
            no_attach = False
            out = ""
            host = ""
            port = 8793
            demo = False

        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.cmd_ui(cast("Any", Args()), Console(color=False))
        text = out.getvalue()
        assert code == 0
        assert "attached" in text, text
        assert opened["url"].startswith(f"http://127.0.0.1:{server.server_port}/"), opened
        assert "data/ui-endpoint.json" not in text
    finally:
        server.shutdown()


def test_a_stale_endpoint_is_ignored(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file left behind by a crash must not hijack the orb."""
    from atlas.cli import _live_endpoint
    from atlas.console import Console
    from atlas_ui.bridge import endpoint_path

    endpoint_path(Path("data")).parent.mkdir(parents=True, exist_ok=True)
    endpoint_path(Path("data")).write_text(
        json.dumps({"url": "http://127.0.0.1:9/?token=x", "pid": 1, "started_at": 15_000_000_000.0}),
        encoding="utf-8",
    )
    out = io.StringIO()
    with redirect_stdout(out):
        found = _live_endpoint(Console(color=False))
    assert found is None
    del monkeypatch


def test_the_orb_assets_are_installable_and_shippable(workspace: Path) -> None:
    """The orb must live inside the package, or a wheel would be a blank window."""
    report = check_assets()
    assert report.ok, report.problems
    package_dir = Path(str(__import__("atlas_ui").__file__)).resolve().parent
    assert (package_dir / "orb" / "index.html").is_file()

    pyproject = Path(str(__file__)).resolve().parents[1] / "pyproject.toml"
    assert pyproject.is_file()  # the app package itself is installed editable


def test_ui_status_never_needs_a_display(workspace: Path) -> None:
    """`--status` is the diagnostic you run *when* the window will not open."""
    code, text = run("ui", "status")
    assert code == 0
    assert "run:" in text
    assert text.count("\n") >= 4


def test_doctor_reports_the_ui_rows(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import atlas.doctor as doctor

    (workspace / "config.toml").write_text(CONFIG, encoding="utf-8")
    code, text = run("doctor")
    assert code in (0, 1)
    assert "orb" in text or "window" in text
    del doctor


def test_the_orb_knows_every_state_the_ears_publish(workspace: Path) -> None:
    """The one contract that spans two packages, checked where both are visible.

    `LoopState` lives in the ears, the motions live in the face; the face must not
    import the ears (the orb has to install on a machine with no audio stack), so
    this app-level test is the place a mismatch is visible.  A state with no
    motion is a frozen orb at the exact moment the user is looking at it.
    """
    from atlas_audio.loop import LoopState
    from atlas_ui.theme import ORB_STATES, ThemeEngine

    missing = [state.value for state in LoopState if state.value not in ORB_STATES]
    assert missing == [], f"the orb does not know {missing}"

    engine = ThemeEngine()
    for state in LoopState:
        assert engine.motion(state.value), state.value


def test_every_published_event_has_a_ui_message_or_is_declared_silent(workspace: Path) -> None:
    """New core events may not be invisible to the orb by accident.

    `atlas_core.events` is the publishing vocabulary; `protocol.TOPIC_MESSAGES` is
    the translation table and `protocol.SILENT_TOPICS` is the reviewed exclusion
    list.  Anything else is a feature nobody would ever see.
    """
    import dataclasses

    from atlas_core import events as core_events
    from atlas_ui.protocol import SILENT_TOPICS, TOPIC_MESSAGES

    topics: set[str] = set()
    for value in vars(core_events).values():
        if not (isinstance(value, type) and issubclass(value, core_events.Event)):
            continue
        if not dataclasses.is_dataclass(value):
            continue
        for entry in dataclasses.fields(value):
            if entry.name == "topic" and isinstance(entry.default, str):
                topics.add(entry.default)
    assert {"state.changed", "llm.token", "mood.changed"} <= topics, topics

    known = set(TOPIC_MESSAGES) | SILENT_TOPICS
    unknown = {topic for topic in topics if topic not in known}
    assert unknown == set(), f"the orb would ignore {sorted(unknown)}"

    from atlas_ui.hub import UiHub

    # The hub's policies and the protocol's table are the same vocabulary.
    assert UiHub().policy_topics() == frozenset(TOPIC_MESSAGES)
    del workspace


@pytest.mark.skipif(
    not __import__("atlas_ui.bridge", fromlist=["bridge_available"]).bridge_available(),
    reason="needs fastapi + uvicorn",
)
def test_listen_ui_serves_the_bus_the_conversation_publishes_on(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`atlas listen --ui`, wired: one bus, served, and advertised on disk.

    The point is not that uvicorn works (test_ui.py covers the bridge) but that
    there is exactly **one** bus: if the CLI built a second one, the orb would sit
    on a live socket showing nothing, which is the failure mode this test exists
    to catch.
    """
    import asyncio
    import json as _json

    import atlas.cli as cli
    import atlas_ui.bridge as bridge_module
    from atlas.console import Console
    from atlas_core.events import AudioLevel, StateChanged, TokenDelta
    from atlas_ui.bridge import endpoint_path

    holder: dict[str, object] = {}

    class RecordingBridge(bridge_module.EventBridge):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            holder["bridge"] = self

    async def fake_serve(self, *, advertise: bool = True, root: object = "data") -> None:
        # uvicorn's own start belongs to test_ui.py; what matters here is that the
        # endpoint is advertised for the run and cleaned up at the end.
        holder["served"] = True
        self.write_endpoint(root)
        try:
            await asyncio.sleep(30)
        finally:
            self.clear_endpoint(root)

    async def fake_live(loop, built, config, console, args, *, recorder, mouth=None):
        for _ in range(20):  # let the serving task reach its first await
            await asyncio.sleep(0)
        bridge = cast("Any", holder["bridge"])
        url = endpoint_path(Path("data"))
        assert url.is_file(), "the orb cannot attach if nothing is advertised"
        advertised = _json.loads(url.read_text(encoding="utf-8"))
        assert advertised["url"] == bridge.http_url()
        assert advertised["pid"] == __import__("os").getpid()
        # A turn happens: the ears and the brain publish, the orb must see it.
        await bridge.bus.publish(StateChanged(state="listening", previous="waking"))
        await bridge.bus.publish(AudioLevel(level=0.42))
        await bridge.bus.publish(TokenDelta(text="Salam"))
        for _ in range(20):
            await asyncio.sleep(0)
        message_kinds = [m.kind for m in bridge.hub.snapshot()]
        assert "state" in message_kinds and "caption" in message_kinds
        assert bridge.hub.state == "listening"
        assert bridge.hub.level == 0.42
        assert "Salam" in (bridge.hub.caption or "")
        return 0

    import atlas_ui

    monkeypatch.setattr(bridge_module.EventBridge, "serve", fake_serve)
    monkeypatch.setattr(atlas_ui, "EventBridge", RecordingBridge)
    monkeypatch.setattr(cli, "_listen_live", fake_live)

    class Args:
        ui = True
        ui_port = 8794
        ptt = False

    out = io.StringIO()
    with redirect_stdout(out):
        code = asyncio.run(
            cli._listen_with_ui(
                None,
                {},
                None,
                Console(color=False),
                Args(),
                bus=bridge_module.EventBridge().bus,
                recorder=None,
                mouth=None,
            )
        )
    assert code == 0
    text = out.getvalue()
    assert "attach: atlas ui run" in text
    assert holder["served"] is True
    assert not endpoint_path(Path("data")).exists(), "the advertisement must be cleaned up"


def test_the_ui_protocol_has_a_generated_runtime_for_the_browser(workspace: Path) -> None:
    """Two artifacts, two consumers: editors read the .ts, WebView2 reads the .js."""
    from atlas_ui.assets import orb_dir
    from atlas_ui.protocol import javascript, typescript

    assert "export interface StateMessage" in typescript()
    runtime = javascript()
    assert "PROTOCOL_VERSION" in runtime and "MOOD_HUES" in runtime
    # The browser copy must be present, or the orb 404s on its own import.
    assert (orb_dir() / "protocol.js").is_file()


def test_the_demo_walkthrough_covers_every_state_and_mood(workspace: Path) -> None:
    """`atlas ui serve --demo` is the manual pass, scripted.

    It walks the orb through the whole vocabulary with no microphone; if someone
    adds a state and forgets the demo, nobody sees it on a laptop until the day
    it matters.
    """
    import re

    import atlas.cli as cli_module
    from atlas_ui.theme import ORB_STATES

    source = Path(str(cli_module.__file__)).read_text(encoding="utf-8")
    demo = source[source.index("async def _demo_stream") : source.index("def _version()")]
    shown = set(re.findall(r'StateChanged\(state="(\w+)"', demo))
    # Every state the ears and the FSM can announce must be scripted here.  Four
    # orb states are not: `dormant` is the face's word for the loop's `idle`,
    # `degraded` arrives as DegradedModeChanged, `muted` comes from the mic gate
    # (Ctrl+Alt+M), and `error` from system.error — each with a named producer.
    for state in ("idle", "waking", "listening", "followup", "thinking", "confirming",
                  "speaking", "stopped"):
        assert state in shown, f"the demo never shows {state}"
    assert set(ORB_STATES) - shown - {"dormant", "degraded", "muted", "error"} == set()
    for event_name in ("MoodChanged", "AudioLevel", "ReplyFinished", "ConfirmationRequested",
                       "DegradedModeChanged", "SpeakerMatched", "TokenDelta"):
        assert event_name in demo, event_name
    del workspace
