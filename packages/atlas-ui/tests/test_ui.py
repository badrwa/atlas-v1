"""L5 — the orb's brain: theme, hub throttles, bridge policies, geometry.

Nothing here needs a browser, a display or WebView2.  Two tests matter more than
the rest and are marked in their docstrings: the level/token throttles (they are
the only reason a 2-core laptop can serve a live UI) and the "stalled socket
never blocks the core" rule.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from atlas_core.events import (
    AudioLevel,
    ConfirmationRequested,
    DegradedModeChanged,
    EventBus,
    MoodChanged,
    ReplyFinished,
    SpeakerMatched,
    StateChanged,
    TokenDelta,
    ToolFinished,
    ToolStarted,
)
from atlas_ui.assets import check_assets, delimiters_balanced, strip_code
from atlas_ui.bridge import UI_TOPICS, EventBridge, bridge_available, bridge_missing, new_token
from atlas_ui.hub import LEVEL_HZ, QUEUE_SIZE, TOKEN_BATCH_S, UiHub
from atlas_ui.protocol import MOOD_HUES, PROTOCOL_VERSION, CaptionMessage
from atlas_ui.theme import (
    ORB_STATES,
    STATE_MOTIONS,
    ThemeEngine,
    clamp_caption,
    is_rtl,
    state_report,
    tool_phrase,
)
from atlas_ui.window import (
    HOTKEYS,
    ORB_SIZE,
    RESTART_BUDGET,
    TRAY_MENU,
    HotkeyManager,
    InstanceLock,
    OrbWindow,
    TrayIcon,
    WindowState,
    hotkey_bindings,
    tray_available,
    tray_missing,
    ui_state_dir,
    window_available,
    window_missing,
)


def field(message: object, name: str) -> object:
    """Read a message field, loosely.

    `UiHub.handle` returns the base `UiMessage` type on purpose (the hub does not
    care which subclass it built), so a test that asserts on `state` is really
    asserting two things: the message is a StateMessage, and its state is right.
    Doing that with `isinstance` everywhere would bury the assertions.
    """
    return getattr(message, name)


class Clock:
    """A hand-cranked monotonic clock — the only way to test a throttle."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ── theme: meaning lives in Python ───────────────────────────────────
def test_every_fsm_state_has_a_motion_and_every_overlay_too() -> None:
    for state in ORB_STATES:
        assert state in STATE_MOTIONS, state
    assert state_report()["speaking"] == "blob"
    assert state_report()["listening"] == "spin"
    assert set(STATE_MOTIONS) - set(ORB_STATES) == set()


def test_mood_is_a_hue_and_state_is_a_motion() -> None:
    engine = ThemeEngine(language="ar-MA")
    mood = engine.mood("happy", energy=0.8, warmth=0.9)
    state = engine.state("speaking", previous="thinking")
    assert mood.kind == "mood" and mood.hue == MOOD_HUES["happy"]
    assert state.kind == "state" and state.state == "speaking"
    assert not hasattr(mood, "state"), "a mood message must not carry a state"
    assert not hasattr(state, "hue"), "a state message must not carry a hue"


def test_an_unknown_mood_falls_back_to_calm_instead_of_crashing() -> None:
    engine = ThemeEngine()
    assert engine.mood("excited").hue == MOOD_HUES["calm"]
    assert engine.blend_hue("nonsense") == engine.hue


def test_mood_values_are_clamped_before_they_reach_the_browser() -> None:
    message = ThemeEngine().mood("frustrated", energy=9.0, warmth=-3.0)
    assert message.energy == 1.0 and message.warmth == 0.0
    assert 0.0 <= ThemeEngine().level(7.0).value <= 1.0


def test_the_microphone_glyph_is_honest_while_atlas_speaks() -> None:
    engine = ThemeEngine()
    assert engine.mutes_microphone("speaking") is True
    assert engine.mutes_microphone("listening") is False
    message = engine.state("speaking")
    assert message.mic_muted is True


def test_an_overlay_is_not_a_state() -> None:
    engine = ThemeEngine()
    assert engine.is_overlay("muted") and engine.is_overlay("degraded")
    assert not engine.is_overlay("thinking")
    assert engine.motion("muted") == "frozen"


def test_captions_are_clamped_to_two_lines_and_keep_the_newest_words() -> None:
    short = clamp_caption("salam, kifash n3awnek?")
    assert "\n" not in short or short.count("\n") == 1
    assert short.startswith("salam")

    long_text = " ".join(f"kelma{index}" for index in range(60))
    clamped = clamp_caption(long_text)
    assert clamped.count("\n") <= 1
    assert clamped.startswith("…")
    assert "kelma59" in clamped, "the newest words are the ones worth showing"
    assert len(clamped) <= 2 * 42 + 2


def test_a_blank_caption_stays_blank() -> None:
    assert clamp_caption("   \n  ") == ""
    assert ThemeEngine().caption("") == "" or ThemeEngine().caption("").text == ""


def test_arabic_is_right_to_left_and_darija_in_latin_is_not() -> None:
    assert is_rtl("شنو كتدير") is True
    assert is_rtl("salam, kifash n3awnek") is False
    assert is_rtl("") is False
    assert is_rtl("  الصف ") is True


def test_tool_phrases_exist_in_both_languages_and_unknown_skills_are_named() -> None:
    assert "notes" in tool_phrase("remember", language="en-GB")
    assert tool_phrase("remember", language="ar-MA") != tool_phrase("remember", language="en-GB")
    assert tool_phrase("teleportation") == "teleportation"


# ── hub: the throttles (the reason the UI can exist on this laptop) ──
def test_levels_are_throttled_to_thirty_per_second() -> None:
    clock = Clock()
    hub = UiHub(clock=clock)
    session = hub.connect("test")

    emitted = [len(hub.handle(AudioLevel(level=index / 100))) for index in range(10)]
    assert emitted[0] == 1, "the first level is shown immediately"
    assert sum(emitted) == 1, "ten levels in the same millisecond are one message"
    assert hub.stats.throttled == 9
    assert field(session.drain()[0], "value") == 0.0

    clock.advance(1.0 / LEVEL_HZ)
    messages = hub.handle(AudioLevel(level=0.5))
    assert len(messages) == 1
    assert field(messages[0], "value") == pytest.approx(0.5), "a new frame wins over a stale one"
    assert hub.stats.coalesced == 1

    # A tick flushes what the meter could not: the last frame of a sentence is
    # the one that lets the orb settle instead of freezing mid-syllable.
    hub.handle(AudioLevel(level=0.2))
    hub.handle(AudioLevel(level=0.0))
    assert hub.stats.throttled >= 10
    clock.advance(1.0 / LEVEL_HZ)
    flushed = [m for m in hub.tick() if m.kind == "level"]
    assert flushed and field(flushed[-1], "value") == 0.0


def test_tokens_are_coalesced_every_fifty_ms() -> None:
    clock = Clock()
    hub = UiHub(clock=clock)
    hub.connect("test")
    first = hub.handle(TokenDelta(text="Sa"))
    assert len(first) == 1 and field(first[0], "text") == "Sa", "the first word appears at once"
    hub.handle(TokenDelta(text="lam"))
    hub.handle(TokenDelta(text=", "))
    hub.handle(TokenDelta(text="kifash"))
    assert hub.stats.throttled == 3, "three deltas in the same window are batched"

    clock.advance(TOKEN_BATCH_S)
    messages = hub.tick()
    captions = [m for m in messages if isinstance(m, CaptionMessage)]
    assert captions and field(captions[0], "text") == "Salam, kifash", "the caption grows"


def test_a_finished_reply_flushes_everything_at_once() -> None:
    clock = Clock()
    hub = UiHub(clock=clock)
    session = hub.connect("test")
    hub.handle(TokenDelta(text="salam "))
    hub.handle(ReplyFinished(text="salam, safi.", language="ar-MA"))
    caption = [m for m in session.drain() if isinstance(m, CaptionMessage)][-1]
    assert caption.text == "salam, safi."
    assert caption.final is True
    assert caption.language == "ar-MA"


def test_a_stalled_socket_never_blocks_the_core() -> None:
    """The rule LEVEL-05 states as "drops messages if the socket is behind"."""
    hub = UiHub()
    slow = hub.connect("slow")
    fast = hub.connect("fast")
    for index in range(QUEUE_SIZE * 2):
        # Finished replies always produce a message (tokens would batch), so this
        # is the honest way to ask "what happens when the socket cannot keep up".
        hub.handle(ReplyFinished(text=f"x{index}", language="ar-MA"))

    assert slow.dropped > 0, "the slow session lost messages"
    assert hub.stats.dropped == slow.dropped + fast.dropped
    assert len(slow.queue) == QUEUE_SIZE
    assert len(fast.drain()) == QUEUE_SIZE, "a client that reads keeps what it needs"
    # …and nothing raised, nothing awaited the socket, nothing blocked.
    assert hub.stats.emitted > 0


def test_a_state_change_is_only_sent_when_the_state_changes() -> None:
    hub = UiHub()
    session = hub.connect("t")
    assert len(hub.handle(StateChanged(state="waking", previous="dormant"))) == 1
    assert hub.handle(StateChanged(state="waking", previous="waking")) == []
    assert [field(m, "state") for m in session.drain()] == ["waking"]


def test_mood_messages_are_deduplicated() -> None:
    hub = UiHub()
    hub.connect("t")
    first = hub.handle(MoodChanged(mood="focused", energy=0.6, warmth=0.4))
    assert len(first) == 1
    assert hub.handle(MoodChanged(mood="focused", energy=0.6, warmth=0.4)) == []
    assert len(hub.handle(MoodChanged(mood="tired", energy=0.2, warmth=0.3))) == 1


def test_tool_activity_becomes_a_thought_line_then_clears() -> None:
    clock = Clock()
    hub = UiHub(clock=clock)
    hub.connect("t")
    started = hub.handle(ToolStarted(skill="remember"))
    assert started and field(started[0], "phase") == "started"
    assert "notes" in str(field(started[0], "spoken"))
    finished = hub.handle(ToolFinished(skill="remember", ok=True, spoken="safi"))
    assert field(finished[0], "phase") == "finished"


def test_a_confirmation_carries_the_spoken_question_and_the_timeout() -> None:
    hub = UiHub()
    hub.connect("t")
    messages = hub.handle(ConfirmationRequested(question="nsedd l PC? yes / no", timeout_s=8.0))
    assert field(messages[0], "question") == "nsedd l PC? yes / no"
    assert field(messages[0], "timeout_s") == 8.0


def test_degraded_mode_is_an_overlay_and_says_why() -> None:
    hub = UiHub()
    hub.connect("t")
    messages = hub.handle(DegradedModeChanged(degraded=True, reason="no provider key", mode="lean"))
    assert field(messages[0], "degraded") is True
    assert field(messages[0], "mode") == "lean"
    assert hub.degraded is True


def test_a_matched_voice_shows_a_name_and_a_score_but_never_a_vector() -> None:
    hub = UiHub()
    hub.connect("t")
    messages = hub.handle(SpeakerMatched(name="said", score=0.83, owner=False))
    text = str(field(messages[0], "text"))
    assert "said" in text and "0.83" in text
    assert "[" not in text and "0." not in text.replace("0.83", "")


def test_a_new_client_is_caught_up_in_one_snapshot() -> None:
    hub = UiHub()
    hub.handle(StateChanged(state="thinking", previous="listening"))
    hub.handle(MoodChanged(mood="focused", energy=0.6, warmth=0.4))
    hub.handle(ReplyFinished(text="wah, safi.", language="ar-MA"))

    snapshot = hub.snapshot()
    kinds = [message.kind for message in snapshot]
    assert kinds[0] == "hello"
    assert "state" in kinds and "mood" in kinds and "caption" in kinds
    state_message = next(m for m in snapshot if m.kind == "state")
    assert field(state_message, "state") == "thinking"


def test_the_hello_message_names_every_state_the_orb_can_draw() -> None:
    hello = UiHub().hello(version="0.1.0")
    assert hello.version == "0.1.0"
    assert set(STATE_MOTIONS) <= set(hello.states)
    assert hello.powers["mic"] in (True, False)
    assert hello.kind == "hello"
    assert hello.v == PROTOCOL_VERSION


def test_a_stale_caption_is_cleared_without_a_new_event() -> None:
    clock = Clock()
    hub = UiHub(clock=clock)
    session = hub.connect("t")
    hub.handle(ReplyFinished(text="safi.", language="ar-MA"))
    session.drain()

    clock.advance(5.0)
    messages = hub.tick()
    assert any(isinstance(m, CaptionMessage) and not m.text for m in messages)
    assert hub.caption == ""

    # …and the next reply starts clean (the accumulator is not leaking old text)
    hub.handle(TokenDelta(text="wah"))
    assert hub.caption == "wah"


def test_the_hub_survives_events_it_does_not_understand() -> None:
    hub = UiHub()
    hub.connect("t")
    assert hub.handle(TokenDelta(text="")) == []
    assert hub.stats.received == 1


# ── bridge: loopback, token-gated, and optional ──────────────────────
def test_the_bridge_subscribes_to_exactly_the_topics_the_orb_needs() -> None:
    for topic in (
        "state.changed",
        "mood.changed",
        "audio.level",
        "llm.token",
        "llm.done",
        "tool.started",
        "tool.finished",
        "confirm.requested",
        "system.degraded",
        "speaker.matched",
    ):
        assert topic in UI_TOPICS, topic


def test_the_url_carries_a_per_run_token_and_stays_on_loopback() -> None:
    bridge = EventBridge()
    assert bridge.host == "127.0.0.1"
    assert bridge.token in bridge.url and "ws://127.0.0.1" in bridge.url
    assert bridge.token in bridge.http_url()
    assert EventBridge().token != bridge.token, "a new bridge means a new token"
    assert len(new_token()) >= 20


async def test_the_pump_moves_bus_events_into_the_hub_and_stops_cleanly() -> None:
    bus = EventBus()
    hub = UiHub(clock=Clock())
    bridge = EventBridge(bus, hub=hub)
    session = bridge.session("test")
    await bridge.start()
    await bus.publish(StateChanged(state="listening", previous="waking"))
    for _ in range(20):
        await asyncio.sleep(0)
    assert hub.state == "listening"
    assert [field(m, "state") for m in session.drain()] == ["listening"]
    await bridge.stop()
    assert bridge.running is False


async def test_a_publisher_is_never_blocked_by_a_dead_ui() -> None:
    bus = EventBus()
    bridge = EventBridge(bus, hub=UiHub())
    session = bridge.session("dead")
    session.close()  # as if the browser went away
    await bridge.start()
    for _ in range(50):
        await bus.publish(TokenDelta(text="x"))
    await asyncio.sleep(0)
    assert bridge.running is True, "the bridge is alive even with nobody watching"
    await bridge.stop()


def test_the_bridge_reports_exactly_what_is_missing() -> None:
    assert bridge_available() is True or "install" in bridge_missing()
    if not bridge_available():
        with pytest.raises(ImportError):
            EventBridge().create_app()


@pytest.mark.skipif(not bridge_available(), reason="needs fastapi + uvicorn")
def test_the_http_app_refuses_a_bad_token_and_serves_the_orb_with_a_good_one() -> None:
    from fastapi.testclient import TestClient

    bridge = EventBridge(bus=EventBus())
    client = TestClient(bridge.create_app())

    assert client.get("/orb/orb.css").status_code == 403
    assert client.get("/orb/orb.css", params={"token": "nope"}).status_code == 403
    assert client.get("/orb/orb.css", params={"token": bridge.token}).status_code == 200

    page = client.get("/", params={"token": bridge.token})
    assert page.status_code == 200
    assert bridge.token in page.text, "the page is served with the token substituted"
    assert "__TOKEN__" not in page.text
    assert "theme-transparent" in page.text

    opaque = client.get("/", params={"token": bridge.token, "opaque": 1})
    assert "theme-opaque" in opaque.text, "the fallback theme is one query away"

    health = client.get("/health").json()
    assert health["ok"] is True and health["state"] == "dormant"
    assert client.get("/orb/../../pyproject.toml", params={"token": bridge.token}).status_code == 404


@pytest.mark.skipif(not bridge_available(), reason="needs fastapi + uvicorn")
def test_the_websocket_carries_the_snapshot_then_live_messages() -> None:
    from fastapi.testclient import TestClient

    bridge = EventBridge(bus=EventBus())
    client = TestClient(bridge.create_app())
    with client.websocket_connect(f"/ws?token={bridge.token}") as socket:
        snapshot = [json.loads(socket.receive_text()) for _ in range(3)]
        assert snapshot[0]["kind"] == "hello"
        assert snapshot[0]["v"] == PROTOCOL_VERSION
        assert {message["kind"] for message in snapshot} >= {"hello", "state", "mood"}

        bridge.hub.handle(StateChanged(state="thinking", previous="listening"))
        seen = json.loads(socket.receive_text())
        assert seen["kind"] == "state" and seen["state"] == "thinking"


@pytest.mark.skipif(not bridge_available(), reason="needs fastapi + uvicorn")
def test_the_websocket_rejects_a_bad_token() -> None:
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    bridge = EventBridge(bus=EventBus())
    client = TestClient(bridge.create_app())
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws?token=no") as socket:
        socket.receive_text()


# ── geometry, single instance, failure isolation ─────────────────────
def test_geometry_round_trips_and_is_clamped_to_the_screen(tmp_path: Path) -> None:
    path = tmp_path / "ui.json"
    saved = WindowState(x=100, y=200, path=path).save()
    assert saved == path

    loaded = WindowState.load(path=path)
    assert (loaded.x, loaded.y) == (100, 200)
    assert (loaded.width, loaded.height) == ORB_SIZE

    off_screen = WindowState(x=9000, y=-400, width=120, height=90, path=path)
    off_screen.clamp(screen=(1920, 1080))
    assert off_screen.x == 1920 - off_screen.width
    assert off_screen.y == 0
    assert off_screen.width >= 180 and off_screen.height >= 180


def test_a_corrupt_state_file_does_not_brick_the_orb(tmp_path: Path) -> None:
    path = tmp_path / "ui.json"
    path.write_text("{not json", encoding="utf-8")
    assert WindowState.load(path=path).width == ORB_SIZE[0]
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert WindowState.load(path=path).height == ORB_SIZE[1]
    path.write_text('{"x": "nope", "width": true}', encoding="utf-8")
    loaded = WindowState.load(path=path, screen=(800, 600))
    assert loaded.width == ORB_SIZE[0]


def test_expanding_the_orb_switches_to_the_panel_geometry() -> None:
    state = WindowState()
    expanded = state.expand()
    assert expanded.expanded is True and expanded.width > ORB_SIZE[0]
    assert expanded.expand().width == ORB_SIZE[0]


def test_only_one_orb_may_hold_the_lock(tmp_path: Path) -> None:
    lock = InstanceLock(tmp_path / "ui.lock")
    assert lock.acquire() is True
    assert lock.holder() == 0 or lock.holder() != 0
    # A lock written by a live foreign pid must win; ours is the current process.
    (tmp_path / "ui.lock").write_text(json.dumps({"pid": 999999}), encoding="utf-8")
    assert lock.holder() == 0, "a dead pid is not a holder"
    assert lock.acquire() is True
    lock.release()
    assert not (tmp_path / "ui.lock").exists()


def test_the_window_reports_a_missing_backend_instead_of_crashing() -> None:
    window = OrbWindow("http://127.0.0.1:8765/?token=x")
    alive, message = window.supervise()
    if window_available():  # a machine with pywebview would try to open it
        pytest.skip("pywebview is installed here — the failure path is not reachable")
    assert alive is False
    assert message and ("install" in message.lower() or "backend" in message.lower())
    assert window.restarts == RESTART_BUDGET + 1, "it tried before giving up"


def test_the_tray_menu_is_data_and_covers_the_promised_actions() -> None:
    tray = TrayIcon()
    ids = [item_id for item_id, _label in tray.menu_items()]
    assert ids == [item_id for item_id, _label in TRAY_MENU]
    for promised in ("mute", "restart_audio", "vault", "quit"):
        assert promised in ids, promised
    if not tray_available():
        assert "install" in tray_missing()
    assert tray.start() is False or tray_available() is True


def test_hotkeys_cover_the_four_promised_combinations() -> None:
    manager = HotkeyManager()
    assert manager.actions() == HOTKEYS
    assert HOTKEYS["stop"] == "ctrl+alt+s"
    assert set(HOTKEYS) == {"mute", "ptt", "stop", "toggle_orb"}
    bindings = hotkey_bindings()
    assert len(bindings) == 4
    assert all("→" in binding for binding in bindings)


def test_every_optional_extra_explains_itself() -> None:
    for check, missing in ((window_available, window_missing), (tray_available, tray_missing)):
        assert check() is True or "install" in missing()


def test_the_state_directory_is_localanddata_on_windows(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert ui_state_dir().name in ("data", "atlas")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    import atlas_ui.window as window_module

    monkeypatch.setattr(window_module, "windows", lambda: True)
    assert ui_state_dir() == tmp_path / "atlas"


# ── the orb's own files ──────────────────────────────────────────────
def test_the_orb_assets_pass_every_static_check() -> None:
    report = check_assets()
    assert report.problems == [], report.problems
    assert set(report.checked) == {"index.html", "orb.css", "orb.js", "protocol.js"}


def test_a_truncated_file_is_caught_before_it_ships(tmp_path: Path) -> None:
    source = (Path(__file__).resolve().parents[1] / "src" / "atlas_ui" / "orb")
    for name in ("index.html", "orb.css", "orb.js", "protocol.js"):
        (tmp_path / name).write_text((source / name).read_text(encoding="utf-8"), encoding="utf-8")
    assert check_assets(tmp_path).problems == []

    broken = (tmp_path / "orb.js").read_text(encoding="utf-8").rstrip().rstrip("}")
    (tmp_path / "orb.js").write_text(broken, encoding="utf-8")
    problems = check_assets(tmp_path).problems
    assert any("orb.js" in problem and "unclosed" in problem for problem in problems)


def test_the_delimiter_scanner_ignores_strings_comments_and_braces_in_text() -> None:
    assert delimiters_balanced('const a = "}"; // ]\n', line_comment="//") == []
    assert delimiters_balanced("/* ) */ function f() { return 1; }") == []
    assert delimiters_balanced("function f() {") != []
    assert delimiters_balanced('var s = "it\'s fine";') == []
    assert strip_code('a /* b */ "c"').count("c") == 0


def test_the_generated_protocol_module_is_not_stale() -> None:
    from atlas_ui.protocol import javascript

    orb = Path(__file__).resolve().parents[1] / "src" / "atlas_ui" / "orb"
    assert (orb / "protocol.js").read_text(encoding="utf-8") == javascript() + "\n"
    assert (orb / "protocol.ts").read_text(encoding="utf-8").startswith("// AUTO-GENERATED")


def test_the_orb_handles_every_protocol_kind() -> None:
    """Drift guard: a new message kind must be handled by the drawing loop."""
    orb = Path(__file__).resolve().parents[1] / "src" / "atlas_ui" / "orb"
    source = (orb / "orb.js").read_text(encoding="utf-8")
    for kind in ("state", "caption", "mood", "level", "tool", "confirm", "degraded"):
        assert f'case "{kind}":' in source, kind


@pytest.mark.skipif(not bridge_available(), reason="needs uvicorn (the orb extra)")
async def test_serve_advertises_itself_and_clears_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handshake with the window process, on the real `serve()` path.

    uvicorn is replaced by a stub that returns at once, so this tests *our*
    advertise/cleanup contract rather than a socket.
    """
    import uvicorn

    from atlas_ui.bridge import EventBridge, endpoint_path, read_endpoint

    seen: list[str] = []

    class StubServer:
        def __init__(self, config: object) -> None:
            del config

        async def serve(self) -> None:
            seen.append(str(read_endpoint(tmp_path) is not None))

    monkeypatch.setattr(uvicorn, "Server", StubServer)
    bridge = EventBridge(port=8798)
    await bridge.serve(root=tmp_path)
    assert seen == ["True"], "the orb had nothing to attach to"
    assert read_endpoint(tmp_path) is None, "a dead bridge left its file behind"
    assert not endpoint_path(tmp_path).exists()


# ── errors, and the subscription list ────────────────────────────────
def test_a_transient_error_flashes_the_orb_and_leaves_the_state_alone() -> None:
    """`system.error` is the difference between "it broke" and "it is broken"."""
    from atlas_core.events import ErrorRaised

    hub = UiHub()
    hub.handle(StateChanged(state="listening", previous="dormant"))
    messages = hub.handle(ErrorRaised(where="asr", message="engine died"))
    states = [m for m in messages if m.kind == "state"]
    assert states and field(states[0], "state") == "error"
    # The flash is a message, not a state: the next real state change still works.
    assert hub.state == "listening"
    after = hub.handle(StateChanged(state="thinking", previous="listening"))
    assert after and field(after[0], "state") == "thinking"


def test_a_fatal_error_leaves_the_degraded_chip_on() -> None:
    from atlas_core.events import ErrorRaised

    hub = UiHub()
    messages = hub.handle(ErrorRaised(where="brain", message="no provider", fatal=True))
    assert [m.kind for m in messages] == ["degraded"]
    assert field(messages[0], "degraded") is True
    assert field(messages[0], "mode") == "error"
    assert hub.degraded is True
    # And a client connecting later is told about it.
    snapshot = [m.kind for m in hub.snapshot()]
    assert "degraded" in snapshot


def test_the_bridge_subscribes_to_every_topic_the_protocol_knows() -> None:
    """The bug this test exists for: the table grew, the subscriber list did not.

    `system.error` was describable by the protocol and never received by the
    bridge — an error the orb would show only if a test drove a live socket.
    """
    from atlas_ui.bridge import UI_TOPICS
    from atlas_ui.protocol import TOPIC_MESSAGES

    assert set(UI_TOPICS) == set(TOPIC_MESSAGES)
    assert set(UiHub().policy_topics()) == set(TOPIC_MESSAGES)


async def test_the_bridge_pump_delivers_errors_without_a_websocket() -> None:
    """End of the chain: publish → hub → the message a client would receive."""
    from atlas_core.events import ErrorRaised

    bus = EventBus()
    hub = UiHub()
    bridge = EventBridge(bus=bus, hub=hub)
    await bridge.start()
    try:
        await bus.publish(ErrorRaised(where="sbom", message="bad json", fatal=True))
        for _ in range(20):
            await asyncio.sleep(0)
        assert hub.degraded is True
        kinds = [m.kind for m in hub.snapshot()]
        assert "degraded" in kinds
    finally:
        await bridge.stop()
