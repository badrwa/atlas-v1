"""The `atlas` command.

Everything here is wiring: config in, package out, result printed.  If a command
grows logic worth testing, it moves into a package — which is why `doctor.py`
lives next door and this file mostly parses arguments and formats output.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atlas import __version__
from atlas.console import CYAN, DIM, Console
from atlas_core.config import AppConfig, ConfigError, load_config
from atlas_core.contracts import Capability, LanguageTag
from atlas_core.events import EventBus
from atlas_core.identity import Audience, DailyGreeter, IdentityConfig

log = logging.getLogger("atlas.cli")

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

USAGE_HINT = "config: config.toml (override with --config or $ATLAS_CONFIG)"


# ── shared plumbing ──────────────────────────────────────────────────
def _load(config_path: str | None, console: Console) -> AppConfig | None:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        console.fail("config", str(exc))
        console.write(f"  {console.paint('→ copy config.toml from the repo root, then retry', DIM)}")
        return None


def _session(config: AppConfig, console: Console, *, structured: bool = False, events=None):
    """Build a chat session wired to every provider that has a key.

    `events` may be an existing bus: the conversation and the orb then share one
    stream, which is what `atlas listen --ui` does (no second way to watch a turn).
    """
    from atlas_core.events import EventBus
    from atlas_core.timings import TimingRecorder
    from atlas_mind import ChatSession, MoodEngine, ProviderRouter, build_providers
    from atlas_mind.router import QuotaTracker

    providers = build_providers(config)
    tracker = QuotaTracker(Path("data/quota.json"))
    events = events or EventBus()
    timings = TimingRecorder(Path("data/timings.jsonl"))
    router = ProviderRouter(
        providers,
        tracker=tracker,
        events=events,
        fallback_language=config.app.language,
    )
    session = ChatSession(
        config,
        router,
        timings=timings,
        events=events,
        mood=MoodEngine(),
        structured=structured,
    )
    return session, router, timings


# ── commands ─────────────────────────────────────────────────────────
def cmd_doctor(args: argparse.Namespace, console: Console) -> int:
    from atlas.doctor import render, run_checks

    report = run_checks(config_path=args.config)
    render(report, console, config_path=args.config or "config.toml")
    return 1 if report.failures else 0


def cmd_providers(args: argparse.Namespace, console: Console) -> int:
    config = _load(args.config, console)
    if config is None:
        return 1

    console.title("Providers")
    usable = [p for p in config.providers if p.is_configured()]
    rows: list[tuple[str, str, str, str, str]] = []
    for entry in config.providers:
        if entry.kind == "llama_cpp":
            from atlas_mind.providers.llama_cpp import SidecarSpec

            spec = SidecarSpec()
            gaps = "missing " + ", ".join(spec.missing) if spec.missing else "ready"
            rows.append((entry.name, entry.kind, entry.model, "local", f"{'on' if entry.enabled else 'off'} · {gaps}"))
            continue
        rows.append(
            (
                entry.name,
                entry.kind,
                entry.model,
                "local" if entry.is_local else ("key ✓" if entry.is_configured() else "no key"),
                "on" if entry.enabled else "off",
            )
        )
    console.table(("provider", "kind", "model", "key", "state"), rows)
    console.write()
    console.write(f"fallback order: {console.paint(' → '.join(p.name for p in usable) or '—', DIM)}")

    if not usable:
        console.write()
        console.warn("no usable provider", "add a key to .env (see .env.example)")
    elif args.live:
        console.write()
        console.title("Live check (one tiny request each)")
        asyncio.run(_live_check(config, console))
    else:
        console.write(f"{console.paint('· --live tries each provider for real', DIM)}")
    return 0 if usable else 1


async def _live_check(config: AppConfig, console: Console) -> None:
    import time

    from atlas_core.contracts import LlmRequest, Message, Role, TextDelta
    from atlas_mind import build_providers

    request = LlmRequest(
        messages=[Message(role=Role.USER, content="Jaweb b 'salam' f kelma wa7da.")],
        max_output_tokens=12,
    )
    for provider in build_providers(config):
        started = time.perf_counter()
        first: float | None = None
        text = ""
        try:
            async for event in provider.stream(request):
                if isinstance(event, TextDelta) and event.text:
                    first = first if first is not None else time.perf_counter()
                    text += event.text
                if len(text) > 40:
                    break
            total = (time.perf_counter() - started) * 1000
            ttft = f"{(first - started) * 1000:.0f}ms" if first else "—"
            console.ok(provider.name, f"ttft {ttft} · total {total:.0f}ms · {text.strip()[:40]!r}")
        except Exception as exc:
            reason = str(exc).splitlines()[0][:90]
            console.fail(provider.name, reason)


def cmd_chat(args: argparse.Namespace, console: Console) -> int:
    config = _load(args.config, console)
    if config is None:
        return 1
    try:
        return asyncio.run(_chat_loop(config, console, once=args.once, structured=args.structured))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.write()
        console.write("bslama 👋")
        return 0


async def _chat_loop(
    config: AppConfig, console: Console, *, once: str = "", structured: bool = False
) -> int:
    from atlas_mind.chat import TurnResult

    session, router, timings = _session(config, console, structured=structured)

    console.title("ATLAS")
    console.write(
        console.paint(
            f"brain: {router.describe()} · language: {config.app.language} "
            f"(secondary {config.app.secondary_language})",
            DIM,
        )
    )
    console.write(
        console.paint(f"mode: {'structured' if structured else 'fast'} · /help for commands", DIM)
    )
    console.write()

    while True:
        if once:
            utterance = once
        else:
            try:
                utterance = console.ask("you ▸ ")
            except (EOFError, KeyboardInterrupt):
                console.write()
                return 0

        text = utterance.strip()
        if not text:
            continue
        if text.startswith("/"):
            if _command(text, session, router, timings, console):
                continue
            return 0

        result = TurnResult()
        console.write(f"{console.paint('atlas ▸', CYAN)} ", end="")
        async for delta in session.stream(text, _result=result):
            console.write(delta, end="")
        console.write()
        _print_meta(result, console)
        if once:
            return 0


def _print_meta(result, console: Console) -> None:
    """One honest line under every reply: who answered, how fast, what mood."""
    if result.ttft_ms is None:
        console.write()
        return
    parts = [
        result.provider or "canned",
        f"ttft {result.ttft_ms:.0f}ms",
        f"total {result.total_ms:.0f}ms",
        result.language,
    ]
    if result.emotion:
        parts.append(result.emotion)
    if result.mood:
        parts.append(f"mood {result.mood.get('label')}")
    if result.repaired:
        parts.append("repaired")
    if result.followup:
        parts.append("asks back")
    console.write(console.paint("  [" + " · ".join(parts) + "]", DIM))
    console.write()


def _command(text: str, session, router, timings, console: Console) -> bool:
    """Handle a /command. Returns False when the loop should end."""
    parts = text.split()
    command, rest = parts[0].lower(), parts[1:]

    if command in {"/quit", "/exit", "/bye"}:
        console.write("bslama 👋")
        return False
    if command == "/help":
        console.table(
            ("command", "what it does"),
            [
                ("/lang <ar-MA|en-GB>", "switch language now"),
                ("/provider <name|auto>", "pin one brain, or let the chain decide"),
                ("/mood", "how Atlas reads the room right now"),
                ("/structured", "toggle metadata mode (slower first token)"),
                ("/reset", "forget this conversation"),
                ("/timing", "latency of the last turns"),
                ("/providers", "who can answer right now"),
                ("/quit", "leave"),
            ],
        )
    elif command == "/lang":
        if not rest:
            console.write(f"language is {session.current_language}")
        elif rest[0] not in {"ar-MA", "en-GB"}:
            console.warn("unknown language", "use ar-MA or en-GB")
        else:
            session.language.route("", hint=rest[0])
            console.write(f"language {session.current_language}")
    elif command == "/reset":
        session.reset()
        console.write("conversation cleared")
    elif command == "/timing":
        console.write(timings.format_summary() if timings else "no timings recorded yet")
    elif command == "/providers":
        console.write(router.describe())
    elif command == "/provider":
        _set_provider(rest, router, console)
    elif command == "/mood":
        console.table(("key", "value"), [(k, str(v)) for k, v in session.mood.view.as_dict().items()])
        if session.mood.history:
            console.write(console.paint(f"  recent: {' → '.join(session.mood.history[-8:])}", DIM))
    elif command == "/structured":
        session.structured = not session.structured
        console.write(f"structured mode {'on' if session.structured else 'off'}")
    else:
        console.warn("unknown command", f"{command} — try /help")
    return True


def _find_template(explicit: str | None = None) -> Path | None:
    """Locate vault-template/: --template, $ATLAS_VAULT_TEMPLATE, repo, cwd."""
    import os

    candidates = [
        explicit,
        os.environ.get("ATLAS_VAULT_TEMPLATE"),
        "vault-template",
        str(Path(__file__).resolve().parents[4] / "vault-template"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


def _set_provider(rest: list[str], router, console: Console) -> None:
    """Pin one provider, or hand control back to the fallback chain."""
    if not rest:
        console.write(f"order: {router.describe()}")
        return
    wanted = rest[0].lower()
    if wanted in {"auto", "chain"}:
        router.pinned = ""
        console.write(f"auto — {router.describe()}")
        return
    if wanted not in router.names:
        console.warn("unknown provider", f"{wanted} — known: {', '.join(router.names)}")
        return
    router.pinned = wanted
    console.write(f"pinned to {wanted}{console.paint(' (falls back if it fails)', DIM)}")


def cmd_vault(args: argparse.Namespace, console: Console) -> int:
    config = _load(args.config, console)
    if config is None:
        return 1

    from atlas_core.config import ObsidianSection
    from atlas_obsidian import VaultAdapter

    if args.vault_action == "init":
        target = Path(args.path or args.vault_path or "").expanduser()
        if not args.path and not args.vault_path:
            console.fail("vault init", "tell me where: atlas vault init ~/AtlasVault")
            return 1
        template = _find_template(args.template)
        if template is None:
            console.fail("vault init", "vault-template/ not found")
            console.write(f"  {console.paint('→ run from the repo root, or --template <path>', DIM)}")
            return 1

        adapter = VaultAdapter.init_from_template(target, template, git=not args.no_git)
        console.title("Vault")
        console.ok("created", str(target))
        console.write(f"  from {template} · git: {'yes' if not args.no_git else 'no'}")
        console.write()
        console.write(f"  {console.paint('next:', DIM)} set vault_path = \"{target}\" in config.toml")
        console.write(f"  {console.paint('then:', DIM)} atlas vault status")
        return 0

    vault_path = args.vault_path or args.path or config.obsidian.vault_path
    if not vault_path:
        console.warn("no vault configured", "atlas vault init ~/AtlasVault")
        return 1

    adapter = VaultAdapter(
        ObsidianSection(vault_path=str(Path(vault_path).expanduser()), git_commit_writes=True)
    )

    if args.vault_action == "status":
        if not adapter.exists:
            console.fail("vault", f"not found: {vault_path}")
            return 1
        console.title("Vault")
        console.table(("key", "value"), [(k, str(v)) for k, v in adapter.stats().items()])
        recent = adapter.recent_notes(5)
        if recent:
            console.write()
            console.write(f"recent: {', '.join(note.title for note in recent)}")
        return 0

    if args.vault_action == "undo":
        if adapter.undo_last_write():
            console.ok("undo", "last Atlas write reverted")
            return 0
        console.warn("undo", "nothing to undo")
        return 1

    console.fail("vault", f"unknown action {args.vault_action!r}")
    return 1


def cmd_skills(args: argparse.Namespace, console: Console) -> int:
    import time

    from atlas_skills import (
        OpenUrlSkill,
        ShutdownSkill,
        SkillRegistry,
        SystemStatsSkill,
    )

    registry = SkillRegistry()
    for built in (SystemStatsSkill(), OpenUrlSkill(), ShutdownSkill()):
        registry.register(built)
    if args.skills_action == "audit":
        console.title("Skill audit")
        rows = registry.audit_tail(args.limit)
        if not rows:
            console.write("nothing called yet")
        else:
            console.table(
                ("when", "skill", "ok", "spoken"),
                [
                    (
                        time.strftime("%H:%M:%S", time.localtime(entry.at)),
                        entry.skill,
                        "✓" if entry.ok else "✗",
                        entry.spoken[:40],
                    )
                    for entry in rows
                ],
            )
        return 0

    console.title("Skills")
    skill_rows: list[tuple[str, str, str]] = []
    for name in registry.names():
        registered = registry.get(name)
        skill_rows.append(
            (
                registered.name,
                registered.permission.value,
                "owner only" if registered.owner_only else "",
            )
        )
    console.table(("skill", "permission", "restriction"), skill_rows)
    console.write()
    console.write(registry.describe("ar-MA"))
    console.write()
    console.write(console.paint("destructive skills ask first; a stranger never counts", DIM))
    return 0


def cmd_ui(args: argparse.Namespace, console: Console) -> int:
    """The face: check it, serve it, or open the window.

    Three modes, and each one is honest about what is missing:

    * `status` — what would run (bridge, orb files, window, tray, hotkeys);
    * `serve` — the bridge on loopback, no window: usable in a normal browser;
    * (default) — bridge + orb window + tray + hotkeys, with failure isolation
      so a missing WebView2 leaves the ears and the brain running.
    """
    config = _load(args.config, console)
    if config is None:
        return 1

    from atlas_ui import EventBridge, WindowState, check_assets, status_rows
    from atlas_ui.bridge import bridge_available, bridge_missing
    from atlas_ui.window import (
        HotkeyManager,
        InstanceLock,
        OrbWindow,
        TrayIcon,
        hotkeys_available,
        hotkeys_missing,
    )

    action = args.ui_action or "run"

    if action == "protocol":
        from atlas_ui import typescript

        text = typescript()
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            console.ok("written", args.out)
        else:
            console.write(text)
        return 0

    report = check_assets()
    for line in report.as_lines():
        console.write(console.paint(f"  {line}", DIM) if report.ok else line)
    for problem in report.problems:
        console.write(console.paint(f"    · {problem}", DIM))

    if action == "status":
        console.title("The orb")
        console.table(("piece", "state"), status_rows())
        console.write()
        console.write(
            console.paint(
                "  run: atlas ui            (bridge + orb window + tray + hotkeys)",
                DIM,
            )
        )
        console.write(console.paint("       atlas ui serve      (bridge only — open the URL in a browser)", DIM))
        return 0

    if not bridge_available():
        console.fail("cannot serve the orb", bridge_missing())
        return 1

    bridge = EventBridge(host=args.host or "127.0.0.1", port=args.port, version=_version())

    if action == "serve":
        return asyncio.run(_serve_orb(bridge, config, console, demo=args.demo))

    # ── the full thing ───────────────────────────────────────────────
    lock = InstanceLock()
    if not lock.acquire():
        console.warn("already running", "an Atlas orb is open — use the tray, or Ctrl+Alt+A")
        return 1

    # A conversation may already be running in another process (that is how the
    # orb stays a viewer: kill the window, Atlas keeps talking).  If one is
    # advertised and still answers, put a face on *that* instead of starting a
    # second, mute bridge.
    attached = _live_endpoint(console) if not args.no_attach else None

    url = attached["url"] if attached else bridge.http_url()
    if args.opaque:
        url = f"{url}&opaque=1"

    hotkeys = HotkeyManager()
    tray = TrayIcon()

    def on_action(action_name: str) -> None:
        console.write(console.paint(f"  · {action_name}", DIM))
        if action_name == "quit":
            bridge.running = False

    hotkeys.on_action = on_action
    tray.on_action = on_action

    console.title("The orb")
    console.write(f"  orb:   {console.paint(url, DIM)}")
    console.write(f"  state: {console.paint('dormant', DIM)} · 30 fps cap · captions in DOM")

    window = OrbWindow(url, state=WindowState.load())
    opened, message = window.supervise()
    if not opened:
        console.warn("no window", message)
        console.write(
            console.paint(f"  → the orb is still served at {url.split('&')[0]} (open it in Edge)", DIM)
        )
    else:
        console.write(
            console.paint(f"  window: open ({window.state.width}×{window.state.height})", DIM)
        )
    if not hotkeys_available():
        console.write(console.paint(f"  · {hotkeys_missing()}", DIM))
    tray.start()
    hotkeys.start()

    stopper = threading.Event()
    backend = (
        _attached_backend(window, console, stopper)
        if attached
        else _bridge_backend(bridge, console, stopper)
    )

    try:
        if opened:
            # The GUI loop owns the main thread (that is a WebView2 rule, not a
            # style choice); the bridge or the conversation runs beside it.
            window.run(backend)
            console.write(console.paint("  window closed", DIM))
        elif attached:
            # A viewer with no window has nothing to do: the conversation is
            # somebody else's process, and it keeps running without us.
            console.write(
                console.paint("  → open the URL above in Edge, or install pywebview", DIM)
            )
        else:
            backend()
    except KeyboardInterrupt:
        pass
    finally:
        stopper.set()
        hotkeys.stop()
        tray.stop()
        window.save_position()
        lock.release()
    return 0


def _live_endpoint(console: Console, *, root: str = "data") -> dict | None:
    """A bridge advertised by another process — but only if it still answers."""
    from atlas_ui.bridge import probe_endpoint, read_endpoint

    found = read_endpoint(root)
    if not found:
        return None
    if probe_endpoint(str(found["url"])) is None:
        console.write(console.paint("  stale: data/ui-endpoint.json (no bridge there)", DIM))
        return None
    console.ok("attached", f"conversation already running (pid {found.get('pid', '?')})")
    return found


def _bridge_backend(bridge, console: Console, stopper: threading.Event) -> Callable[[], None]:
    """This process is the orb: serve the bridge, then wait for the window to go."""

    def backend() -> None:
        console.write(console.paint("  Ctrl+C to stop", DIM))
        try:
            asyncio.run(_serve_orb_process(bridge, console, stopper))
        except KeyboardInterrupt:  # pragma: no cover - interactive
            pass
        finally:
            console.write(console.paint(f"  sessions: {bridge.hub.session_count()}", DIM))

    return backend


def _attached_backend(window, console: Console, stopper: threading.Event) -> Callable[[], None]:
    """This process is only the face: the other process owns the conversation."""

    def backend() -> None:
        console.write(console.paint("  Ctrl+C (or closing the orb) stops the window only", DIM))
        while not stopper.is_set():
            stopper.wait(0.4)
        with contextlib.suppress(Exception):
            window.close()

    return backend


async def _serve_orb_process(bridge, console: Console, stopper: threading.Event | None = None) -> int:
    """Bridge up, serve until stopped — and always clean up after ourselves."""
    await bridge.start()
    try:
        await bridge.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as exc:  # pragma: no cover - uvicorn's own failures
        console.fail("bridge", str(exc))
        return 1
    finally:
        if stopper is not None:
            stopper.set()
        await bridge.stop()
    return 0


async def _serve_orb(bridge, config, console: Console, *, demo: bool = False) -> int:
    """Bridge only: no window, no tray — the orb in any browser on this machine."""
    console.title("Orb bridge")
    console.write(f"  open: {console.paint(bridge.http_url(), DIM)}")
    if bridge.host not in ("127.0.0.1", "localhost"):
        console.warn(
            "not loopback",
            f"the bridge is reachable on {bridge.host} — the token in the URL is the only gate",
        )
    console.write(console.paint("  token-gated · Ctrl+C to stop", DIM))
    await bridge.start()
    tasks = []
    if demo:
        console.write(console.paint("  demo: cycling every state and mood (no microphone)", DIM))
        tasks.append(asyncio.create_task(_demo_stream(bridge)))
    try:
        await bridge.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for task in tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await bridge.stop()
    del config
    console.write(f"  sessions: {bridge.hub.session_count()} · state: {bridge.hub.state}")
    return 0


async def _demo_stream(bridge, *, pause: float = 2.4) -> None:
    """Walk every state, mood and caption shape — LEVEL-05 step 10, automated.

    This is how the orb is checked without a microphone on the machine: the same
    messages the real loop publishes, in the same order a conversation produces
    them, so *"does the visual match the meaning"* is a question with an answer.
    Every state the orb can draw appears here except `muted` (a hotkey or the mic
    gate produces it) and `error` (a real failure produces it) — and a test fails
    if a new state is added without a line here.
    """
    from atlas_core.events import (
        AudioLevel,
        ConfirmationRequested,
        DegradedModeChanged,
        MoodChanged,
        ReplyFinished,
        SpeakerMatched,
        StateChanged,
        TokenDelta,
    )

    script: list[object] = [
        StateChanged(state="idle", previous="stopped"),
        MoodChanged(mood="calm", energy=0.4, warmth=0.7),
        StateChanged(state="waking", previous="idle"),
        StateChanged(state="listening", previous="waking"),
        AudioLevel(level=0.28),
        AudioLevel(level=0.41),
        SpeakerMatched(name="Badr", score=0.83, owner=True),
        StateChanged(state="followup", previous="listening"),
        StateChanged(state="listening", previous="followup"),
        TokenDelta(text="Salam, ana Atlas."),
        StateChanged(state="thinking", previous="listening"),
        MoodChanged(mood="focused", energy=0.6, warmth=0.5),
        TokenDelta(text=" Kifash n3awnek lyoum?"),
        ConfirmationRequested(question="nsedd l PC?", timeout_s=8.0),
        StateChanged(state="confirming", previous="thinking"),
        DegradedModeChanged(degraded=True, reason="no cloud key — lean mode", mode="lean"),
        DegradedModeChanged(degraded=False, reason="", mode="full"),
        StateChanged(state="speaking", previous="confirming"),
        ReplyFinished(text="Safi, l9it liya 3 chwiya: kayn tsera f 10:30.", language="ar-MA"),
        AudioLevel(level=0.55),
        AudioLevel(level=0.12),
        MoodChanged(mood="happy", energy=0.7, warmth=0.9),
        TokenDelta(
            text=(
                "Wah, hadi mzyana — nsifto lik daba. — a caption that is deliberately "
                "long enough to exercise the two-line clamp and the ellipsis at the front "
                "of the reply, so the card never grows into a dashboard."
            )
        ),
        StateChanged(state="stopped", previous="speaking"),
    ]
    index = 0
    while True:
        event = script[index % len(script)]
        index += 1
        # The bridge subscribes to the bus; publishing keeps one code path for
        # real turns and for this walkthrough (no second way to drive the orb).
        await bridge.bus.publish(event)  # type: ignore[arg-type]
        await asyncio.sleep(pause)


def _version() -> str:
    from atlas import __version__

    return __version__


def cmd_ui_protocol_alias(args: argparse.Namespace, console: Console) -> int:
    """`atlas ui-protocol` — the L0 command, kept working (one implementation)."""
    args.ui_action = "protocol"
    return cmd_ui(args, console)


def cmd_version(args: argparse.Namespace, console: Console) -> int:
    import platform

    import atlas_core

    console.write(
        f"atlas {__version__} · atlas-core {atlas_core.__version__} · python {platform.python_version()}"
    )
    return 0


# ── L2: the ears ─────────────────────────────────────────────────────
def _ear_parts(config: AppConfig, console: Console, *, use_silero: bool = True):
    """Build the whole listening chain, reporting what could not be built.

    One function, used by `listen`, `audio replay` and `bench_asr.py`'s sibling
    paths — so a machine that cannot do cloud ASR behaves the same everywhere.
    """
    from atlas_audio import (
        SileroVad,
        VadSegmenter,
        build_recognizers,
        build_wake_engine,
    )
    from atlas_audio.vad import SegmenterConfig as VadConfig

    built = build_recognizers(config)
    segmenter = VadSegmenter(VadConfig.from_config(config))
    notes: list[str] = []
    if use_silero and not segmenter.use_silero(SileroVad()):
        notes.append("VAD: energy backend (sherpa-onnx Silero not installed)")
    wake = build_wake_engine(config)
    if wake.name == "energy-wake":
        notes.append("wake: energy backend — a demo, not a keyword spotter")
    return built, segmenter, wake, notes


def cmd_audio(args: argparse.Namespace, console: Console) -> int:
    """Device plumbing, the microphone self-test, and replaying a recording."""
    from atlas_audio import check_audio_stack, list_devices
    from atlas_audio.capture import platform_notes, self_test

    action = args.audio_action

    if action == "devices":
        report = check_audio_stack()
        console.title("Audio devices")
        console.table(
            ("state", "detail"),
            [
                ("sounddevice", "✓ installed" if report.sounddevice_available else "✗ missing"),
                ("numpy", "✓ installed" if report.numpy_available else "✗ missing"),
                ("inputs", str(len(report.inputs))),
                ("outputs", str(len(report.outputs))),
                ("host APIs", ", ".join(report.host_apis) or "—"),
            ],
        )
        inputs, outputs, _host_apis = list_devices()
        for device in inputs + outputs:
            console.write(f"  {device.label()}")
        for problem in report.problems:
            console.warn("problem", problem)
        if not report.ok:
            console.write(f"  {console.paint('→ pip install atlas-audio[audio]', DIM)}")
        console.write()
        for note in platform_notes():
            console.write(console.paint(f"  · {note}", DIM))
        return 0 if report.ok else 1

    if action == "test":
        result = self_test(seconds=args.seconds)
        console.title("Microphone self-test")
        if result.ok:
            console.ok("round trip", result.summary())
            console.write(f"  {console.paint('→ now try: atlas listen --status', DIM)}")
            return 0
        console.fail("round trip", result.summary())
        return 1

    if action == "replay":
        return _replay(args, console)

    console.fail("audio", f"unknown action {action!r}")
    return 1


def _replay(args: argparse.Namespace, console: Console) -> int:
    """Run a recording back through the whole chain — no microphone, no cloud."""
    config = _load(args.config, console)
    if config is None:
        return 1

    from atlas_audio import VoiceLoop, WavFile, load_session
    from atlas_audio.loop import LoopConfig as VoiceLoopConfig

    target = Path(args.path).expanduser()
    if not target.exists():
        console.fail("replay", f"no such file or directory: {target}")
        return 1

    frames = None
    turns_recorded: list[dict] = []
    if target.is_dir() or target.suffix == ".json":
        session = load_session(target)
        if session.audio is None:
            console.fail("replay", "this session has no session.wav (record with --capture-dump)")
            return 1
        frames = session.audio.frames()
        turns_recorded = session.turns
    else:
        frames = WavFile.read(target).frames()

    built, segmenter, wake, notes = _ear_parts(config, console, use_silero=not args.no_silero)
    processor = built["processor"].__class__.from_config(config)
    loop = VoiceLoop(
        wake=wake,
        segmenter=segmenter,
        recognizer=built["factory"],
        processor=processor,
        config=VoiceLoopConfig.from_config(config, followup_ms=args.followup_ms),
    )
    loop.arm()

    if not built["factory"].candidates("ar-MA"):
        console.fail("no ASR engine", "add a cloud key to .env, or install 'atlas-audio[local]'")
        console.write(f"  {console.paint('→ atlas providers  shows which brains have keys', DIM)}")
        return 1

    console.title("Replay")
    console.write(f"  {console.paint(f'{len(frames)} frames · {Path(target).name}', DIM)}")
    for note in notes:
        console.write(console.paint(f"  · {note}", DIM))
    console.write()

    turns = asyncio.run(loop.feed_frames(frames, source="replay"))
    if not turns:
        console.warn("nothing transcribed", "was the wake word in the recording?")
        return 1
    for turn in turns:
        console.write(f"  [{turn.language} {turn.transcript.confidence:.2f} {turn.transcript.engine}] {turn.text}")
    if turns_recorded:
        console.write()
        console.write(console.paint("  recorded vs replayed:", DIM))
        for expected, turn in zip(turns_recorded, turns, strict=False):
            mark = "✓" if expected.get("transcript", "") == turn.text else "≠"
            console.write(f"   {mark} {expected.get('transcript', '')!r} → {turn.text!r}")
    return 0


def _voice_engine(config, args) -> Any:
    """The CLI's one place for "which voice did the user ask for?"."""
    return getattr(args, "tts_engine", "") or None


def cmd_say(args: argparse.Namespace, console: Console) -> int:
    """Speak a line — the L3 mouth, without the ears.

    `--out FILE.wav` writes the audio instead of playing it, which is how a
    Darija sentence gets to a Moroccan friend for a second opinion.
    """
    config = _load(args.config, console)
    if config is None:
        return 1

    import asyncio
    from array import array

    from atlas_audio import Mouth, build_synthesizer, voice_problems
    from atlas_audio.capture import pcm_to_wav_bytes
    from atlas_audio.frames import SAMPLE_RATE, resample_pcm

    text = args.text or ""
    if text == "-":
        text = sys.stdin.read().strip()
    if not text:
        console.fail("nothing to say", "pass the text, or - to read it from stdin")
        return 1

    language = args.language or config.app.language
    playing = not args.out
    if playing and not args.engine and not args.force:
        from atlas_audio.playback import SoundDeviceWriter

        if not SoundDeviceWriter.available():
            console.warn(
                "no output device",
                "sounddevice is not available — use --out FILE.wav, or pip install 'atlas-audio[audio]'",
            )
            return 1

    for note in voice_problems(config, language=language):
        console.warn("voice", note)

    mouth = Mouth.from_config(
        config,
        language=language,
        voice=args.voice or "",
        engine=args.engine or None,
        playing=playing,
    )
    if mouth.cache is not None and args.no_cache:
        mouth.streamer.cache = None

    started = time.monotonic()
    sentences = asyncio.run(mouth.say(text, language=_language_tag(language), play=playing))
    total_ms = (time.monotonic() - started) * 1000

    console.title("Say")
    engine = build_synthesizer(config, language=language, engine=args.engine or None)
    console.write(f"  engine: {console.paint(engine.name if engine else 'none', DIM)}")
    for sentence in sentences:
        cost = "cached" if sentence.cached else f"{sentence.synth_ms:.0f} ms"
        label = f"[{sentence.prosody.label} · {sentence.engine} · {cost}]"
        console.write(f"  {console.paint(label, DIM)}  {sentence.text}")
    console.write()
    if not any(sentence.audible for sentence in sentences):
        console.fail(
            "no voice",
            "every engine in the chain failed — run `atlas voice` for the install lines",
        )
        mouth.close()
        return 1
    if args.out:
        pcm = bytearray()
        rate = SAMPLE_RATE
        for sentence in sentences:
            if not sentence.pcm:
                continue
            samples = array("h")
            samples.frombytes(sentence.pcm)
            if sentence.sample_rate != SAMPLE_RATE:
                samples = resample_pcm(samples, source_rate=sentence.sample_rate)
            pcm.extend(samples.tobytes())
            rate = SAMPLE_RATE
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pcm_to_wav_bytes(bytes(pcm), sample_rate=rate))
        console.ok("written", f"{target} · {len(pcm) / 2 / rate:.1f}s")
    stats = mouth.stats()
    first = console.paint(f"{stats['first_audio_ms']:.0f} ms", DIM)
    console.write(
        f"  first audio {first} · total {total_ms:.0f} ms · "
        f"cache {stats['cache_hits']}/{stats['sentences']} · {stats['audio_s']}s spoken"
    )
    mouth.close()
    return 0


def cmd_voice(args: argparse.Namespace, console: Console) -> int:
    """The mouth: which engines exist, what they sound like, what is cached."""
    config = _load(args.config, console)
    if config is None:
        return 1

    from atlas_audio import TtsCache, build_synthesizer, tts_status, voice_problems

    action = args.voice_action or "status"

    if action == "warm":
        from atlas_audio import Mouth

        mouth = Mouth.from_config(config, language=config.app.language, playing=False)
        if mouth.cache is None:
            console.warn("no cache", "config.toml has no cache_path — nothing to warm")
            return 1
        warmed = asyncio.run(mouth.warm())
        console.title("Warming the voice cache")
        if not warmed and mouth.cache.stats().entries == 0:
            console.fail(
                "nothing warmed",
                "no engine could synthesise — run `atlas voice` for the install lines",
            )
            return 1
        console.ok(
            "warmed",
            f"{warmed} line(s) synthesised · cache now {mouth.cache.stats().entries} clips",
        )
        console.write(console.paint("  → `atlas voice cache` for the size and hit rate", DIM))
        return 0

    if action == "cache":
        cache = TtsCache(config.tts.cache_path, max_mb=config.tts.cache_max_mb)
        if args.clear:
            removed = cache.clear()
            console.ok("cache cleared", f"{removed} clips")
            return 0
        stats = cache.stats()
        console.title("TTS cache")
        console.write(f"  path:  {config.tts.cache_path}")
        console.write(f"  clips: {stats.entries} · {stats.bytes / 1e6:.1f} MB of {config.tts.cache_max_mb} MB")
        console.write(f"  hits:  {stats.hits} · misses {stats.misses} · hit rate {stats.hit_rate:.0%}")
        return 0

    if action == "test":
        return cmd_say(
            argparse.Namespace(
                config=args.config,
                text=args.text or "",
                language=args.language,
                voice=args.voice,
                engine=args.engine or None,
                out=args.out,
                no_cache=False,
                force=args.force,
            ),
            console,
        )

    console.title("Voices")
    console.table(("engine", "state"), tts_status(config))
    console.write()
    for language in ("ar-MA", "en-GB"):
        engine = build_synthesizer(config, language=language)
        console.write(
            f"  {language}: {console.paint(engine.name if engine else 'none — captions only', DIM)}"
        )
        for note in voice_problems(config, language=language):
            console.warn("fallback", note)
    console.write()
    console.write(console.paint("  try: atlas say \"salam, ana Atlas\" --language ar-MA --out /tmp/a.wav", DIM))
    return 0


def cmd_identity(args: argparse.Namespace, console: Console) -> int:
    """Who Atlas knows, who owns it, and what each voice may reach.

    Every action here is deliberately file-or-mic based: enrol from clips or from
    the microphone, verify a recording, read the scores.  Nothing in this command
    needs the cloud, and the profiles never leave `data/`.
    """
    config = _load(args.config, console)
    if config is None:
        return 1

    from atlas_audio import (
        EnrollmentSession,
        SherpaSpeakerVerifier,
        build_identity,
        speaker_status,
    )
    from atlas_audio.devices import list_devices
    from atlas_core.identity import ProfileRepository

    action = args.identity_action or "status"
    identity = build_identity(config)

    if action == "status":
        console.title("Identity")
        console.table(("piece", "state"), speaker_status(config))
        stats = identity["repository"].stats()
        console.write()
        console.write(f"  profiles: {console.paint(stats['path'], DIM)}")
        for name, profile in identity["repository"].load_all().items():
            role = "owner" if profile.owner else "known"
            console.write(
                f"    · {name} ({role}) · quality {profile.quality:.2f} · "
                f"{len(profile.embeddings)} samples · dim {profile.dimension}"
            )
        if not stats["people"]:
            console.warn(
                "nobody enrolled",
                "every voice is treated as a guest — run: atlas identity enrol --name <you>",
            )
        summary = identity["log"].summary()
        if summary["events"]:
            console.write()
            console.write(
                f"  speaker log: {summary['events']} utterances · "
                f"{summary['accept_rate']:.0%} accepted"
            )
            for name, row in dict(summary["speakers"]).items():
                console.write(console.paint(f"    · {name}: n={row['n']} p50={row['p50']} max={row['max']}", DIM))
        console.write()
        console.write(
            console.paint(
                "  · voice profiles are local (data/, gitignored) and are a convenience "
                "gate, not a cryptographic identity — destructive actions still confirm",
                DIM,
            )
        )
        return 0

    if action == "log":
        from atlas_core.identity import SpeakerLog

        speaker_log = SpeakerLog(config.identity.log_path)
        console.title("Speaker log")
        rows = [
            (
                time.strftime("%H:%M:%S", time.localtime(event.at)),
                event.speaker if event.accepted else f"≈{event.speaker or '?'}",
                f"{event.score:.2f}",
                "✓" if event.accepted else "·",
                event.reason,
            )
            for event in speaker_log.tail(args.limit)
        ]
        console.table(("time", "speaker", "score", "ok", "reason"), rows)
        return 0

    if action == "forget":
        repository: ProfileRepository = identity["repository"]
        if args.all:
            if not args.yes:
                console.warn("confirmation", "this wipes every voice profile — pass --yes")
                return 1
            removed = repository.wipe()
            console.ok("erased", f"{removed} voice profile(s) deleted")
            console.write(
                console.paint("  · the person notes in the vault are untouched", DIM)
            )
            return 0
        if not args.name:
            console.fail("no name", "atlas identity forget --name <who> (or --all --yes)")
            return 1
        if not repository.delete(args.name):
            console.warn("nothing to forget", f"no profile named {args.name!r}")
            return 1
        console.ok("erased", f"{args.name}'s voice profile is gone")
        return 0

    # ── enrol and verify need a verifier ─────────────────────────────
    if action == "enrol" and not (args.name or config.identity.owner_name or "").strip():
        # A missing name is the user's typo, not a missing model: say the useful
        # thing first, even on a machine where the model is not installed yet.
        console.fail("no name", "atlas identity enrol --name <who>")
        return 1
    verifier: SherpaSpeakerVerifier | None = identity["verifier"]
    if verifier is None:
        console.fail("identity disabled", "config.toml has [identity] enabled = false")
        return 1
    if not verifier.available():
        console.fail("no speaker model", verifier.missing())
        console.write(
            console.paint(
                f"  → expected at {config.identity.model_path}, or set it in [identity]",
                DIM,
            )
        )
        return 1

    if action == "verify":
        from atlas_audio import WavFile

        if not args.path:
            console.fail("no file", "atlas identity verify <file.wav> [--name who]")
            return 1
        samples = WavFile.read(args.path)
        profiles = identity["repository"].load_all()
        if not profiles:
            console.warn("nothing to compare", "enrol first: atlas identity enrol --name <you>")
            return 1
        embedding = verifier.embed_or_none(samples.samples)
        if embedding is None:
            console.fail("too short", "give me a couple of seconds of speech")
            return 1
        console.title("Verify")
        from atlas_core.identity import best_match

        for name, profile in profiles.items():
            score = best_match(embedding, profile)
            mark = "✓" if score >= config.identity.threshold else "·"
            console.write(
                f"  {mark} {name:<16} {score:.3f}  "
                f"{console.paint('owner' if profile.owner else 'known', DIM)}"
            )
        console.write()
        console.write(
            f"  threshold {config.identity.threshold:.2f} · "
            f"{console.paint(f'{len(samples.samples) / 16000:.1f}s of audio', DIM)}"
        )
        return 0

    if action == "enrol":
        name = (args.name or config.identity.owner_name or "").strip()
        repository = identity["repository"]
        existing = repository.load(name)
        session = EnrollmentSession(
            name,
            verifier=verifier,
            config=IdentityConfig.from_config(config),
            language=config.app.language,
        )
        console.title(f"Enrolling {name}")
        if existing:
            console.write(
                console.paint(
                    f"  · replacing {len(existing.embeddings)} old sample(s) "
                    f"(quality {existing.quality:.2f})",
                    DIM,
                )
            )

        if args.from_files:
            clips = [_read_clip(Path(path), console) for path in args.from_files]
            for clip in clips:
                if clip is None:
                    return 1
                step = session.add(clip)
                _print_step(console, step, session)
        else:
            from atlas_audio import FrameBus, FrameProducer  # noqa: F401 - same graph

            console.write(f"  · {session.required} samples × {config.identity.enrol_seconds:.0f}s")
            console.write(f"  · inputs: {len(list_devices()[0])}")
            for index in range(session.required):
                console.write()
                console.write(f"  {console.paint(session.prompt(index), DIM)}")
                try:
                    input("    press Enter, then read it aloud ▸ ")
                except EOFError:
                    console.fail("no terminal", "use --from FILE.wav for each sample")
                    return 1
                clip = _record_sample(config, console, args)
                if clip is None:
                    return 1
                step = session.add(clip)
                _print_step(console, step, session)

        problems = session.problems()
        if problems:
            console.write()
            for problem in problems:
                console.warn("not enrolled", problem)
            console.write(
                console.paint(
                    "  → quiet room, same microphone, speak normally — then try again", DIM
                )
            )
            return 1

        profile = session.profile()
        if profile is None:  # pragma: no cover - guarded by problems() above
            console.fail("not enrolled", "the samples did not clear the quality floor")
            return 1
        profile.samples = session.index
        identity["guard"].enroll(profile, owner=bool(args.owner or not repository.owner_name()))
        if not config.identity.owner_name:
            console.write(
                console.paint(
                    "  · tip: set owner_name in [identity] so the logs and greetings use it",
                    DIM,
                )
            )
        from atlas_obsidian.vault import VaultAdapter

        vault = VaultAdapter(config.obsidian)
        if vault.exists:
            path = vault.ensure_person(name, relationship="owner" if profile.owner else "")
            console.write(f"  · vault note: {path}")
        console.write()
        console.ok(
            "enrolled",
            f"{name} · quality {profile.quality:.2f} · "
            f"{len(profile.embeddings)} samples ({'owner' if profile.owner else 'known'})",
        )
        console.write(
            console.paint("  → now: atlas listen  (Atlas greets you by name once a day)", DIM)
        )
        return 0

    console.fail("unknown action", f"{action}: use status, enrol, verify, forget or log")
    return 1


def _read_clip(path: Path, console: Console):
    """One enrolment sample from a WAV file."""
    from atlas_audio import WavFile

    if not path.exists():
        console.fail("missing file", str(path))
        return None
    try:
        return WavFile.read(path).samples
    except (ValueError, OSError) as exc:
        console.fail("cannot read", f"{path}: {exc}")
        return None


def _record_sample(config, console: Console, args):
    """One enrolment sample from the microphone, trimmed of silence."""
    import asyncio

    from atlas_audio import FrameBus, FrameProducer
    from atlas_audio.frames import frames_to_pcm
    from atlas_audio.speaker import as_samples, trim_silence

    device = args.device or config.audio.input_device or None
    bus = FrameBus()
    producer = FrameProducer(bus, device=device)
    producer.start()
    console.write(console.paint(f"    ● recording {config.identity.enrol_seconds:.0f}s…", DIM))

    async def capture():
        frames = []
        stream = bus.stream()
        try:
            async for packet in stream:
                frames.append(packet.frame)
                if len(frames) * 80 >= config.identity.enrol_seconds * 1000:
                    break
        finally:
            await stream.aclose()
        return frames

    try:
        frames = asyncio.run(capture())
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.warn("stopped", "no sample recorded")
        return None
    finally:
        producer.stop()

    if not frames:
        console.fail("no audio", "the microphone produced nothing")
        return None
    pcm = frames_to_pcm(frames)
    samples = as_samples(pcm)
    return trim_silence(samples)


def _print_step(console: Console, step, session) -> None:
    """One line per enrolment sample: what happened and what is next."""
    if not step.ok:
        console.warn(f"sample {step.index + 1}", step.spoken or step.reason)
        return
    console.ok(
        f"sample {step.index + 1}",
        f"{step.speech_ms / 1000:.1f}s of speech · agreement {session.quality:.2f}",
    )


def _build_mouth(config, console: Console, loop, engine: str | None):
    """Build the mouth and wire its gate to the loop's mute.  `None` = captions.

    Returns `None` — with a warning, never an error — when there is no way to
    make a sound on this machine.  A deaf-and-mute first run is a valid first
    run; refusing to start is not.
    """
    from atlas_audio import MicGate, Mouth, voice_problems
    from atlas_audio.playback import SoundDeviceWriter

    if not SoundDeviceWriter.available():
        console.warn("voice off", "no audio output — captions only (pip install 'atlas-audio[audio]')")
        return None

    mouth = Mouth.from_config(
        config,
        language=config.app.language,
        engine=engine,
        gate=MicGate(on_close=loop.mute, on_open=loop.unmute),
        playing=True,
    )
    console.write(console.paint(f"  voice: {mouth.engine_name} · half duplex", DIM))
    for note in voice_problems(config, language=config.app.language):
        console.warn("voice", note)
    return mouth


def _build_identity_parts(config, console: Console):
    """Guard, verifier, store and log for one session — with the loud warnings.

    The two states a user must never be surprised by are stated at startup: nobody
    enrolled (so everyone is a guest) and a verifier that cannot load (so the
    identity gate is effectively off, and the log will say so every turn).
    """
    from atlas_audio import build_identity

    identity = build_identity(config)
    settings = identity["config"]
    if not settings.enabled:
        console.write(console.paint("  identity: off — single user, full access", DIM))
        return identity

    verifier = identity["verifier"]
    if verifier is not None and not verifier.available():
        console.warn("identity", verifier.missing())
        console.write(
            console.paint("  → until then Atlas treats every voice as you (logged each turn)", DIM)
        )
    profiles = identity["repository"].load_all()
    owner = identity["guard"].owner_name
    if profiles:
        names = ", ".join(f"{name}{' (owner)' if profile.owner else ''}" for name, profile in profiles.items())
        console.write(console.paint(f"  identity: {names} · threshold {settings.threshold:.2f}", DIM))
    else:
        console.warn(
            "nobody enrolled",
            "every voice is a guest — run: atlas identity enrol --name <you>",
        )
    if not owner:
        console.write(console.paint("  → the first person enrolled becomes the owner", DIM))
    return identity


def _memory_for(config, permissions):
    """The vault memory this speaker may see — empty for anyone who may not.

    `remember()` is the write side (L7); this is the read side, and it is gated by
    the same verdict the tool loop uses.  A guest gets their own note, never the
    owner's.
    """
    from atlas_obsidian.vault import VaultAdapter

    vault = VaultAdapter(config.obsidian)
    if not vault.exists:
        return ""
    if permissions.owner:
        return vault.read_memory()
    if permissions.subject:
        return vault.read_person(permissions.subject)
    return ""


def _language_tag(language: str) -> LanguageTag:
    """`ar-MA` / `en-GB` — the contract's own vocabulary, one conversion."""
    return "en-GB" if str(language).startswith("en") else "ar-MA"


def cmd_listen(args: argparse.Namespace, console: Console) -> int:
    """The real thing: wake word, endpointing, ASR, and the L1 brain."""
    config = _load(args.config, console)
    if config is None:
        return 1

    from atlas_audio import (
        TurnRecorder,
        WakeLog,
        build_recognizers,
        list_devices,
        wake_engine_status,
    )
    from atlas_audio.loop import LoopConfig as VoiceLoopConfig
    from atlas_audio.loop import VoiceLoop

    if args.status:
        console.title("Ears")
        rows = [(name, status) for name, status in wake_engine_status(config)]
        console.table(("wake engine", "state"), rows)
        built = build_recognizers(config)
        policy = built["factory"].policy
        console.write()
        console.write(
            f"  asr mode: {console.paint(policy.mode, DIM)} · "
            f"cloud keys: {', '.join(built['cloud'].backends) or 'none'}"
        )
        inputs, _outputs, _host_apis = list_devices()
        console.write(f"  input devices: {len(inputs)}")
        for device in inputs[:4]:
            console.write(console.paint(f"    · {device.label()}", DIM))
        log = WakeLog()
        console.write(f"  wake log: {len(log.hits)} hits · false-wake rate "
                      f"{log.false_wake_rate(hours=4):.2f}/h over the last 4 h")
        from atlas_audio import TtsCache, build_synthesizer, tts_status

        console.write()
        console.table(("voice", "state"), tts_status(config))
        chosen = build_synthesizer(config, language=config.app.language)
        console.write(
            f"  voice mode: {console.paint(chosen.name if chosen else 'captions only', DIM)}"
        )
        cache = TtsCache(config.tts.cache_path, max_mb=config.tts.cache_max_mb)
        stats = cache.stats()
        console.write(
            f"  tts cache: {stats.entries} clips · {stats.bytes / 1e6:.1f}/{config.tts.cache_max_mb} MB"
        )
        # L4 rows in the same view: "who is talking" is part of the ear report,
        # and the two states that need a decision (nobody enrolled, no model) are
        # exactly the ones a user would otherwise discover by being ignored.
        _build_identity_parts(config, console)
        return 0

    if args.replay:
        args.path = args.replay
        return _replay(args, console)

    built, segmenter, wake, notes = _ear_parts(config, console, use_silero=not args.no_silero)
    recorder = TurnRecorder("data/recordings", include_raw=args.capture_dump) if args.capture_dump else None

    # One bus for the ears, the brain and the face.  Built here even without
    # `--ui` (an idle bus costs nothing) so every state change has exactly one
    # destination and there is no second, UI-only code path to keep in sync.
    ui_bus = EventBus() if args.ui else None
    session, _router, _timings = _session(config, console, events=ui_bus)
    identity = _build_identity_parts(config, console)

    from atlas_audio import Mouth

    mouth: Mouth | None = None

    # The greeting hook runs on the loop's first verified owner turn of the day.
    greeter = DailyGreeter("data/last_greeting.txt")

    async def respond(text: str, language: str):
        permissions = loop.permissions
        audience = Audience.from_permissions(
            permissions, language=_language_tag(language)
        )
        # Once a day, and only for the verified owner: a greeting by name before
        # the answer.  It is spoken, not prepended to the prompt — the model does
        # not get to "remember" a greeting it did not receive.
        if permissions.owner and greeter.due(permissions.speaker):
            greeter.mark(permissions.speaker)
            line = identity["guard"].greeting(
                permissions.speaker, language=_language_tag(language)
            )
            if line:
                console.write(f"{console.paint('atlas ▸', CYAN)} {line}")
                if mouth is not None:
                    await mouth.say(line, language=_language_tag(language))
        # Two guards, deliberately independent: the prompt tells the model it is
        # talking to a guest, and the caller withholds the memory it may not see.
        memory = ""
        if permissions.allows(Capability.READ_MEMORY):
            memory = _memory_for(config, permissions)
        if permissions.restricted:
            console.write(console.paint(f"  [unknown voice {permissions.score:.2f}]", DIM))
        tokens = session.stream(text, memory=memory, audience=audience)
        if mouth is None:
            async for delta in tokens:
                yield delta
            return
        # The mouth speaks each sentence as it is completed, so the first word
        # arrives after one sentence is written, not after the whole answer.
        async for spoken in mouth.speak(
            tokens, language=_language_tag(language), mood=session.mood.view
        ):
            yield spoken
        if mouth.streamer.truncated:
            console.write(console.paint("  …(long answer stopped — say “kemmel” for the rest)", DIM))

    def say(delta: str) -> None:
        console.write(delta, end="")

    loop = VoiceLoop(
        wake=wake,
        segmenter=segmenter,
        recognizer=built["factory"],
        respond=respond,
        say=say,
        processor=built["processor"],
        recorder=recorder,
        wake_log=WakeLog(),
        config=VoiceLoopConfig.from_config(config, followup_ms=args.followup_ms, snappy=args.snappy),
        verifier=identity["verifier"],
        guard=identity["guard"],
        profiles=identity["repository"].load_all(),
        speaker_log=identity["log"],
        events=ui_bus,
    )

    # The mouth is built *after* the loop, because the gate it closes is the
    # loop's mute: the microphone must be shut before the first byte is played,
    # and reopened only after the tail.
    if not args.no_voice:
        mouth = _build_mouth(config, console, loop, _voice_engine(config, args))
    elif args.voice_engine:
        console.warn("voice ignored", "--no-voice was also passed")

    console.title("ATLAS")
    console.write(f"  {console.paint('say “atlas” and speak · Ctrl+C to stop', DIM)}")
    for note in notes:
        console.warn("degraded", note)
    if recorder is not None:
        console.write(console.paint(f"  recording every frame → {recorder.path}", DIM))
    console.write()

    if args.ui and args.ptt:
        console.warn("ui ignored", "--ptt runs a single push-to-talk turn")

    if args.ptt:
        return asyncio.run(_listen_ptt(loop, built, config, console, args, mouth=mouth))

    if args.ui and ui_bus is not None:
        return asyncio.run(
            _listen_with_ui(
                loop, built, config, console, args, bus=ui_bus, recorder=recorder, mouth=mouth
            )
        )

    return asyncio.run(
        _listen_live(loop, built, config, console, args, recorder=recorder, mouth=mouth)
    )


async def _listen_with_ui(
    loop, built, config, console: Console, args, *, bus, recorder, mouth=None
) -> int:
    """`atlas listen --ui`: the conversation, with the orb mirroring it.

    The bridge runs as a task in this same event loop — no thread, no queue, no
    second bus.  `data/ui-endpoint.json` is written for the first seconds of the
    run, so `atlas ui run` in another window attaches to *this* conversation
    instead of starting an idle bridge of its own.  If pywebview is missing, open
    the printed URL in Edge: exactly the same page.
    """
    from atlas_ui import EventBridge
    from atlas_ui.bridge import bridge_available, bridge_missing

    if not bridge_available():
        console.warn("ui off", bridge_missing())
        return await _listen_live(loop, built, config, console, args, recorder=recorder, mouth=mouth)

    if args.ui_port:
        args.port = args.ui_port
    bridge = EventBridge(host="127.0.0.1", port=args.ui_port or 8765, bus=bus, version=_version())
    await bridge.start()
    console.title("ATLAS · orb")
    console.write(f"  orb: {console.paint(bridge.http_url(), DIM)}")
    console.write(
        console.paint("  attach: atlas ui run   (same orb, its own process)", DIM)
    )
    serving = asyncio.create_task(_serve_quietly(bridge))
    try:
        return await _listen_live(loop, built, config, console, args, recorder=recorder, mouth=mouth)
    finally:
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving
        await bridge.stop()
        console.write(console.paint("  orb bridge stopped", DIM))


async def _serve_quietly(bridge) -> None:
    """uvicorn, without letting its own exit take the conversation down."""
    try:
        await bridge.serve()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - uvicorn's own failures
        log.warning("bridge_failed error=%s", exc)


async def _listen_live(loop, built, config, console: Console, args, *, recorder, mouth=None) -> int:
    """Microphone → loop, until Ctrl+C or --max-turns."""
    from atlas_audio import FrameBus, FrameProducer

    bus = FrameBus()
    producer = FrameProducer(bus, device=args.device or config.audio.input_device or None)

    def on_error(exc: Exception) -> None:
        console.fail("microphone", str(exc))

    producer.on_error = on_error
    producer.start()
    stream = bus.stream()
    try:
        async for packet in stream:
            if turn := await loop.feed(packet):
                console.write(f"{console.paint('atlas ▸', CYAN)} {turn.reply}")
                console.write()
                if args.max_turns and len(loop.turns) >= args.max_turns:
                    break
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        producer.stop()
        await stream.aclose()
        if recorder:
            path = recorder.flush()
            if path:
                console.write(console.paint(f"  session saved: {path.parent}", DIM))
        if hasattr(built["factory"], "aclose"):
            await built["factory"].aclose()
        if mouth is not None:
            mouth.close()
        console.write()
        console.ok("stopped", f"{len(loop.turns)} turns · {loop.wake_hits} wake hits")
    return 0


async def _listen_ptt(loop, built, config, console: Console, args, *, mouth=None) -> int:
    """Terminal push-to-talk: Enter starts a recording, Enter ends it."""
    from atlas_audio import FrameBus, FrameProducer
    from atlas_audio.ptt import HOTKEYS, HotkeyListener, PushToTalk

    bus = FrameBus()
    producer = FrameProducer(bus, device=args.device or config.audio.input_device or None)
    ptt = PushToTalk(loop)
    listener = HotkeyListener(backend=None) if args.hotkeys else None
    stop_flag = asyncio.Event()

    def request_stop() -> None:  # pragma: no cover - from a hotkey thread
        stop_flag.set()
        if mouth is not None:
            mouth.stop("hotkey")

    if listener is not None and not listener.start(on_ask=lambda: ptt.begin(), on_stop=request_stop):
        console.warn("hotkeys unavailable", "pip install 'atlas-audio[hotkeys]' — using Enter instead")
        console.write(console.paint(f"  keys would be: {listener.describe()} ({HOTKEYS['ask']})", DIM))

    producer.start()
    stream = bus.stream()

    async def consume() -> None:
        async for packet in stream:
            if ptt.recording:
                ptt.push(packet.frame)
            elif not listener and (turn := await loop.feed(packet)):
                console.write(f"  [{turn.language}] {turn.text}")

    consumer = asyncio.create_task(consume())
    console.write()
    try:
        while True:
            prompt = "press Enter to talk · q to quit ▸ "
            answer = await asyncio.to_thread(input, prompt)
            if answer.strip().lower() in {"q", "quit", "exit"}:
                break
            ptt.begin()
            console.write(console.paint("  ● recording — Enter to stop", DIM))
            await asyncio.to_thread(input)
            frames = ptt.end()
            console.write(console.paint(f"  {len(frames) * 80 / 1000:.1f}s captured", DIM))
            if turn := await ptt.finish():
                console.write(f"{console.paint('atlas ▸', CYAN)} {turn.reply}")
            else:
                console.warn("no utterance", "too short, or the VAD heard nothing")
            console.write()
    finally:
        consumer.cancel()
        producer.stop()
        await stream.aclose()
        if listener is not None:
            listener.stop()
        if mouth is not None:
            mouth.close()
        if hasattr(built["factory"], "aclose"):
            await built["factory"].aclose()
    return 0


def cmd_health(args: argparse.Namespace, console: Console) -> int:
    """Machine-readable doctor for scripts and the future UI."""
    from atlas.doctor import run_checks

    report = run_checks(config_path=args.config)
    payload = {
        "ok": not report.failures,
        "checks": [
            {"name": c.name, "status": c.status, "detail": c.detail, "hint": c.hint}
            for c in report.checks
        ],
    }
    console.write(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not report.failures else 1


# ── argument parsing ─────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="Atlas — a Darija-speaking assistant that lives in your PC.",
        epilog=USAGE_HINT,
    )
    parser.add_argument("--config", help="path to config.toml (or set ATLAS_CONFIG)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="is this machine ready for Atlas?")
    doctor.set_defaults(func=cmd_doctor)

    health = sub.add_parser("health", help="doctor as JSON (for scripts)")
    health.set_defaults(func=cmd_health)

    providers = sub.add_parser("providers", help="configured brains and their keys")
    providers.add_argument("--live", action="store_true", help="actually call each provider")
    providers.set_defaults(func=cmd_providers)

    chat = sub.add_parser("chat", help="talk to Atlas")
    chat.add_argument("--once", default="", metavar="TEXT", help="send one message and exit")
    chat.add_argument(
        "--structured",
        action="store_true",
        help="also get language/emotion metadata (costs time-to-first-token)",
    )
    chat.set_defaults(func=cmd_chat)

    vault = sub.add_parser("vault", help="the Obsidian second brain")
    vault.add_argument("vault_action", choices=["init", "status", "undo"])
    vault.add_argument("path", nargs="?", help="vault path (init) or existing vault")
    vault.add_argument("--vault", dest="vault_path", help="existing vault path")
    vault.add_argument("--template", help="path to vault-template/")
    vault.add_argument("--no-git", action="store_true", help="skip git history for the vault")
    vault.set_defaults(func=cmd_vault)

    skills = sub.add_parser("skills", help="what Atlas can do, and who may ask")
    skills.add_argument("skills_action", nargs="?", default="list", choices=["list", "audit"])
    skills.add_argument("--limit", type=int, default=20)
    skills.set_defaults(func=cmd_skills)

    ui = sub.add_parser("ui", help="the orb: state, mood and captions (L5)")
    ui.add_argument("ui_action", nargs="?", default="run", choices=["run", "serve", "status", "protocol"])
    ui.add_argument("--opaque", action="store_true", help="rounded card instead of a transparent window")
    ui.add_argument("--host", default="", help="bridge host (default 127.0.0.1; loopback unless you insist)")
    ui.add_argument("--port", type=int, default=8765, help="bridge port")
    ui.add_argument("--demo", action="store_true", help="serve + cycle every state and mood (no microphone)")
    ui.add_argument("--no-attach", action="store_true", help="ignore a running conversation")
    ui.add_argument("--out", help="ui protocol: write to a file instead of stdout")
    ui.set_defaults(func=cmd_ui)

    # Kept as an alias: the L0 README and the CI job both call it.
    ui_protocol = sub.add_parser("ui-protocol", help="generate the TypeScript UI contract")
    ui_protocol.add_argument("--out", help="write to a file instead of stdout")
    ui_protocol.set_defaults(func=cmd_ui_protocol_alias)

    version = sub.add_parser("version", help="versions of the pieces")
    version.set_defaults(func=cmd_version)

    identity = sub.add_parser("identity", help="who Atlas knows, and what each voice may do (L4)")
    identity.add_argument(
        "identity_action",
        nargs="?",
        default="status",
        choices=["status", "enrol", "verify", "forget", "log"],
    )
    identity.add_argument("path", nargs="?", help="WAV file for `verify`")
    identity.add_argument("--name", default="", help="whose voice (enrol / forget)")
    identity.add_argument("--owner", action="store_true", help="enrol as the owner")
    identity.add_argument(
        "--from",
        dest="from_files",
        action="append",
        default=[],
        metavar="FILE.wav",
        help="enrolment sample from a file (repeat for each sample)",
    )
    identity.add_argument("--device", help="input device for enrolment")
    identity.add_argument("--all", action="store_true", help="forget: every profile")
    identity.add_argument("--yes", action="store_true", help="forget --all: I mean it")
    identity.add_argument("--limit", type=int, default=20, help="log: how many lines")
    identity.set_defaults(func=cmd_identity)

    say = sub.add_parser("say", help="speak a line (L3) — no microphone involved")
    say.add_argument("text", nargs="?", default="", help="what to say, or - to read stdin")
    say.add_argument("--language", choices=["ar-MA", "en-GB"], help="which voice (default: config)")
    say.add_argument("--voice", default="", help="voice name override (Piper voice, SAPI name)")
    say.add_argument(
        "--engine",
        choices=["darija_tts_sidecar", "piper", "piper_arabic", "sapi"],
        help="force an engine (otherwise the chain decides)",
    )
    say.add_argument("--out", metavar="FILE.wav", help="write a 16 kHz WAV instead of playing")
    say.add_argument("--no-cache", action="store_true", help="bypass the TTS cache")
    say.add_argument("--force", action="store_true", help="try even if the output device looks missing")
    say.set_defaults(func=cmd_say)

    voice = sub.add_parser("voice", help="voices: status, test, cache")
    voice.add_argument(
        "voice_action",
        nargs="?",
        default="status",
        choices=["status", "test", "cache", "warm"],
    )
    voice.add_argument("text", nargs="?", default="", help="text for `voice test`")
    voice.add_argument("--language", choices=["ar-MA", "en-GB"], help="voice test language")
    voice.add_argument("--voice", default="", help="voice name override")
    voice.add_argument(
        "--engine",
        choices=["darija_tts_sidecar", "piper", "piper_arabic", "sapi"],
        help="force an engine",
    )
    voice.add_argument("--out", metavar="FILE.wav", help="voice test: write a WAV")
    voice.add_argument("--clear", action="store_true", help="cache: drop every clip")
    voice.add_argument("--force", action="store_true", help="voice test: ignore the device check")
    voice.set_defaults(func=cmd_voice)

    listen = sub.add_parser("listen", help="wake word + listening (L2)")
    listen.add_argument("--status", action="store_true", help="what the ears can do, then exit")
    listen.add_argument("--ptt", action="store_true", help="push-to-talk: press Enter to talk")
    listen.add_argument("--replay", metavar="PATH", help="play a WAV or capture dump instead of the mic")
    listen.add_argument("--capture-dump", action="store_true", help="record the whole session for replay")
    listen.add_argument("--device", help="input device name or index")
    listen.add_argument("--max-turns", type=int, default=0, help="stop after N turns (0 = forever)")
    listen.add_argument("--followup-ms", type=int, default=None, help="post-reply grace window")
    listen.add_argument("--snappy", action="store_true", help="end utterances after 350 ms of silence")
    listen.add_argument("--no-silero", action="store_true", help="force the energy VAD")
    listen.add_argument("--hotkeys", action="store_true", help="global hotkeys (needs the extra)")
    listen.add_argument(
        "--ui",
        action="store_true",
        help="serve the orb bridge for this conversation (attach with: atlas ui run)",
    )
    listen.add_argument("--ui-port", type=int, default=0, help="bridge port for --ui (default 8765)")
    listen.add_argument("--no-voice", action="store_true", help="captions only, no TTS")
    listen.add_argument(
        "--voice-engine",
        dest="tts_engine",
        choices=["darija_tts_sidecar", "piper", "piper_arabic", "sapi"],
        help="force a TTS engine for this session",
    )
    listen.set_defaults(func=cmd_listen)

    audio = sub.add_parser("audio", help="devices, microphone self-test, replay")
    audio.add_argument(
        "audio_action",
        nargs="?",
        default="devices",
        choices=["devices", "test", "replay"],
    )
    audio.add_argument("path", nargs="?", help="recording to replay (action: replay)")
    audio.add_argument("--seconds", type=float, default=1.0, help="self-test capture length")
    audio.add_argument("--no-silero", action="store_true", help="force the energy VAD")
    audio.add_argument("--followup-ms", type=int, default=None)
    audio.set_defaults(func=cmd_audio)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    # Logging is off by default: the CLI already reports what happened, in
    # Darija, under each reply.  -v is for when something needs explaining.
    from atlas_core.logging import setup_logging

    if getattr(args, "verbose", False):
        setup_logging(level="DEBUG", log_dir="logs")
    else:
        setup_logging(level="ERROR")
    try:
        return args.func(args, console)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.write()
        return 130
    except Exception as exc:
        console.fail("error", f"{type(exc).__name__}: {exc}")
        if getattr(args, "verbose", False):
            raise
        console.write(f"  {console.paint('→ rerun with -v for the traceback', DIM)}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
