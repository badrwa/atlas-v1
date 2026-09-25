"""The `atlas` command.

Everything here is wiring: config in, package out, result printed.  If a command
grows logic worth testing, it moves into a package — which is why `doctor.py`
lives next door and this file mostly parses arguments and formats output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from atlas import __version__
from atlas.console import CYAN, DIM, Console
from atlas_core.config import AppConfig, ConfigError, load_config

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


def _session(config: AppConfig, console: Console, *, structured: bool = False):
    """Build a chat session wired to every provider that has a key."""
    from atlas_core.events import EventBus
    from atlas_core.timings import TimingRecorder
    from atlas_mind import ChatSession, MoodEngine, ProviderRouter, build_providers
    from atlas_mind.router import QuotaTracker

    providers = build_providers(config)
    tracker = QuotaTracker(Path("data/quota.json"))
    events = EventBus()
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


def cmd_ui_protocol(args: argparse.Namespace, console: Console) -> int:
    from atlas_ui import typescript

    typescript_text = typescript()
    if args.out:
        Path(args.out).write_text(typescript_text, encoding="utf-8")
        console.ok("written", args.out)
        return 0
    console.write(typescript_text)
    return 0


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
        return 0

    if args.replay:
        args.path = args.replay
        return _replay(args, console)

    built, segmenter, wake, notes = _ear_parts(config, console, use_silero=not args.no_silero)
    recorder = TurnRecorder("data/recordings", include_raw=args.capture_dump) if args.capture_dump else None

    session, _router, _timings = _session(config, console)

    async def respond(text: str, language: str):
        async for delta in session.stream(text):
            yield delta

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
    )

    console.title("ATLAS")
    console.write(f"  {console.paint('say “atlas” and speak · Ctrl+C to stop', DIM)}")
    for note in notes:
        console.warn("degraded", note)
    if recorder is not None:
        console.write(console.paint(f"  recording every frame → {recorder.path}", DIM))
    console.write()

    if args.ptt:
        return asyncio.run(_listen_ptt(loop, built, config, console, args))

    return asyncio.run(
        _listen_live(loop, built, config, console, args, recorder=recorder)
    )


async def _listen_live(loop, built, config, console: Console, args, *, recorder) -> int:
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
        console.write()
        console.ok("stopped", f"{len(loop.turns)} turns · {loop.wake_hits} wake hits")
    return 0


async def _listen_ptt(loop, built, config, console: Console, args) -> int:
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

    ui = sub.add_parser("ui-protocol", help="generate the TypeScript UI contract")
    ui.add_argument("--out", help="write to a file instead of stdout")
    ui.set_defaults(func=cmd_ui_protocol)

    version = sub.add_parser("version", help="versions of the pieces")
    version.set_defaults(func=cmd_version)

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
