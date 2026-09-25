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


def _session(config: AppConfig, console: Console):
    """Build a chat session wired to every provider that has a key."""
    from atlas_core.events import EventBus
    from atlas_core.timings import TimingRecorder
    from atlas_mind import ChatSession, ProviderRouter, build_providers
    from atlas_mind.router import QuotaTracker

    providers = build_providers(config)
    tracker = QuotaTracker(Path("data/quota.json"))
    events = EventBus()
    timings = TimingRecorder()
    router = ProviderRouter(
        providers,
        tracker=tracker,
        events=events,
        fallback_language=config.app.language,
    )
    session = ChatSession(config, router, timings=timings, events=events)
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
    rows = [
        (
            p.name,
            p.kind,
            p.model,
            "key ✓" if p.is_configured() else "no key",
            "on" if p.enabled else "off",
        )
        for p in config.providers
    ]
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
        return asyncio.run(_chat_loop(config, console, once=args.once))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.write()
        console.write("bslama 👋")
        return 0


async def _chat_loop(config: AppConfig, console: Console, *, once: str = "") -> int:
    from atlas_mind.chat import TurnResult

    session, router, timings = _session(config, console)

    console.title("ATLAS")
    console.write(
        console.paint(
            f"brain: {router.describe()} · language: {config.app.language} "
            f"(secondary {config.app.secondary_language})",
            DIM,
        )
    )
    console.write(console.paint("type /help for commands, /quit to leave", DIM))
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
    """One honest line under every reply: who answered, how fast, in what language."""
    if result.ttft_ms is None:
        console.write()
        return
    meta = (
        f"[{result.provider or 'canned'} · ttft {result.ttft_ms:.0f}ms · "
        f"total {result.total_ms:.0f}ms · {result.language}]"
    )
    console.write(console.paint("  " + meta, DIM))
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


def cmd_listen(args: argparse.Namespace, console: Console) -> int:
    console.title("Listening")
    console.warn("not yet", "the ear arrives in L2 (wake word, VAD, cloud ASR + local fallback)")
    console.write(f"  {console.paint('what works now:', DIM)} atlas chat")
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
    listen.set_defaults(func=cmd_listen)

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
