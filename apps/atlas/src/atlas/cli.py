"""Atlas CLI — `python -m atlas <command>`.

L0/L1 commands: doctor, providers, chat, vault, skills, ui-protocol, version.
`listen` (voice) arrives in L2; the command exists and says so honestly.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from atlas import console
from atlas.doctor import run_checks

REPO_ROOT = Path(__file__).resolve().parents[4]
VERSION = "0.1.0"


# ── helpers ──────────────────────────────────────────────────────────
def _config_path() -> Path:
    """ATLAS_CONFIG wins, then ./config.toml (so tests can use a sandbox config)."""
    import os

    return Path(os.environ.get("ATLAS_CONFIG") or (Path.cwd() / "config.toml"))


def _load_config():
    from atlas_core.config import ConfigError, load_config

    try:
        return load_config(_config_path()), None
    except ConfigError as exc:
        return None, str(exc)


def _build_router(config):
    from atlas_core.events import EventBus
    from atlas_core.fakes import FakeProvider
    from atlas_core.timings import TimingRecorder
    from atlas_mind.providers.factory import build_providers
    from atlas_mind.router import ProviderRouter, QuotaTracker

    providers = build_providers(config)
    if not providers:
        providers = [FakeProvider(name="offline")]
    events = EventBus()
    timings = TimingRecorder(config_path_timings())
    return ProviderRouter(
        providers,
        tracker=QuotaTracker(),
        events=events,
        fallback_language=config.app.language,
    ), timings


def config_path_timings() -> Path:
    directory = Path("data")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "timings.jsonl"


# ── commands ─────────────────────────────────────────────────────────
def cmd_doctor(args: argparse.Namespace) -> int:
    config, error = _load_config()
    report = run_checks(config)
    console.write(report.render())
    if error:
        console.write(console.fail(f"config error: {error}"))
        return 1
    return 1 if report.failures else 0


def cmd_providers(args: argparse.Namespace) -> int:
    config, error = _load_config()
    if error or config is None:
        console.write(console.fail(f"config error: {error}"))
        return 1

    rows = []
    for provider in config.providers:
        key = provider.api_key()
        rows.append(
            [
                provider.name,
                provider.kind,
                provider.model,
                "yes" if key else ("local" if provider.is_local else "MISSING"),
                "on" if provider.enabled else "off",
            ]
        )
    console.write(console.title("Configured providers"))
    console.write(console.table(["name", "kind", "model", "key", "enabled"], rows))

    usable = config.ordered_providers()
    console.write("")
    if not usable:
        console.write(console.fail("no usable provider — add a key to .env"))
        return 1
    console.write(console.ok("order: " + " → ".join(provider.name for provider in usable)))

    if args.live:
        return asyncio.run(_ping_providers(config))
    console.write(console.info("\nadd --live to actually call them (uses your free quota)"))
    return 0


async def _ping_providers(config) -> int:
    import time

    from atlas_mind.providers.factory import build_provider

    console.write("")
    console.write(console.title("Live check (one tiny prompt each)"))
    any_ok = False
    for provider_config in config.ordered_providers():
        provider = build_provider(provider_config)
        started = time.perf_counter()
        try:
            health = await provider.health()
            if not health.ok:
                console.write(console.warn(f"{provider.name:<14} health: {health.detail}"))
            text = ""
            from atlas_core.contracts import LlmRequest, Message, Role, TextDelta

            request = LlmRequest(
                messages=[Message(role=Role.USER, content="goul 'salam' f kelma wa7da")],
                max_output_tokens=16,
            )
            async for event in provider.stream(request):
                if isinstance(event, TextDelta):
                    text += event.text
                    if len(text) > 40:
                        break
            ttft = (time.perf_counter() - started) * 1000
            any_ok = True
            console.write(
                console.ok(f"{provider.name:<14} {ttft:6.0f} ms → {text.strip()[:40] or '(empty)'}")
            )
        except Exception as exc:
            console.write(console.fail(f"{provider.name:<14} {type(exc).__name__}: {str(exc)[:110]}"))
        finally:
            close = getattr(provider, "aclose", None)
            if close:
                await close()
    return 0 if any_ok else 1


def cmd_chat(args: argparse.Namespace) -> int:
    config, error = _load_config()
    if error or config is None:
        console.write(console.fail(f"config error: {error}"))
        return 1
    return asyncio.run(_chat_loop(config))


async def _chat_loop(config) -> int:
    from atlas_mind.chat import ChatSession

    router, timings = _build_router(config)
    session = ChatSession(config, router, timings=timings)

    console.write(console.title("ATLAS") + console.info(f"  · {router.describe()}"))
    console.write(
        console.info("speak Darija or English · /help for commands · /quit to leave\n")
    )

    while True:
        try:
            text = await asyncio.to_thread(input, console.paint("you ▸ ", console.CYAN))
        except (EOFError, KeyboardInterrupt):
            console.write("\n" + console.info("b slama 👋"))
            return 0

        text = text.strip()
        if not text:
            continue
        if text.startswith("/"):
            if _handle_command(text, session, timings) == "quit":
                return 0
            continue

        from atlas_mind.chat import TurnResult

        console.write(console.paint("atlas ▸ ", console.MAGENTA))
        result = TurnResult()
        async for delta in session.stream(text, _result=result):
            console.stream(delta)   # speak-while-thinking: print tokens as they arrive
        console.write("")
        if result.ttft_ms is not None:
            console.write(
                console.info(
                    f"        [{result.provider} · ttft {result.ttft_ms:.0f} ms · total {result.total_ms:.0f} ms · {result.language}]"
                )
            )


def _handle_command(text: str, session, timings) -> str | None:
    from atlas_mind.dialects import PACKS

    command, _, argument = text.partition(" ")
    if command in ("/quit", "/exit"):
        console.write(console.info("b slama 👋"))
        return "quit"
    if command == "/help":
        console.write(
            console.table(
                ["command", "what"],
                [
                    ["/help", "this list"],
                    ["/lang <ar-MA|en-GB>", "switch language explicitly"],
                    ["/reset", "forget the conversation window"],
                    ["/timing", "p50/p95 latency summary"],
                    ["/providers", "who is configured and who answered last"],
                    ["/context", "how the token budget was spent"],
                    ["/quit", "leave"],
                ],
            )
        )
    elif command == "/lang":
        target = argument.strip() or "ar-MA"
        if target not in PACKS:
            console.write(console.warn(f"unknown language {target!r}; use one of {', '.join(PACKS)}"))
        else:
            session.language.route("", hint=target)
            console.write(console.ok(f"language → {PACKS[target].label}"))
    elif command == "/reset":
        session.reset()
        console.write(console.ok("conversation reset"))
    elif command == "/timing":
        console.write(timings.format_summary())
    elif command == "/providers":
        console.write(console.info("order: " + session.router.describe()))
        console.write(console.info(f"last answered by: {session.router.last_provider or '(none yet)'}"))
        for name, quota in session.router.quotas().items():
            console.write(
                console.info(
                    f"  {name}: {quota.used_today} today"
                    + (f" / cap {quota.daily_cap}" if quota.daily_cap else "")
                )
            )
    elif command == "/context":
        console.write(console.info(str(session.context.last_build or "nothing built yet")))
    else:
        console.write(console.warn(f"unknown command {command!r} — try /help"))
    return None


def cmd_vault(args: argparse.Namespace) -> int:
    from atlas_core.config import ObsidianSection
    from atlas_core.errors import VaultError
    from atlas_obsidian import VaultAdapter

    config, _error = _load_config()

    if args.action == "init":
        if not args.path:
            console.write(console.fail("usage: python -m atlas vault init <path>"))
            return 1
        template = REPO_ROOT / "vault-template"
        adapter = VaultAdapter.init_from_template(args.path, template, git=not args.no_git)
        console.write(console.ok(f"vault ready at {adapter.root}"))
        console.write(console.info("add this to .env:  ATLAS_VAULT_PATH=" + str(adapter.root)))
        console.write(console.info("then open it in Obsidian as a vault"))
        return 0

    vault_path = (
        getattr(args, "vault_path", None)
        or args.path
        or (config.obsidian.vault_path if config else "")
    )
    if not vault_path:
        console.write(console.fail("no vault configured — python -m atlas vault init <path>"))
        return 1
    adapter = VaultAdapter(ObsidianSection(vault_path=str(vault_path)))

    if args.action == "status":
        try:
            stats = adapter.stats()
        except VaultError as exc:
            console.write(console.fail(str(exc)))
            return 1
        console.write(console.title("Vault"))
        console.write(console.table(["key", "value"], [[k, str(v)] for k, v in stats.items()]))
        recent = adapter.recent_notes(5)
        if recent:
            console.write("")
            console.write(console.info("recent: " + ", ".join(note.title for note in recent)))
        return 0

    if args.action == "undo":
        undone = adapter.undo_last_write()
        console.write(console.ok("reverted the last vault write") if undone else console.warn("nothing to undo"))
        return 0 if undone else 1

    if args.action == "log":
        for line in adapter.journal.log(limit=args.limit):
            console.write(console.info(line))
        return 0

    console.write(console.fail(f"unknown vault action {args.action!r}"))
    return 1


def cmd_skills(args: argparse.Namespace) -> int:
    from atlas_core.config import ObsidianSection
    from atlas_obsidian import VaultAdapter
    from atlas_skills import (
        OpenUrlSkill,
        RememberSkill,
        ShutdownSkill,
        SkillRegistry,
        SystemStatsSkill,
    )

    registry = SkillRegistry()
    if args.action == "audit":
        registry.audit_tail()
        console.write(console.info("(audit is populated while Atlas runs; empty in a fresh CLI)"))
        return 0

    vault = None
    config, _error = _load_config()
    if config and config.obsidian.vault_path:
        vault = VaultAdapter(ObsidianSection(vault_path=config.obsidian.vault_path))

    registry.register(SystemStatsSkill())
    registry.register(OpenUrlSkill())
    registry.register(ShutdownSkill())
    if vault is not None:
        registry.register(RememberSkill(vault))

    rows = [
        [skill.name, skill.permission.value, "owner only" if skill.owner_only else ""]
        for skill in sorted(registry._skills.values(), key=lambda s: s.name)
    ]
    console.write(console.title("Skills"))
    console.write(console.table(["skill", "permission", "restriction"], rows))
    console.write("")
    console.write(console.info(registry.describe("ar-MA")))
    return 0


def cmd_ui_protocol(_args: argparse.Namespace) -> int:
    from atlas_ui import typescript

    console.write(typescript())
    return 0


def cmd_listen(_args: argparse.Namespace) -> int:
    console.write(console.warn("voice listening arrives in L2 (wake word + Darija ASR)."))
    console.write(console.info("until then, use text: python -m atlas chat"))
    console.write(console.info("check the microphone now with: python -m atlas doctor"))
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    from atlas_core import __version__ as core_version

    console.write(f"atlas {VERSION} · atlas-core {core_version} · python {sys.version.split()[0]}")
    return 0


# ── argument parsing ─────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas",
        description="ATLAS — voice-first Darija/English assistant for this PC",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check hardware, config, keys, vault and audio").set_defaults(func=cmd_doctor)

    providers = sub.add_parser("providers", help="list configured LLM providers")
    providers.add_argument("--live", action="store_true", help="actually call each provider")
    providers.set_defaults(func=cmd_providers)

    sub.add_parser("chat", help="text conversation with Atlas (L1)").set_defaults(func=cmd_chat)

    vault = sub.add_parser("vault", help="Obsidian vault management")
    vault.add_argument("action", choices=["init", "status", "undo", "log"])
    vault.add_argument("path", nargs="?", help="vault path (for init)")
    vault.add_argument("--vault", dest="vault_path", help="vault path for status/log/undo")
    vault.add_argument("--no-git", action="store_true", help="skip git init (no undo support)")
    vault.add_argument("--limit", type=int, default=20)
    vault.set_defaults(func=cmd_vault)

    skills = sub.add_parser("skills", help="list skills and their permissions")
    skills.add_argument("action", nargs="?", choices=["list", "audit"], default="list")
    skills.set_defaults(func=cmd_skills)

    sub.add_parser("ui-protocol", help="print the generated orb protocol").set_defaults(func=cmd_ui_protocol)
    sub.add_parser("listen", help="voice mode (arrives in L2)").set_defaults(func=cmd_listen)
    sub.add_parser("version", help="print versions").set_defaults(func=cmd_version)
    parser.add_argument("--verbose", "-v", action="store_true", help="show debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from atlas_core.logging import setup_logging

    setup_logging("DEBUG" if getattr(args, "verbose", False) else "WARNING", log_dir=None)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        console.write("\n" + console.info("interrupted"))
        return 130


__all__ = ["build_parser", "main"]
