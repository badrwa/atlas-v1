"""Configuration: pydantic models, TOML + .env loading, profile overlays.

Precedence (lowest → highest):
    model defaults  <  config.toml  <  environment variables  <  explicit args

Secrets never live here: provider entries name the *environment variable* that
holds the key (``api_key_env``), and `.env` is gitignored.
"""

from __future__ import annotations

import logging
import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

from atlas_core.errors import ConfigError

log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("config.toml")


# ─────────────────────────────────────────────────────────────────────
#  Sections
# ─────────────────────────────────────────────────────────────────────
class AppSection(BaseModel):
    profile: Literal["lean", "bunker"] = "lean"
    language: str = "ar-MA"
    secondary_language: str = "en-GB"
    call_name: str = "صاحبي"


class AudioSection(BaseModel):
    input_device: str = ""
    output_device: str = ""
    sample_rate: int = 16_000
    frame_ms: int = 80
    pre_roll_ms: int = 1500

    @property
    def frame_samples(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)


class WakeSection(BaseModel):
    enabled: bool = True
    threshold: float = 0.60
    confirm_frames: int = 2
    keywords: list[str] = Field(default_factory=lambda: ["atlas"])


class VadSection(BaseModel):
    silence_ms: int = 500
    min_utterance_ms: int = 300
    max_utterance_ms: int = 12_000
    energy_threshold: float = 0.012
    #: How much quiet may precede the first syllable of an utterance. The
    #: pre-roll ring is 1.5 s so nothing is clipped; uploading all of it is not
    #: the same thing.
    lead_in_ms: int = 200


class AsrSection(BaseModel):
    mode: Literal["cloud_first", "local_first", "cloud_only", "local_only"] = "cloud_first"
    cloud_order: list[str] = Field(default_factory=lambda: ["gemini", "groq"])
    darija_model: str = "whisper-small-darija"
    english_model: str = "small.en"
    lease_ttl_s: int = 120
    #: The owner-editable correction list (L2). It is also the hotword prompt
    #: sent to cloud ASR, which is where most of the accuracy comes from.
    lexicon_path: str = "data/lexicon.md"
    #: The switch the orb shows a cloud glyph for. False = no audio leaves this
    #: machine, full stop: `build_recognizers` then refuses to use the cloud.
    cloud_audio: bool = True


class IdentitySection(BaseModel):
    """Voice identity (L4).  Defaults are the safe ones.

    `enabled = false` is a legitimate choice for a single-user machine: Atlas
    then behaves exactly like L3 (full capabilities, no verifier loaded).  With
    identity on and nobody enrolled, every voice is a guest — which is the secure
    reading, and `atlas listen` says so out loud at startup.
    """

    enabled: bool = True
    owner_name: str = ""
    #: Cosine threshold. Tune with `scripts/bench_speaker.py` on your own clips —
    #: the plan's rule is FAR ≈ 0 even if FRR rises: re-asking is cheaper than a leak.
    threshold: float = 0.65
    #: A very short utterance cannot carry an owner-only decision (the television
    #: says one word too). Short matches keep general capabilities.
    trust_min_ms: float = 1000.0
    model_path: str = "models/speaker/3dspeaker_speech_eres2net_base.onnx"
    profiles_path: str = "data/speakers.sqlite3"
    log_path: str = "data/speaker_log.jsonl"
    window: int = 6
    enrol_samples: int = 3
    enrol_seconds: float = 10.0
    enrol_min_speech_ms: float = 2500.0
    quality_floor: float = 0.55
    drift_rejections: int = 2
    greet_once_per_day: bool = True
    #: Guests may use SAFE PC skills (never CONFIRM: that needs the owner).
    guest_pc_control: bool = False


class TtsSection(BaseModel):
    """The mouth (L3).  Every field here is one the user may legitimately want.

    `darija_engine = "piper_arabic"` is the fully-offline setting (no sidecar,
    no extra process); `sapi` exists for the first run on a Windows box that has
    no models yet.
    """

    darija_engine: str = "darija_tts_sidecar"  # piper_arabic | darija_tts_sidecar | sapi
    english_engine: str = "piper"
    english_voice: str = "en_GB-alan-medium"
    darija_voice: str = "darija"
    #: Piper voices live here as `<voice>.onnx` + `<voice>.onnx.json`.
    models_dir: str = "models/tts"
    #: The offline Darija bootstrap: a Piper Arabic voice, wrong accent but local.
    arabic_voice: str = "ar_JO-kareem-medium"
    #: Optional Windows voice name for the last-resort engine.
    sapi_voice: str = ""
    cache_path: str = "data/tts-cache.sqlite3"
    cache_max_mb: int = 300
    realtime_streaming: bool = True
    #: Speak at most this many sentences before offering to continue.  A wall of
    #: speech is worse than a short answer; 0 disables the cap.
    max_sentences: int = 3
    ask_to_continue: bool = True
    prosody: bool = True
    quiet_hours: list[str] = Field(default_factory=lambda: ["22:30", "07:00"])
    gain: float = 1.0
    #: Voice barge-in needs echo cancellation this laptop does not have; the stop
    #: hotkey always works, and stop words work when this is on.
    barge_in: bool = False
    stop_words: list[str] = Field(default_factory=lambda: ["stop", "safi", "skut"])
    #: Seconds of audio held ahead of playback.  Two sentences is the RAM budget.
    prefetch: int = 2
    #: The Darija voice runs in its own venv; these describe that process.
    sidecar_url: str = "http://127.0.0.1:8125"
    sidecar_python: str = "vendor/darija-tts/.venv/Scripts/python.exe"
    sidecar_command: list[str] = Field(default_factory=list)
    sidecar_timeout_s: float = 30.0
    sidecar_autostart: bool = True


class DialectSection(BaseModel):
    code_switch: bool = True
    script: Literal["arabic", "arabizi"] = "arabic"


class MindSection(BaseModel):
    provider_order: list[str] = Field(
        default_factory=lambda: ["gemini", "groq", "openrouter", "together", "huggingface", "ollama", "llamacpp"]
    )
    temperature: float = 0.7
    max_output_tokens: int = 512
    request_timeout_s: float = 20.0
    max_tool_iterations: int = 3
    context_token_budget: int = 1200
    memory_token_budget: int = 400
    first_token_budget_ms: int = 1500
    dialect: DialectSection = Field(default_factory=DialectSection)


class ObsidianSection(BaseModel):
    vault_path: str = ""
    folder_inbox: str = "00_Inbox"
    folder_daily: str = "10_Daily"
    folder_notes: str = "20_Notes"
    folder_people: str = "30_People"
    folder_projects: str = "40_Projects"
    folder_atlas: str = "50_Atlas"
    folder_archive: str = "90_Archive"
    git_commit_writes: bool = True
    rest_api_url: str = "https://127.0.0.1:27124"
    watch_vault: bool = True

    def folder(self, name: str) -> str:
        return str(getattr(self, f"folder_{name}"))

    @property
    def has_vault(self) -> bool:
        return bool(self.vault_path)


class DialogueSection(BaseModel):
    """Turn-taking timings — how long Atlas waits, and what it does about it."""

    #: After a reply, this much grace before the wake word is needed again.
    followup_ms: int = 2500
    #: Frames dropped right after the wake word (the keyword's own tail).
    wake_guard_ms: int = 160
    #: A short tone when the wake word fires — the cheapest feedback there is.
    chime: bool = True


class GovernorSection(BaseModel):
    ram_floor_mb: int = 400
    idle_lease_ttl_s: int = 120
    watch_interval_s: int = 5
    degrade_below_mb: int = 500


class UiSection(BaseModel):
    enabled: bool = False
    always_on_top: bool = True
    orb_size_px: int = 220
    fps_cap: int = 30
    cloud_glyph: bool = True


class SkillsSection(BaseModel):
    dry_run: bool = False
    workspace_dir: str = ""


class ProviderConfig(BaseModel):
    """One LLM provider. `kind` selects the implementation; everything else is data."""

    name: str
    # Adding a provider kind means adding it here *and* registering it in
    # atlas_mind.providers.factory — two deliberate edits, never a typo that
    # silently falls back to the wrong wire protocol.
    kind: Literal["openai_compatible", "gemini", "llama_cpp"] = "openai_compatible"
    model: str = ""
    api_key_env: str = ""
    base_url: str = ""
    enabled: bool = True
    timeout_s: float = 20.0
    max_output_tokens: int = 512
    daily_request_cap: int | None = None
    supports_tools: bool = False
    temperature: float | None = None
    extra_headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _slug(cls, value: str) -> str:
        if not value or not value.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"provider name must be a slug, got {value!r}")
        return value.lower()

    @property
    def is_local(self) -> bool:
        return any(host in self.base_url for host in ("127.0.0.1", "localhost"))

    def api_key(self, env: Mapping[str, str] | None = None) -> str:
        if not self.api_key_env:
            return ""
        return (env or os.environ).get(self.api_key_env, "").strip()

    def is_configured(self, env: Mapping[str, str] | None = None) -> bool:
        return bool(self.api_key(env)) or self.is_local


# ─────────────────────────────────────────────────────────────────────
#  Root config
# ─────────────────────────────────────────────────────────────────────
class AppConfig(BaseModel):
    app: AppSection = Field(default_factory=AppSection)
    audio: AudioSection = Field(default_factory=AudioSection)
    wake: WakeSection = Field(default_factory=WakeSection)
    vad: VadSection = Field(default_factory=VadSection)
    asr: AsrSection = Field(default_factory=AsrSection)
    tts: TtsSection = Field(default_factory=TtsSection)
    mind: MindSection = Field(default_factory=MindSection)
    obsidian: ObsidianSection = Field(default_factory=ObsidianSection)
    governor: GovernorSection = Field(default_factory=GovernorSection)
    dialogue: DialogueSection = Field(default_factory=DialogueSection)
    identity: IdentitySection = Field(default_factory=IdentitySection)
    ui: UiSection = Field(default_factory=UiSection)
    skills: SkillsSection = Field(default_factory=SkillsSection)
    providers: list[ProviderConfig] = Field(default_factory=list)

    # ── derived views ────────────────────────────────────────────────
    def provider(self, name: str) -> ProviderConfig:
        for provider in self.providers:
            if provider.name == name:
                return provider
        raise ConfigError(f"unknown provider {name!r}")

    def ordered_providers(
        self, env: Mapping[str, str] | None = None
    ) -> list[ProviderConfig]:
        """Enabled providers that have a key (or are local), in preferred order."""
        order = {name: idx for idx, name in enumerate(self.mind.provider_order)}
        usable = [
            provider
            for provider in self.providers
            if provider.enabled
            and provider.is_configured(env)
            and provider.model
            and provider.base_url
        ]
        return sorted(usable, key=lambda p: order.get(p.name, len(order)))

    def with_env(self, env: Mapping[str, str] | None = None) -> AppConfig:
        """Apply ATLAS_* environment overrides (e.g. ATLAS_VAULT_PATH)."""
        resolved: Mapping[str, str] = env if env is not None else os.environ
        env = resolved
        data = self.model_dump()
        if vault := env.get("ATLAS_VAULT_PATH", "").strip():
            data["obsidian"]["vault_path"] = vault
        if profile := env.get("ATLAS_PROFILE", "").strip():
            if profile not in {"lean", "bunker"}:
                raise ConfigError(f"ATLAS_PROFILE must be lean|bunker, got {profile!r}")
            data["app"]["profile"] = profile
        return AppConfig.model_validate(data)


# ─────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────
def load_config(
    path: str | Path | None = None,
    *,
    env_file: str | Path | None = ".env",
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load config.toml (+ .env for secrets) and apply env overrides."""
    if env_file and Path(env_file).exists():
        load_dotenv(env_file, override=False)

    resolved = (
        Path(path)
        if path is not None
        else Path(os.environ.get("ATLAS_CONFIG", str(DEFAULT_CONFIG_PATH)))
    )
    if not resolved.exists():
        log.warning("config_not_found path=%s — using built-in defaults", resolved)
        return AppConfig().with_env(env)

    try:
        with resolved.open("rb") as handle:
            raw: dict[str, Any] = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{resolved}: invalid TOML — {exc}") from exc

    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"{resolved}: {exc}") from exc
    return config.with_env(env)


__all__ = [
    "AppConfig",
    "AppSection",
    "AsrSection",
    "AudioSection",
    "DialectSection",
    "GovernorSection",
    "MindSection",
    "ObsidianSection",
    "ProviderConfig",
    "SkillsSection",
    "TtsSection",
    "UiSection",
    "VadSection",
    "WakeSection",
    "load_config",
]
