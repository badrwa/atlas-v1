"""The mouth: engines that turn text into speech, and the policy that picks one.

Three engines, in the plan's order of preference:

* **`PiperSynthesizer`** — British English (`en_GB-alan-medium`) and, via a second
  voice, Arabic as the Darija bootstrap.  ONNX Runtime, no torch, ~10× realtime on
  two Skylake cores, 63 MB of model.  This is the workhorse.
* **`DarijaTtsSidecarSynthesizer`** — the real Darija voice, `DarijaTTS-v0.1-500M`
  running under llama.cpp in its **own virtualenv** (`vendor/darija-tts/`).  Atlas
  talks to it over HTTP on 127.0.0.1 so the 700 MB model and its dependencies
  never enter the core environment: the no-torch check stays honest, and a
  sidecar crash is a fallback instead of an outage.
* **`SapiSynthesizer`** — Windows' own `System.Speech`, last resort.  The voice is
  not Atlas's, but "Atlas can talk even with every model missing" beats silence —
  and it is what L5's UI uses for error announcements before Piper is fetched.

Every engine implements `VoiceSynthesizer.speak(text, voice=, language=, prosody=)`;
plain `synthesize()` is the contract-shaped wrapper so the rest of Atlas can treat
them as `SpeechSynthesizer`s and never learn about prosody.

Failure is normal and handled by `Mouth` (see `speech.py`): sidecar down → Arabic
Piper → SAPI → captions.  The plan's rule "a mid-answer engine switch must not
shorten the answer" is why the chain lives in one place (`build_synthesizer`)
rather than being re-derived by each caller.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_audio.capture import WavFile, pcm_to_wav_bytes
from atlas_audio.prosody import Prosody
from atlas_core.contracts import AudioChunk, LanguageTag, ResourceCost, SpeechSynthesizer
from atlas_core.engines import LoadedFlag
from atlas_core.errors import Unsupported

log = logging.getLogger(__name__)

#: Where the Darija sidecar lives.  Not a constant of the code but a default the
#: config can move, because a user may keep their models anywhere.
DEFAULT_SIDECAR_PYTHON = "vendor/darija-tts/.venv/Scripts/python.exe"
DEFAULT_SIDECAR_PORT = 8125


def http_health(url: str, *, timeout_s: float = 2.0) -> bool:
    """Is something answering `GET {url}/health`?  Never raises.

    Localhost only, by construction: the sidecar is a process on this machine,
    and opening a socket to it is cheap enough to do before every first request.
    """
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=timeout_s) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


class VoiceSynthesizer(SpeechSynthesizer):
    """A `SpeechSynthesizer` that also understands prosody.

    Adding an optional keyword-only parameter to a subclass is not a contract
    violation, but *relying* on it through the ABC would be — so `synthesize()`
    stays exactly as L0 declared it, and callers that know more ask for `speak()`.
    """

    language: LanguageTag = "ar-MA"
    #: True when the engine can actually use `Prosody.rate`; the sidecar can,
    #: SAPI can (through SSML-free rate presets), Piper can.
    supports_prosody = True

    def speak(
        self,
        text: str,
        *,
        voice: str = "",
        language: LanguageTag = "ar-MA",
        prosody: Prosody | None = None,
    ) -> Iterator[AudioChunk]:
        raise NotImplementedError

    def synthesize(
        self, text: str, *, voice: str = "", language: LanguageTag = "ar-MA"
    ) -> Iterator[AudioChunk]:
        return self.speak(text, voice=voice, language=language, prosody=None)


def is_voice(engine: Any) -> bool:
    """Does this synthesizer understand `speak()`?  One check, one place."""
    return isinstance(engine, VoiceSynthesizer)


# ── Piper ────────────────────────────────────────────────────────────
def _load_piper_voice(model_path: str, config_path: str | None = None) -> Any:
    """Load a Piper voice across the two package layouts in the wild.

    `piper-tts` 1.x exposes `piper.voice.PiperVoice`; 2.x moved it to the package
    root.  Both are the same ONNX graph and the same API, so supporting both is a
    two-line import dance — not two implementations.
    """
    voice_class: Any = None
    for module_name in ("piper", "piper.voice"):
        try:
            module = __import__(module_name, fromlist=["PiperVoice"])
        except ImportError:
            continue
        voice_class = getattr(module, "PiperVoice", None)
        if voice_class is not None:
            break
    if voice_class is None:
        raise Unsupported(
            "piper is not installed — run: pip install 'atlas-audio[local]'"
        )
    kwargs = {"config_path": config_path} if config_path else {}
    return voice_class.load(model_path, **kwargs)


def _piper_rate(voice: Any) -> int:
    """The voice's own sample rate (`22050` for most, `16000` for low-quality)."""
    rate = int(getattr(getattr(voice, "config", None), "sample_rate", 0) or 0)
    return rate or 22050


def _piper_audio(chunk: Any) -> tuple[bytes, int]:
    """Pull (pcm, rate) out of whatever shape this Piper version yields.

    Known shapes: an `AudioChunk` with `audio_int16_bytes` + `sample_rate`, a
    raw `bytes`, or a `(bytes, sample_rate)` tuple.  One adapter, because the
    version drift is real and the pipeline should not care.
    """
    for attribute in ("audio_int16_bytes", "audio"):
        data = getattr(chunk, attribute, None)
        if isinstance(data, bytes):
            rate = int(getattr(chunk, "sample_rate", 0) or 0)
            return data, rate
    if isinstance(chunk, bytes):
        return chunk, 0
    if isinstance(chunk, tuple) and len(chunk) == 2 and isinstance(chunk[0], bytes):
        return chunk[0], int(chunk[1])
    raise Unsupported(f"unrecognised Piper audio chunk: {type(chunk).__name__}")


class PiperSynthesizer(LoadedFlag, VoiceSynthesizer):
    """A local ONNX voice.  British English by default, Arabic on request."""

    name = "piper"
    ram_mb = 150
    cold_start_s = 0.5

    def __init__(
        self,
        model_path: str | Path = "models/tts/en_GB-alan-medium.onnx",
        *,
        voice: str = "en_GB-alan-medium",
        language: LanguageTag = "en-GB",
        config_path: str | Path | None = None,
        voice_loader: Any = None,
        ram_mb: int | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.config_path = str(config_path) if config_path else None
        self.voice = voice
        self.language = language
        # The lease keys engines by name; two Piper voices in one process must
        # not share it (that bug cost L2 an afternoon with the ASR models).
        self.name = f"piper:{voice}"
        self.ram_mb = ram_mb if ram_mb is not None else type(self).ram_mb
        self._voice_loader = voice_loader
        self._model: Any = None
        self.load_error = ""
        self.sample_rate = 22050

    # ── engine lifecycle ─────────────────────────────────────────────
    async def load(self) -> None:
        if self._model is not None:
            return
        try:
            loader = self._voice_loader or _load_piper_voice
            self._model = await asyncio.to_thread(loader, self.model_path, self.config_path)
            self.sample_rate = _piper_rate(self._model)
            self.load_error = ""
            log.info("piper_loaded voice=%s rate=%s", self.voice, self.sample_rate)
        except Exception as exc:
            self.load_error = str(exc)
            self._model = None
            log.warning("piper_load_failed voice=%s error=%s", self.voice, exc)
        self._loaded = self._model is not None

    async def unload(self) -> None:
        self._model = None
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._model is not None

    def cost_hint(self) -> ResourceCost:
        return ResourceCost(ram_mb=self.ram_mb, cold_start_s=type(self).cold_start_s)

    def available(self) -> bool:
        """Cheap precondition: is there a model file and a piper to load it?"""
        if self._model is not None:
            return True
        if self._voice_loader is None:
            try:
                import piper  # noqa: F401
            except ImportError:
                return False
        return Path(self.model_path).exists()

    def missing(self) -> str:
        """One line a user can act on — used by `atlas doctor` and the CLI."""
        if self.is_loaded() or self.available():
            return ""
        if not Path(self.model_path).exists():
            return (
                f"voice model not found at {self.model_path} — download "
                f"{self.voice}.onnx (+ .onnx.json) into {Path(self.model_path).parent}/"
            )
        return "piper is not installed — pip install 'atlas-audio[local]'"

    # ── speech ───────────────────────────────────────────────────────
    def speak(
        self,
        text: str,
        *,
        voice: str = "",
        language: LanguageTag = "ar-MA",
        prosody: Prosody | None = None,
    ) -> Iterator[AudioChunk]:
        if self._model is None:
            raise Unsupported(self.missing() or "piper voice is not loaded")
        plan = prosody or Prosody()
        kwargs = plan.for_piper() if self.supports_prosody else {}
        try:
            # Materialised *inside* the try on purpose: a generator rejects the
            # arguments when it is first iterated, not when it is called, and a
            # version check that only catches the eager case would miss exactly
            # the Piper builds this fallback exists for.
            chunks = list(self._model.synthesize(text, **kwargs))
        except TypeError:
            # Older Piper builds take no synthesis arguments.  Say so in the log
            # rather than silently dropping the mood, and still speak.
            if kwargs:
                log.info("piper_prosody_unsupported voice=%s", self.voice)
                self.supports_prosody = False
            chunks = list(self._model.synthesize(text))

        rate = self.sample_rate
        for chunk in chunks:
            pcm, chunk_rate = _piper_audio(chunk)
            if not pcm:
                continue
            rate = chunk_rate or rate
            yield AudioChunk(pcm=pcm, sample_rate=rate, final=False)
        yield AudioChunk(pcm=b"", sample_rate=rate, final=True)


# ── the Darija sidecar ───────────────────────────────────────────────
@dataclass
class SidecarConfig:
    """Where the sidecar is and how to start it — straight from `[tts]`."""

    url: str = f"http://127.0.0.1:{DEFAULT_SIDECAR_PORT}"
    python: str = DEFAULT_SIDECAR_PYTHON
    command: list[str] = field(default_factory=list)
    timeout_s: float = 30.0
    autostart: bool = True

    @classmethod
    def from_config(cls, config: Any = None) -> SidecarConfig:
        tts = getattr(config, "tts", None)
        if tts is None:
            return cls()
        command = list(getattr(tts, "sidecar_command", []) or [])
        return cls(
            url=str(getattr(tts, "sidecar_url", cls.url)),
            python=str(getattr(tts, "sidecar_python", cls.python)),
            command=command,
            timeout_s=float(getattr(tts, "sidecar_timeout_s", 30.0)),
            autostart=bool(getattr(tts, "sidecar_autostart", True)),
        )

    def argv(self) -> list[str]:
        """The command to run, with `{python}`/`{port}` filled in."""
        if self.command:
            return [part.format(python=self.python, port=os.environ.get("ATLAS_TTS_PORT", DEFAULT_SIDECAR_PORT)) for part in self.command]
        return [self.python, "-m", "darija_tts.server", "--port", str(DEFAULT_SIDECAR_PORT)]


class SidecarProcess:
    """Starts the isolated Darija voice, and knows when it is not there.

    Kept separate from the synthesizer so the *policy* (autostart, timeout,
    restart-once) is testable with a fake runner, and so a user who prefers to
    start the server by hand can turn autostart off.
    """

    def __init__(
        self,
        config: SidecarConfig | None = None,
        *,
        runner: Any = None,
        health: Any = None,
    ) -> None:
        self.config = config or SidecarConfig()
        self._runner = runner or subprocess.Popen
        self._health = health
        self.process: Any = None
        self.started_at = 0.0

    def running(self) -> bool:
        return self.process is not None and getattr(self.process, "poll", lambda: 1)() is None

    async def healthy(self) -> bool:
        probe = self._health or http_health
        try:
            return bool(await asyncio.to_thread(probe, self.config.url))
        except Exception:
            return False

    @staticmethod
    def _http_health(url: str) -> bool:
        """Default probe; injectable so tests never open a socket."""
        return http_health(url)

    async def ensure_started(self) -> bool:
        """Health check → start → poll health.  True if the voice is reachable."""
        if await self.healthy():
            return True
        if not self.config.autostart:
            log.info("sidecar_autostart_off url=%s", self.config.url)
            return False
        argv = self.config.argv()
        if not Path(argv[0]).exists() and shutil.which(argv[0]) is None:
            log.info("sidecar_python_missing path=%s", argv[0])
            return False
        log.info("sidecar_start argv=%s", " ".join(argv))
        try:
            self.process = self._runner(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path.cwd()),
            )
        except OSError as exc:
            log.warning("sidecar_spawn_failed error=%s", exc)
            return False
        self.started_at = time.monotonic()

        deadline = self.started_at + self.config.timeout_s
        while time.monotonic() < deadline:
            if await self.healthy():
                log.info(
                    "sidecar_ready cold_start_s=%.1f", time.monotonic() - self.started_at
                )
                return True
            if not self.running():
                log.warning("sidecar_exited code=%s", getattr(self.process, "returncode", "?"))
                return False
            await asyncio.sleep(0.25)
        log.warning("sidecar_timeout after=%.1fs", self.config.timeout_s)
        return False

    def stop(self) -> None:
        if self.process is None:
            return
        with suppress(OSError):  # already gone is fine
            self.process.terminate()
        self.process = None


class DarijaTtsSidecarSynthesizer(LoadedFlag, VoiceSynthesizer):
    """Moroccan Darija through a local HTTP sidecar (`vendor/darija-tts/`).

    The protocol is small on purpose: `GET /health` → 200, `POST /synthesize`
    with `{"text", "rate"}` → a WAV.  Anything more (voices, streaming, SSML)
    would be a protocol to maintain in two places; the sidecar answers sentences,
    and sentences are what the streamer sends.
    """

    name = "darija-tts"
    language: LanguageTag = "ar-MA"
    ram_mb = 700  # the model lives in the other process; this is the reservation
    cold_start_s = 8.0

    def __init__(
        self,
        config: SidecarConfig | None = None,
        *,
        sidecar: SidecarProcess | None = None,
        poster: Any = None,
        voice: str = "darija",
    ) -> None:
        self.config = config or SidecarConfig()
        self.sidecar = sidecar or SidecarProcess(self.config)
        self._poster = poster
        self.voice = voice
        self.load_error = ""
        self.last_ms = 0.0

    async def load(self) -> None:
        ok = await self.sidecar.ensure_started()
        self._loaded = ok
        self.load_error = "" if ok else f"sidecar not reachable at {self.config.url}"

    async def unload(self) -> None:
        self.sidecar.stop()
        self._loaded = False

    def is_loaded(self) -> bool:
        return self.sidecar.running() or self._loaded

    def cost_hint(self) -> ResourceCost:
        return ResourceCost(ram_mb=type(self).ram_mb, cold_start_s=type(self).cold_start_s)

    async def available(self) -> bool:
        return await self.sidecar.healthy()

    def speak(
        self,
        text: str,
        *,
        voice: str = "",
        language: LanguageTag = "ar-MA",
        prosody: Prosody | None = None,
    ) -> Iterator[AudioChunk]:
        plan = prosody or Prosody()
        poster = self._poster or self._http_post
        started = time.monotonic()
        data = poster(self.config.url, {"text": text, "voice": voice or self.voice, **plan.for_sidecar()})
        self.last_ms = (time.monotonic() - started) * 1000

        if not data:
            raise Unsupported(f"sidecar returned no audio for {len(text)} chars")
        wav = WavFile.read_bytes(data, label="sidecar")
        yield AudioChunk(pcm=wav.samples.tobytes(), sample_rate=wav.sample_rate, final=False)
        yield AudioChunk(pcm=b"", sample_rate=wav.sample_rate, final=True)

    def _http_post(self, url: str, payload: dict[str, Any]) -> bytes:
        import json

        request = urllib.request.Request(
            f"{url}/synthesize",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
            return bytes(response.read())


# ── Windows SAPI, the last resort ────────────────────────────────────
SAPI_SCRIPT = """
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
{select_voice}
$synth.Rate = {rate}
$synth.SetOutputToWaveFile('{out}')
$synth.Speak([System.IO.File]::ReadAllText('{text_file}', [System.Text.Encoding]::UTF8))
$synth.Dispose()
"""


class SapiSynthesizer(LoadedFlag, VoiceSynthesizer):
    """`System.Speech` on Windows — a voice Atlas did not choose, but a voice.

    The text goes through a **temp file**, never through the command line: it is
    user and model output, and quoting it into PowerShell would be an injection
    hole for the sake of a shortcut.
    """

    name = "sapi"
    language: LanguageTag = "en-GB"
    ram_mb = 40
    cold_start_s = 1.0
    supports_prosody = True

    def __init__(self, *, voice: str = "", runner: Any = None, platform: str | None = None) -> None:
        self.voice = voice
        self._runner = runner
        self._platform = platform or sys.platform
        self.load_error = ""

    async def load(self) -> None:
        self._loaded = self.available()
        if not self._loaded:
            self.load_error = "System.Speech is Windows-only"

    def is_loaded(self) -> bool:
        return self.available()

    def cost_hint(self) -> ResourceCost:
        return ResourceCost(ram_mb=type(self).ram_mb, cold_start_s=type(self).cold_start_s)

    def available(self) -> bool:
        if self._platform != "win32":
            return False
        return self._runner is not None or shutil.which("powershell") is not None

    def speak(
        self,
        text: str,
        *,
        voice: str = "",
        language: LanguageTag = "ar-MA",
        prosody: Prosody | None = None,
    ) -> Iterator[AudioChunk]:
        if not self.available():
            raise Unsupported("SAPI is only available on Windows")
        plan = prosody or Prosody()
        selected = voice or self.voice
        with tempfile.TemporaryDirectory(prefix="atlas-sapi-") as folder:
            text_file = Path(folder) / "text.txt"
            out_file = Path(folder) / "out.wav"
            text_file.write_text(text, encoding="utf-8")
            script = SAPI_SCRIPT.format(
                select_voice=(
                    f"$synth.SelectVoice('{selected}')" if selected else "# default voice"
                ),
                rate=_sapi_rate(plan.rate),
                out=out_file.as_posix(),
                text_file=text_file.as_posix(),
            )
            runner = self._runner or _run_powershell
            runner(script)
            if not out_file.exists():
                raise Unsupported("SAPI produced no WAV")
            wav = WavFile.read_bytes(out_file.read_bytes(), label="sapi")
        yield AudioChunk(pcm=wav.samples.tobytes(), sample_rate=wav.sample_rate, final=False)
        yield AudioChunk(pcm=b"", sample_rate=wav.sample_rate, final=True)


def _sapi_rate(rate: float) -> int:
    """SAPI's `Rate` is -10..10, not a multiplier.  Map it, clamp it."""
    return max(-10, min(10, round((rate - 1.0) * 20)))


def _run_powershell(script: str) -> None:
    subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        timeout=60,
    )


# ── the factory ──────────────────────────────────────────────────────
@dataclass(slots=True)
class VoiceConfig:
    """Which engine speaks which language, and with which voice."""

    english_engine: str = "piper"
    english_voice: str = "en_GB-alan-medium"
    darija_engine: str = "darija_tts_sidecar"
    darija_voice: str = "darija"
    models_dir: str = "models/tts"
    arabic_voice: str = "ar_JO-kareem-medium"
    sapi_voice: str = ""

    @classmethod
    def from_config(cls, config: Any = None) -> VoiceConfig:
        tts = getattr(config, "tts", None)
        if tts is None:
            return cls()
        return cls(
            english_engine=str(getattr(tts, "english_engine", "piper")),
            english_voice=str(getattr(tts, "english_voice", "en_GB-alan-medium")),
            darija_engine=str(getattr(tts, "darija_engine", "darija_tts_sidecar")),
            darija_voice=str(getattr(tts, "darija_voice", "darija")),
            models_dir=str(getattr(tts, "models_dir", "models/tts")),
            arabic_voice=str(getattr(tts, "arabic_voice", "ar_JO-kareem-medium")),
            sapi_voice=str(getattr(tts, "sapi_voice", "")),
        )

    def piper_path(self, voice: str) -> str:
        return str(Path(self.models_dir) / f"{voice}.onnx")

    def for_language(self, language: str) -> tuple[str, str]:
        """(engine, voice) for a language tag — the one language→voice mapping."""
        text = str(language)
        if text.startswith("en"):
            return self.english_engine, self.english_voice
        return self.darija_engine, self.darija_voice


def build_voice_chain(
    config: Any = None,
    *,
    language: str | None = None,
    engine: str | None = None,
    sidecar: SidecarProcess | None = None,
) -> list[VoiceSynthesizer]:
    """Every voice that could speak this language, best first.

    The order is a *policy*, so it is written once, here:

    * English — `piper` (the real British voice), then `sapi` (Windows' own).
    * Darija — the `darija_tts_sidecar` (the real Moroccan voice), then Piper with
      an Arabic voice (readable, wrong accent), then SAPI.
    * An `engine=` argument from the config or the CLI leads the chain, and its
      neighbours follow — so "use piper_arabic" still has a fallback behind it.
    """
    voices = VoiceConfig.from_config(config)
    language = str(language or getattr(getattr(config, "app", None), "language", "ar-MA"))
    english = language.startswith("en")
    primary = engine or voices.for_language(language)[0]

    order = ["piper", "sapi"] if english else ["darija_tts_sidecar", "piper_arabic", "sapi"]
    if primary not in order:
        order.insert(0, primary)
    else:
        order.remove(primary)
        order.insert(0, primary)

    def make(name: str) -> VoiceSynthesizer:
        if name in {"darija_tts_sidecar", "darija-tts"}:
            return DarijaTtsSidecarSynthesizer(
                SidecarConfig.from_config(config), sidecar=sidecar
            )
        if name == "piper_arabic":
            voice_name = voices.arabic_voice
            tag: LanguageTag = "ar-MA"
        elif name == "piper":
            voice_name = voices.english_voice if english else voices.arabic_voice
            tag = "en-GB" if english else "ar-MA"
        elif name == "sapi":
            return SapiSynthesizer(voice=voices.sapi_voice)
        else:
            raise ValueError(f"unknown tts engine: {name}")
        path = voices.piper_path(voice_name)
        return PiperSynthesizer(path, voice=voice_name, language=tag, config_path=path + ".json")

    return [make(name) for name in order]


def engine_ready(engine: VoiceSynthesizer, config: Any = None) -> bool:
    """Can this engine speak *right now*, without starting anything?

    The sidecar is the awkward one: it cannot be probed synchronously without
    opening a socket, so "ready" means "its interpreter is where the config says,
    or it is already answering".
    """
    if isinstance(engine, PiperSynthesizer):
        return engine.available()
    if isinstance(engine, SapiSynthesizer):
        return engine.available()
    if isinstance(engine, DarijaTtsSidecarSynthesizer):
        return Path(engine.config.python).exists() or http_health(engine.config.url)
    return True


def build_synthesizer(
    config: Any = None,
    *,
    language: str | None = None,
    engine: str | None = None,
    sidecar: SidecarProcess | None = None,
) -> VoiceSynthesizer | None:
    """The first voice in the chain that can actually speak — or `None`.

    `None` is a real answer, not an error: on a fresh Windows machine with no
    models downloaded, Atlas runs text-only and says so.  `Mouth` keeps the whole
    chain, so a *mid-answer* failure falls through to the next voice.
    """
    chain = build_voice_chain(config, language=language, engine=engine, sidecar=sidecar)
    for candidate in chain:
        if engine_ready(candidate, config):
            return candidate
        log.info("tts_not_ready engine=%s", candidate.name)
    return chain[0] if chain else None


def voice_problems(config: Any = None, *, language: str = "ar-MA") -> list[str]:
    """Why the chain might not lead with the voice the user expects.

    Not errors: a missing Darija sidecar is the normal state until L9 publishes
    the model, and the next voice in the chain speaks instead.  These strings are
    what `atlas voice` and `atlas listen` print so the *fallback* is never silent.
    """
    problems: list[str] = []
    for engine in build_voice_chain(config, language=language):
        if isinstance(engine, PiperSynthesizer) and not engine.available():
            problems.append(f"{engine.name} unavailable ({engine.missing()})")
        elif isinstance(engine, DarijaTtsSidecarSynthesizer):
            reachable = Path(engine.config.python).exists() or http_health(engine.config.url)
            if not reachable:
                problems.append(
                    f"{engine.name} not installed yet (no interpreter at {engine.config.python}, "
                    f"nothing answering {engine.config.url}) — see vendor/darija-tts/README.md"
                )
        elif isinstance(engine, SapiSynthesizer) and not engine.available():
            problems.append("sapi unavailable (Windows only)")
    return problems


def tts_status(config: Any = None) -> list[tuple[str, str]]:
    """One row per engine for `atlas doctor` / `atlas voice`."""
    voices = VoiceConfig.from_config(config)
    sidecar = SidecarProcess(SidecarConfig.from_config(config))
    rows: list[tuple[str, str]] = []
    for name in ("piper", "piper_arabic"):
        voice_name = voices.english_voice if name == "piper" else voices.arabic_voice
        engine = PiperSynthesizer(voices.piper_path(voice_name), voice=voice_name)
        state = "available" if engine.available() else engine.missing()
        rows.append((f"piper · {voice_name}", state))
    rows.append(
        (
            "darija-tts sidecar",
            f"{voices.darija_engine} · {sidecar.config.url} · "
            f"python {'found' if Path(sidecar.config.python).exists() else 'missing'}",
        )
    )
    rows.append(("sapi", "available (Windows last resort)" if SapiSynthesizer().available() else "not on this OS"))
    return rows


def wav_bytes(chunks: Iterator[AudioChunk]) -> tuple[bytes, int]:
    """Collect a synthesis into one WAV — for `atlas say --out file.wav`."""
    pcm = bytearray()
    rate = 16000
    for chunk in chunks:
        if chunk.pcm:
            pcm.extend(chunk.pcm)
            rate = chunk.sample_rate
    return pcm_to_wav_bytes(bytes(pcm), sample_rate=rate), rate


__all__ = [
    "DEFAULT_SIDECAR_PORT",
    "DarijaTtsSidecarSynthesizer",
    "PiperSynthesizer",
    "SapiSynthesizer",
    "SidecarConfig",
    "SidecarProcess",
    "VoiceConfig",
    "VoiceSynthesizer",
    "build_synthesizer",
    "is_voice",
    "tts_status",
    "voice_problems",
    "wav_bytes",
]
