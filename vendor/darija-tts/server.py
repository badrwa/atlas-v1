#!/usr/bin/env python3
"""The Darija voice sidecar: text in, WAV out, over localhost HTTP.

Why a separate process at all?  Because the only usable open-source Darija TTS
(`KandirResearch/DarijaTTS-v0.1-500M`, an OuteTTS-style model) needs an inference
stack Atlas refuses to put in its core: this venv may contain whatever the model
needs, the core venv stays torch-free (`scripts/check_no_torch.py` enforces it),
and a crash here is a fallback instead of an outage.

The protocol is deliberately two endpoints:

    GET  /health      -> 200 {"ok", "model", "loaded", "sample_rate"}
    POST /synthesize  -> 200 audio/wav   {"text", "voice", "rate"}

`rate` is applied by resampling, which moves pitch with speed.  On OuteTTS there
is no other knob, the prosody table only asks for 0.92–1.05, and lying about it
would be worse than saying it — pass `--honour-rate` to enable it.

Run it:

    python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
    .venv/Scripts/python server.py --port 8125

The standard-library-only rule applies to this file: it is the process Atlas has
to be able to start on a machine where *nothing* is installed yet, so its own
dependencies (outetts, llama-cpp-python) are imported lazily and their absence is
reported as 503 with the exact install line, never as a traceback to the user.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

log = logging.getLogger("darija-tts")

DEFAULT_MODEL = os.environ.get(
    "ATLAS_DARIJA_MODEL",
    "vendor/darija-tts/models/darija-tts-v0.1-500m.Q8_0.gguf",
)
SAMPLE_RATE = 24_000  # OuteTTS-0.2-500M speaks at 24 kHz


class Voice:
    """The model, loaded once, guarded by a lock (one request at a time).

    A single lock rather than a pool: this laptop has two cores, generation is
    CPU-bound, and two concurrent generations would make both sentences late
    instead of making one sentence on time.
    """

    def __init__(self, model_path: str, *, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self.interface: Any = None
        self.error = ""
        self.lock = threading.Lock()
        self.loaded_at = 0.0

    # ── loading ──────────────────────────────────────────────────────
    def load(self) -> bool:
        if self.interface is not None:
            return True
        try:
            import outetts  # type: ignore[import-not-found]
        except ImportError:
            self.error = (
                "outetts is not installed in this venv — "
                "pip install -r vendor/darija-tts/requirements.txt"
            )
            log.warning(self.error)
            return False

        if not Path(self.model_path).exists():
            self.error = (
                f"model not found at {self.model_path} — download the GGUF from "
                "https://huggingface.co/KandirResearch/DarijaTTS-v0.1-500M and point "
                "ATLAS_DARIJA_MODEL at it"
            )
            log.warning(self.error)
            return False

        try:
            started = time.monotonic()
            self.interface = _build_interface(outetts, self.model_path, self.device)
            self.loaded_at = time.monotonic() - started
            log.info("model loaded in %.1fs: %s", self.loaded_at, self.model_path)
            return True
        except Exception as exc:
            self.error = f"could not load the model: {exc}"
            log.exception("model load failed")
            self.interface = None
            return False

    def ready(self) -> bool:
        return self.interface is not None

    # ── synthesis ────────────────────────────────────────────────────
    def synth(self, text: str, *, rate: float, honour_rate: bool) -> bytes:
        with self.lock:
            if not self.load():
                raise RuntimeError(self.error)
            started = time.monotonic()
            samples, sample_rate = _generate(self.interface, text)
            compute_ms = (time.monotonic() - started) * 1000
            log.info(
                "synthesised %d chars in %.0f ms (%.1fx realtime)",
                len(text),
                compute_ms,
                (len(samples) / sample_rate) / max(compute_ms / 1000, 1e-6),
            )
        if honour_rate and abs(rate - 1.0) > 0.01:
            samples, sample_rate = _rate_shift(samples, sample_rate, rate)
        return _wav_bytes(samples, sample_rate)


def _build_interface(outetts: Any, model_path: str, device: str) -> Any:
    """Build the OuteTTS interface across the versions in the wild.

    OuteTTS changed its constructor between 0.1 and 0.3 (`ModelConfig.auto_config`
    gained arguments, `InterfaceConfig` came and went).  Trying the documented
    call first and falling back to the simplest one is cheaper than pinning a
    version that may not have a llama.cpp build for this machine.
    """
    attempts: list[Any] = []
    try:
        config = outetts.ModelConfig.auto_config(
            model_config=outetts.ModelConfigType.LLAMA_CPP,
            model_path=model_path,
            interface_version=outetts.InterfaceVersion.V3,
            backend=outetts.Backend.LLAMACPP,
        )
        attempts.append(lambda: outetts.Interface(config=config))
    except Exception as exc:
        log.debug("auto_config failed: %s", exc)
    attempts.append(lambda: outetts.Interface(model_path=model_path))
    attempts.append(lambda: outetts.Interface(model_path=model_path, device=device))

    errors: list[str] = []
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError("; ".join(errors) or "no usable outetts API")


def _generate(interface: Any, text: str) -> tuple[Any, int]:
    """Text → (samples, sample_rate) for whichever API this outetts has."""
    output = interface.generate(
        text=text,
        temperature=0.4,  # low: a voice should be repeatable, not creative
        repetition_penalty=1.1,
        max_length=4096,
    )
    if hasattr(output, "save"):
        with _TempWav() as path:
            output.save(str(path))
            return _read_wav(path)
    audio = getattr(output, "audio", None)
    if audio is not None:
        return audio, int(getattr(output, "sample_rate", SAMPLE_RATE) or SAMPLE_RATE)
    if isinstance(output, bytes):
        with _TempWav() as path:
            path.write_bytes(output)
            return _read_wav(path)
    raise RuntimeError(f"unrecognised generation output: {type(output).__name__}")


class _TempWav:
    """A temporary path that behaves like a Path and deletes itself."""

    def __init__(self) -> None:
        import tempfile

        self._folder = tempfile.TemporaryDirectory(prefix="darija-tts-")
        self.path = Path(self._folder.name) / "out.wav"

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, *exc: object) -> None:
        self._folder.cleanup()


def _read_wav(path: Path) -> tuple[Any, int]:
    import array

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if width != 2:
        raise RuntimeError(f"only 16-bit PCM is supported (got {width * 8}-bit)")
    samples = array.array("h")
    samples.frombytes(raw)
    if channels > 1:
        mono = array.array("h")
        for index in range(0, len(samples) - channels + 1, channels):
            mono.append(int(sum(samples[index + offset] for offset in range(channels)) / channels))
        samples = mono
    return samples, rate


def _rate_shift(samples: Any, sample_rate: int, rate: float) -> tuple[Any, int]:
    """`rate` by resampling — faster speech, higher pitch.  Documented, opt-in."""
    import array

    target = int(sample_rate * rate)
    shifted = array.array("h")
    position = 0.0
    step = target / sample_rate
    while int(position) < len(samples):
        shifted.append(samples[int(position)])
        position += step
    return shifted, target


def _wav_bytes(samples: Any, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


class Handler(BaseHTTPRequestHandler):
    voice: Voice
    honour_rate = False
    server_version = "atlas-darija-tts/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quieter default
        log.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        if self.path.rstrip("/") not in {"", "/health"}:
            self._json(404, {"error": "try /health or POST /synthesize"})
            return
        self._json(
            200,
            {
                "ok": True,
                "ready": self.voice.ready(),
                "model": self.voice.model_path,
                "loaded": self.voice.ready(),
                "load_s": round(self.voice.loaded_at, 1),
                "error": self.voice.error,
                "sample_rate": SAMPLE_RATE,
            },
        )

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/synthesize":
            self._json(404, {"error": "POST /synthesize"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            self._json(400, {"error": f"bad JSON: {exc}"})
            return

        text = str(payload.get("text", "")).strip()
        if not text:
            self._json(400, {"error": "no text"})
            return
        try:
            data = self.voice.synth(
                text,
                rate=float(payload.get("rate", 1.0) or 1.0),
                honour_rate=self.honour_rate,
            )
        except RuntimeError as exc:
            # Not installed / not downloaded: a 503 with the install line, which
            # is what `atlas voice` prints verbatim.
            self._json(503, {"error": str(exc)})
            return
        except Exception as exc:
            log.exception("synthesis failed")
            self._json(500, {"error": f"synthesis failed: {exc}"})
            return
        self._send(200, data, "audio/wav")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Atlas Darija TTS sidecar")
    parser.add_argument("--port", type=int, default=8125)
    parser.add_argument("--host", default="127.0.0.1", help="localhost only, by contract")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="path to the GGUF")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preload", action="store_true", help="load at startup, not on first text")
    parser.add_argument(
        "--honour-rate",
        action="store_true",
        help="apply the requested rate by resampling (speed changes pitch)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    if args.host not in {"127.0.0.1", "localhost"}:
        log.warning("--host %s: this server is meant to be reachable only by Atlas", args.host)

    handler = type("BoundHandler", (Handler,), {"honour_rate": args.honour_rate})
    handler.voice = Voice(args.model, device=args.device)
    if args.preload:
        handler.voice.load()

    server = ThreadingHTTPServer((args.host, args.port), handler)
    log.info("listening on http://%s:%s (model: %s)", args.host, args.port, args.model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
