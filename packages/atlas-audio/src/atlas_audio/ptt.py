"""Push-to-talk: the escape hatch when a wake word is not appropriate.

Two reasons this exists rather than "just use the wake word":

* a noisy room where a false wake is likely, and a lecture where saying "atlas"
  out loud is not an option;
* the L2 gate itself — the first ten hands-free commands are so much easier to
  debug when a key press can prove the rest of the chain works.

Terminal PTT (press Enter, speak, press Enter) needs no dependency and works
everywhere.  The global hotkeys the plan asks for (Ctrl+Alt+Space / M / S) need
a keyboard hook, which is optional here: without it, the honest failure is one
sentence, not a crash.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from atlas_audio.frames import Frame
from atlas_audio.loop import Turn, VoiceLoop

log = logging.getLogger(__name__)

#: The plan's hotkeys.  Kept as data so the UI and the docs cannot disagree.
HOTKEYS = {
    "ask": "ctrl+alt+space",
    "mute": "ctrl+alt+m",
    "stop": "ctrl+alt+s",
}


class PushToTalk:
    """Buffers microphone frames between `begin()` and `end()`.

    The loop is *not* consuming during a push-to-talk press: the key press is
    the trigger, so a wake-word detection in the middle of it must not start a
    second turn.  Frames are buffered here and handed to `VoiceLoop.push_to_talk`
    in one go.
    """

    def __init__(self, loop: VoiceLoop, *, max_seconds: float = 60.0) -> None:
        self.loop = loop
        self.max_frames = int(max_seconds * 1000 / 80)
        self.recording = False
        self.frames: list[Frame] = []
        self.presses = 0

    def begin(self) -> None:
        self.recording = True
        self.frames = []
        self.presses += 1

    def end(self) -> list[Frame]:
        self.recording = False
        frames = self.frames
        self.frames = []
        return frames

    def push(self, frame: Frame) -> None:
        """Called for every captured frame; buffers only while recording."""
        if not self.recording:
            return
        if len(self.frames) < self.max_frames:
            self.frames.append(frame)
        elif len(self.frames) == self.max_frames:  # a stuck key must not eat RAM
            log.warning("ptt_capped frames=%s seconds=%.0f", len(self.frames), self.max_frames * 80 / 1000)

    async def finish(self) -> Turn | None:
        """Transcribe what was buffered and complete one turn."""
        frames = self.end()
        if not frames:
            return None
        return await self.loop.push_to_talk(frames)

    @property
    def seconds(self) -> float:
        return len(self.frames) * 80 / 1000


class HotkeyListener:
    """Global hotkeys, if a keyboard hook is available.

    `backend` is injected (a callable taking `(keys, callback)` and returning a
    stop function) so the terminal PTT, the tests and the future UI all drive
    the same class.  Without a backend, `available()` is False and the caller
    prints the one sentence that matters.
    """

    def __init__(self, *, keys: dict[str, str] | None = None, backend: Any = None) -> None:
        self.keys = dict(keys or HOTKEYS)
        self.backend = backend
        self._stop: Callable[[], None] | None = None
        self.bindings: list[str] = []

    @staticmethod
    def available() -> bool:
        try:
            import keyboard  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError:
            return False

    def start(self, *, on_ask: Callable[[], None], on_mute: Callable[[], None] | None = None,
              on_stop: Callable[[], None] | None = None) -> bool:
        backend = self.backend
        if backend is None:
            if not self.available():
                log.info("hotkeys_unavailable — pip install 'atlas-audio[hotkeys]'")
                return False
            import keyboard

            backend = lambda keys, callback: keyboard.add_hotkey(keys, callback)  # noqa: E731

        handlers = {"ask": on_ask, "mute": on_mute, "stop": on_stop}
        for action, keys in self.keys.items():
            handler = handlers.get(action)
            if handler is None:
                continue
            backend(keys, handler)
            self.bindings.append(keys)
        self._stop = getattr(backend, "stop", None)
        return True

    def stop(self) -> None:
        if self._stop:
            self._stop()
            self._stop = None

    def describe(self) -> str:
        return " · ".join(f"{action}: {keys}" for action, keys in self.keys.items())


__all__ = ["HOTKEYS", "HotkeyListener", "PushToTalk"]
