"""The orb window: pywebview/WebView2, a tray icon, and four global hotkeys.

Three things this module has to get right on the target machine, and one it has
to survive:

* **geometry that survives a restart and a DPI change.**  `WindowState` stores
  position/size in `%LOCALAPPDATA%\\atlas\\ui.json` and clamps them against the
  real screen bounds, because a frameless window at 150 % scaling is the classic
  way to end up with an invisible orb one inch off-screen.
* **one instance.**  A second `atlas ui` focuses the first one instead of opening
  a twin (a lock file plus a pid check; stale locks are stolen).
* **failure isolation.**  WebView2 missing, or the process dying, must leave the
  ears and the brain running: `OrbWindow.supervise()` restarts up to three times
  and then *says so once*, per LEVEL-05 step 8.
* **optional extras stay optional.**  Every third-party import is lazy, and each
  adapter answers `available()`/`missing()`, so `atlas ui` on a machine without
  pywebview prints an install line instead of a traceback.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: LEVEL-05 geometry: a floating orb, expandable to a caption panel.
ORB_SIZE = (220, 220)
PANEL_SIZE = (420, 520)
MIN_SIZE = (180, 180)
#: A second launch focuses the first; a third restart of a crashing window stops.
RESTART_BUDGET = 3

HOTKEYS: dict[str, str] = {
    "mute": "ctrl+alt+m",
    "ptt": "ctrl+alt+space",
    "stop": "ctrl+alt+s",
    "toggle_orb": "ctrl+alt+a",
}

#: The tray menu, as data — the adapter below is the only Windows-specific part.
TRAY_MENU: tuple[tuple[str, str], ...] = (
    ("state", "Show Atlas"),
    ("mute", "Mute / unmute"),
    ("restart_audio", "Restart audio"),
    ("vault", "Open vault"),
    ("quit", "Quit Atlas"),
)


def windows() -> bool:
    return sys.platform.startswith("win")


def ui_state_dir() -> Path:
    """`%LOCALAPPDATA%\\atlas` on Windows, `data/` everywhere else (tests, CI)."""
    base = os.environ.get("LOCALAPPDATA", "")
    if windows() and base:
        return Path(base) / "atlas"
    return Path("data")


def _module_present(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def window_available() -> bool:
    """True when pywebview *and* a usable webview backend are importable."""
    if not _module_present("webview"):
        return False
    try:  # pywebview itself imports the platform module lazily
        import webview  # noqa: F401

        return True
    except Exception:  # pragma: no cover - platform-specific failures
        return False


def window_missing() -> str:
    if not _module_present("webview"):
        return "pywebview is not installed — pip install 'atlas-ui[orb]' (needs WebView2 on Windows 10)"
    return "pywebview is installed but no backend started — on Windows, install the WebView2 runtime"


def tray_available() -> bool:
    return _module_present("pystray") and _module_present("PIL")


def tray_missing() -> str:
    if not _module_present("pystray"):
        return "pystray is not installed — pip install 'atlas-ui[tray]'"
    if not _module_present("PIL"):
        return "Pillow is not installed — pip install 'atlas-ui[tray]'"
    return ""


def hotkeys_available() -> bool:
    return _module_present("keyboard")


def hotkeys_missing() -> str:
    return "the 'keyboard' package is not installed — pip install 'atlas-ui[hotkeys]' (Ctrl+Alt+S still works in-window)"


def hotkey_bindings() -> list[str]:
    return [f"{combo} → {action}" for action, combo in HOTKEYS.items()]


# ── geometry ─────────────────────────────────────────────────────────
@dataclass(slots=True)
class WindowState:
    """Position, size and visibility of the orb, with the clamping rules.

    `clamp` is the DPI guard: at 150 % scaling the stored coordinates can describe
    a window that is entirely off-screen, which looks exactly like a crash to the
    person looking for it.
    """

    x: int = 40
    y: int = 40
    width: int = ORB_SIZE[0]
    height: int = ORB_SIZE[1]
    visible: bool = True
    expanded: bool = False
    on_top: bool = True
    path: Path = field(default_factory=lambda: ui_state_dir() / "ui.json")

    def as_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "visible": self.visible,
            "expanded": self.expanded,
            "on_top": self.on_top,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object], *, path: Path | None = None) -> WindowState:
        state = cls(path=path or (ui_state_dir() / "ui.json"))
        for key in ("x", "y", "width", "height"):
            if key in data:
                try:
                    setattr(state, key, int(str(data[key])))
                except (TypeError, ValueError):
                    log.warning("ui_state_bad_value key=%s value=%r", key, data[key])
        for key in ("visible", "expanded", "on_top"):
            if key in data:
                setattr(state, key, bool(data[key]))
        return state

    def clamp(self, *, screen: tuple[int, int] = (1920, 1080), min_size: tuple[int, int] = MIN_SIZE) -> WindowState:
        """Keep the window on screen and big enough to grab."""
        width, height = screen
        self.width = max(min_size[0], min(self.width, width))
        self.height = max(min_size[1], min(self.height, height))
        self.x = max(0, min(self.x, width - self.width))
        self.y = max(0, min(self.y, height - self.height))
        return self

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        return self.path

    @classmethod
    def load(cls, *, path: Path | None = None, screen: tuple[int, int] = (1920, 1080)) -> WindowState:
        target = path or (ui_state_dir() / "ui.json")
        if not target.is_file():
            return cls(path=target).clamp(screen=screen)
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # a corrupt file must not brick the orb
            log.warning("ui_state_unreadable path=%s error=%s", target, exc)
            return cls(path=target).clamp(screen=screen)
        if not isinstance(data, dict):
            return cls(path=target).clamp(screen=screen)
        return cls.from_dict(data, path=target).clamp(screen=screen)

    def expand(self) -> WindowState:
        self.expanded = not self.expanded
        self.width, self.height = PANEL_SIZE if self.expanded else ORB_SIZE
        return self


# ── one instance ─────────────────────────────────────────────────────
@dataclass(slots=True)
class InstanceLock:
    """A pid file that keeps `atlas ui` single-instance without a mutex."""

    path: Path = field(default_factory=lambda: ui_state_dir() / "ui.lock")

    def _alive(self, pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except (OSError, ValueError):
            return False
        return True

    def holder(self) -> int:
        """The pid of the running orb, or 0."""
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
        except (OSError, ValueError, json.JSONDecodeError, AttributeError):
            return 0
        if self._alive(pid) and pid != os.getpid():
            return pid
        return 0

    def acquire(self) -> bool:
        """True when this process may open the orb (stale locks are stolen)."""
        if self.holder():
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        return True

    def release(self) -> None:
        try:
            if self.path.is_file():
                self.path.unlink()
        except OSError:  # pragma: no cover - Windows file-in-use
            log.debug("ui_lock_release_failed path=%s", self.path)


# ── the window ───────────────────────────────────────────────────────
class OrbWindow:
    """pywebview around the orb page, with a restart budget and a real failure path."""

    def __init__(
        self,
        url: str,
        *,
        state: WindowState | None = None,
        title: str = "Atlas",
        on_closed: Callable[[], None] | None = None,
    ) -> None:
        self.url = url
        self.state = state or WindowState.load()
        self.title = title
        self.on_closed = on_closed
        self.restarts = 0
        self.window: Any = None
        self._webview: Any = None

    def open(self) -> bool:
        """Open (or focus) the window.  False means "no window, keep running"."""
        if not window_available():
            log.warning("ui_window_unavailable reason=%s", window_missing())
            return False
        try:
            import webview

            self._webview = webview
            self.window = webview.create_window(
                self.title,
                self.url,
                width=self.state.width,
                height=self.state.height,
                x=self.state.x,
                y=self.state.y,
                frameless=True,
                easy_drag=True,
                transparent=True,
                on_top=self.state.on_top,
                resizable=True,
                min_size=MIN_SIZE,
            )
            return True
        except Exception as exc:  # pragma: no cover - needs WebView2
            log.warning("ui_window_failed error=%s", exc)
            self.window = None
            return False

    def start(self, *, background: bool = True) -> bool:  # pragma: no cover - needs a display
        """Hand control to the GUI loop (`background=False` blocks)."""
        if self.window is None or self._webview is None:
            return False
        try:
            self._webview.start(gui="edgechromium", debug=False, private_mode=False)
            return True
        except Exception as exc:
            log.warning("ui_loop_failed error=%s", exc)
            return False

    def run(
        self, backend: Callable[[], None] | None = None
    ) -> bool:  # pragma: no cover - needs a display
        """Own the main thread until the orb closes; run `backend` beside it.

        This shape is WebView2's requirement, not a preference: the GUI message
        loop must be on the main thread, so the bridge (or the whole voice loop)
        runs in pywebview's own worker thread instead.  It also means a bridge
        crash cannot take the window down, and closing the window cannot take
        Atlas down — the other half keeps running until asked to stop.
        """
        if self.window is None or self._webview is None:
            return False
        try:
            if backend is None:
                self._webview.start(gui="edgechromium", debug=False, private_mode=False)
            else:
                self._webview.start(backend, gui="edgechromium", debug=False, private_mode=False)
            return True
        except Exception as exc:
            log.warning("ui_loop_failed error=%s", exc)
            return False

    def supervise(self) -> tuple[bool, str]:
        """Open the window, retrying a *crash* up to `RESTART_BUDGET` times.

        Returns `(alive, message)`; the message is meant to be said **once**, in
        plain words, because "UI crashed" that nobody hears is the same as a
        silent failure.
        """
        while self.restarts <= RESTART_BUDGET:
            if self.open():
                return True, ""
            self.restarts += 1
            log.warning("ui_restart attempt=%d budget=%d", self.restarts, RESTART_BUDGET)
        if not window_available():
            return False, window_missing()
        return False, (
            f"the orb window failed {RESTART_BUDGET} times — "
            "running without it; say 'restart interface' or run: atlas ui"
        )

    def save_position(self) -> Path:
        """Persist where the window ended up (called on close)."""
        window = self.window
        for attr, field_name in (("x", "x"), ("y", "y"), ("width", "width"), ("height", "height")):
            try:
                value = getattr(window, attr, None)
            except Exception:  # pragma: no cover - window already destroyed
                continue
            if isinstance(value, int):
                setattr(self.state, field_name, value)
        return self.state.clamp().save()

    def close(self):  # pragma: no cover - needs a display
        try:
            if self.window is not None:
                self.window.destroy()
        except Exception as exc:
            log.debug("ui_window_close_failed error=%s", exc)


# ── tray ─────────────────────────────────────────────────────────────
class TrayIcon:
    """pystray adapter.  The menu is data (`TRAY_MENU`); this only wires clicks.

    The tray must work while the orb is hidden — that is the whole reason it
    exists, so it never depends on `OrbWindow`.
    """

    def __init__(self, *, title: str = "Atlas", on_action: Callable[[str], None] | None = None) -> None:
        self.title = title
        self.on_action = on_action
        self.icon: Any = None

    def menu_items(self) -> list[tuple[str, str]]:
        return list(TRAY_MENU)

    def start(self) -> bool:  # pragma: no cover - needs a desktop session
        if not tray_available():
            log.info("ui_tray_unavailable reason=%s", tray_missing())
            return False
        try:
            import pystray
            from PIL import Image, ImageDraw

            image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            ImageDraw.Draw(image).ellipse((8, 8, 56, 56), fill=(20, 160, 150, 255))

            def make(item_id: str):
                def handler(icon, item):
                    callback = self.on_action
                    if callable(callback):
                        callback(item_id)

                return handler

            items = [
                pystray.MenuItem(label, make(item_id)) for item_id, label in TRAY_MENU
            ]
            self.icon = pystray.Icon("atlas", image, self.title, pystray.Menu(*items))
            self.icon.run_detached()
            return True
        except Exception as exc:
            log.warning("ui_tray_failed error=%s", exc)
            return False

    def stop(self) -> None:  # pragma: no cover - needs a desktop session
        try:
            if self.icon is not None:
                self.icon.stop()
        except Exception as exc:
            log.debug("ui_tray_stop_failed error=%s", exc)


# ── hotkeys ──────────────────────────────────────────────────────────
class HotkeyManager:
    """Global hotkeys, with one action per binding and an honest failure mode."""

    def __init__(
        self, *, on_action: Callable[[str], None] | None = None, hotkeys: dict[str, str] | None = None
    ) -> None:
        self.hotkeys = dict(hotkeys or HOTKEYS)
        self.on_action = on_action
        self.bound: list[str] = []

    def actions(self) -> dict[str, str]:
        """`{action: combo}` — what the README and the doctor row show."""
        return dict(self.hotkeys)

    def start(self) -> bool:  # pragma: no cover - needs the keyboard package on Windows
        if not hotkeys_available():
            log.info("ui_hotkeys_unavailable reason=%s", hotkeys_missing())
            return False
        try:
            import keyboard

            for action, combo in self.hotkeys.items():
                keyboard.add_hotkey(combo, self._fire, args=(action,), suppress=False)
                self.bound.append(combo)
            return True
        except Exception as exc:  # keyboard needs Windows privileges for some combos
            log.warning("ui_hotkeys_failed error=%s", exc)
            return False

    def _fire(self, action: str) -> None:
        callback = self.on_action
        if callable(callback):
            callback(action)

    def stop(self) -> None:  # pragma: no cover - needs the keyboard package
        try:
            import keyboard

            keyboard.unhook_all_hotkeys()
        except Exception as exc:
            log.debug("ui_hotkeys_stop_failed error=%s", exc)
        self.bound.clear()


__all__ = [
    "HOTKEYS",
    "MIN_SIZE",
    "ORB_SIZE",
    "PANEL_SIZE",
    "RESTART_BUDGET",
    "TRAY_MENU",
    "HotkeyManager",
    "InstanceLock",
    "OrbWindow",
    "TrayIcon",
    "WindowState",
    "hotkey_bindings",
    "hotkeys_available",
    "hotkeys_missing",
    "tray_available",
    "tray_missing",
    "ui_state_dir",
    "window_available",
    "window_missing",
    "windows",
]
