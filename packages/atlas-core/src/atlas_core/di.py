"""A 40-line dependency injector.

Why not a framework?  Because the plan's rule is "no dependency without a
reason", and a conversation engine needs exactly three things: register,
resolve, override-in-tests.  Anything more would be architecture theatre.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from atlas_core.errors import ConfigError

T = TypeVar("T")

Factory = Callable[["Container"], Any]


class Container:
    """Tiny DI container: interface → factory, singleton or transient."""

    def __init__(self) -> None:
        self._factories: dict[Any, Factory] = {}
        self._instances: dict[Any, Any] = {}
        self._singletons: set[Any] = set()

    # ── registration ─────────────────────────────────────────────────
    def register(self, key: Any, factory: Factory, *, singleton: bool = True) -> Container:
        self._factories[key] = factory
        if singleton:
            self._singletons.add(key)
        return self

    def register_instance(self, key: Any, instance: Any) -> Container:
        self._factories[key] = lambda _c: instance
        self._instances[key] = instance
        self._singletons.add(key)
        return self

    # ── resolution ───────────────────────────────────────────────────
    def resolve(self, key: Any) -> Any:
        if key in self._instances:
            return self._instances[key]
        if key not in self._factories:
            raise ConfigError(f"nothing registered for {key!r}")
        instance = self._factories[key](self)
        if key in self._singletons:
            self._instances[key] = instance
        return instance

    # ── tests ────────────────────────────────────────────────────────
    def override(self, key: Any, instance: Any) -> Container:
        """Replace a registration (used by the fake-injection fixture)."""
        self._instances[key] = instance
        return self

    def clear_caches(self) -> None:
        """Drop singletons but keep factories (rebuild between tests)."""
        self._instances = {k: v for k, v in self._instances.items() if k not in self._factories}

    def __contains__(self, key: Any) -> bool:
        return key in self._factories or key in self._instances

    def describe(self) -> dict[str, str]:
        return {
            str(key): "instance" if key in self._instances else "factory"
            for key in {*self._factories, *self._instances}
        }


__all__ = ["Container", "Factory"]
