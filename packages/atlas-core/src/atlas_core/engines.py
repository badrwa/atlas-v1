"""Shared behaviour for engines.

Almost every engine answers "are you loaded?" the same way, and the ones that
don't are the interesting ones: they own a model, a subprocess or a socket, and
they override `load`/`unload` to acquire it.  Keeping the trivial version in one
place stops five copies of the same three methods from drifting apart — which is
exactly what the duplicate gate caught between `fakes.FakeEngine` and the cloud
recogniser.
"""

from __future__ import annotations


class LoadedFlag:
    """The bookkeeping half of the `Engine` contract.

    Subclasses with real resources override whatever they need; a subclass that
    only needs `load()` to do nothing inherits all three.
    """

    _loaded: bool = False

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded


__all__ = ["LoadedFlag"]
