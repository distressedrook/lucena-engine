"""A small pool of Stockfish engines for concurrent search.

⚠️ SHELVED / currently unused. Built for the forcing-line tree, but measurement
showed thread+pool parallelism made that builder *slower*, not faster — the tree
is a mostly-sequential main line (near-zero parallel width) and Python's GIL
serialises the per-engine result reads (see learnings.md #13). `line_tree.py` is
therefore serial. This module is kept on the shelf for a future case where a
workload is measured to have genuine parallel width AND uses process-level
parallelism to beat the GIL — not imported anywhere today.

Each engine is its own Stockfish process at `Threads=1`, so a `nodes`-limited
search is deterministic regardless of how many run at once. GPL hygiene holds:
still just Stockfish subprocesses over UCI, only more than one.
"""

from __future__ import annotations

import queue
from contextlib import contextmanager

from .uci import Engine


class EnginePool:
    def __init__(self, size: int = 4, *, path: str | None = None):
        if size < 1:
            raise ValueError("pool size must be >= 1")
        self._free: queue.Queue = queue.Queue()
        self._all: list[Engine] = []
        for _ in range(size):
            e = Engine(path=path, threads=1)   # Threads=1 -> per-engine determinism
            self._all.append(e)
            self._free.put(e)

    @property
    def size(self) -> int:
        return len(self._all)

    @contextmanager
    def checkout(self):
        """Borrow an engine for the duration of one search, then return it. Blocks
        if all engines are busy — the pool size is the real concurrency cap."""
        engine = self._free.get()
        try:
            yield engine
        finally:
            self._free.put(engine)

    def close(self) -> None:
        for e in self._all:
            e.close()

    def __enter__(self) -> "EnginePool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
