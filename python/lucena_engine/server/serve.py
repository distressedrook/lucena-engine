"""gRPC server bootstrap for the grounding engine.

Holds one lock-guarded Stockfish (threads=1 for reproducible node-limited search,
which poisoned-line detection also requires) and a lazy Maia. Stateless per the
contract: processes stay warm, the API carries no session.
"""

from __future__ import annotations

import os
import threading
from concurrent import futures
from contextlib import contextmanager

import grpc

from ..uci import Engine
from .._pb import engine_pb2_grpc as pbg
from .truth import TruthServicer
from .behaviour import BehaviourServicer

STOCKFISH_MAJOR = 18   # version-pinned (NNUE eval parser); asserted by Engine at startup
ENGINE_VERSION = "lucena-engine/0.1.0"


class EngineHolder:
    """One Stockfish, serialized. `acquire()` yields it under a reentrant lock with a
    fresh `ucinewgame` so node-limited search is reproducible."""

    def __init__(self, *, threads: int = 1):
        self._threads = threads
        self._lock = threading.RLock()
        self._engine: Engine | None = None

    @contextmanager
    def acquire(self):
        with self._lock:
            if self._engine is None:
                self._engine = Engine(threads=self._threads)
            self._engine.new_game()
            yield self._engine

    def close(self):
        if self._engine is not None:
            self._engine.close()
            self._engine = None


class MaiaHolder:
    """Lazy Maia — constructed on first use so the engine runs without it."""

    def __init__(self):
        self._lock = threading.RLock()
        self._maia = None

    def get(self):
        with self._lock:
            if self._maia is None:
                from ..maia import MaiaEngine
                self._maia = MaiaEngine()
            return self._maia


def build_server(*, port: int = 50051, threads: int = 1, max_workers: int = 8):
    engines = EngineHolder(threads=threads)
    maia = MaiaHolder()
    info = {
        "engine_version": ENGINE_VERSION,
        "stockfish_major": STOCKFISH_MAJOR,
        "maia_available": bool(os.environ.get("LUCENA_MAIA")),
    }
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pbg.add_TruthServicer_to_server(TruthServicer(engines, info), server)
    pbg.add_BehaviourServicer_to_server(BehaviourServicer(engines, maia), server)
    server.add_insecure_port(f"[::]:{port}")
    return server, engines


def serve(*, port: int = 50051, threads: int = 1):
    server, engines = build_server(port=port, threads=threads)
    server.start()
    print(f"lucena-engine gRPC on :{port} (threads={threads})", flush=True)
    try:
        server.wait_for_termination()
    finally:
        engines.close()


if __name__ == "__main__":
    serve(port=int(os.environ.get("LUCENA_ENGINE_PORT", "50051")))
