"""Persistent Stockfish session over UCI/stdio.

GPL posture: Stockfish is a separate, unmodified process; we speak UCI text
to it and never link it. See LICENSE / PRD §11.

Request/response is synchronized: after `go` we read stdout until `bestmove`
before issuing the next command. (A naïve fire-and-forget pipe lets a
following command interrupt the search — verified against SF 18.)

Determinism split (the load-bearing API choice): production callers pass
`movetime_ms`; tests pass `nodes` and construct with `threads=1`. Node-count
search with a single thread is reproducible across machines; movetime is not.
"""

from __future__ import annotations

import os
import subprocess
import threading
from dataclasses import dataclass

from .evalmodel import Score, parse_score

PINNED_MAJOR = "Stockfish 18"


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Line:
    """One analysed line (one MultiPV slot)."""

    rank: int          # 1 = best
    score: Score       # from side-to-move POV
    pv: list[str]      # UCI moves, best-first


@dataclass(frozen=True)
class Analysis:
    fen: str
    lines: list[Line]  # sorted by rank; len == requested multipv (or fewer near mate)

    @property
    def best(self) -> Line:
        return self.lines[0]


def _default_threads() -> int:
    return max(2, (os.cpu_count() or 4) - 2)


class Engine:
    """A live Stockfish process. Not thread-safe; the MCP server serializes
    access behind one lock (LLD-analysis OQ#11)."""

    def __init__(
        self,
        path: str | None = None,
        *,
        threads: int | None = None,
        hash_mb: int = 256,
        assert_version: bool = True,
    ):
        self._path = path or os.environ.get("LUCENA_STOCKFISH", "stockfish")
        self._threads = threads if threads is not None else _default_threads()
        self._hash = hash_mb
        self._assert_version = assert_version
        self._proc: subprocess.Popen | None = None
        self._restarts = 0
        self._cur_multipv = 1
        # One Stockfish subprocess = one UCI conversation. Serialize every analyse/eval so a move
        # pushed over /move (the drill's Maia call) can't interleave commands with a coach tool
        # call. Reentrant: _with_restart may re-enter after a restart.
        self._lock = threading.RLock()
        self._start()

    # -- lifecycle ---------------------------------------------------------
    def _start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                [self._path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            raise EngineError(f"Stockfish not found at {self._path!r} "
                              f"(set LUCENA_STOCKFISH)") from e
        banner = self._handshake()
        self._name = banner or "Stockfish"     # e.g. "Stockfish 18" — for the analysis readout
        if self._assert_version and PINNED_MAJOR not in banner:
            self.close()
            raise EngineError(
                f"pinned {PINNED_MAJOR!r} required; engine reports {banner!r}. "
                f"The NNUE eval parser is version-specific."
            )
        self._set("Threads", self._threads)
        self._set("Hash", self._hash)
        self._set("MultiPV", 1)
        self._isready()

    def _handshake(self) -> str:
        self._send("uci")
        name = ""
        for line in self._readlines("uciok"):
            if line.startswith("id name "):
                name = line[len("id name "):]
        return name

    def close(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._send("quit")
                self._proc.wait(timeout=2)
            except Exception:
                self._proc.kill()
        self._proc = None

    @property
    def is_alive(self) -> bool:
        """True if the Stockfish child is running (for the MCP heartbeat)."""
        return self._proc is not None and self._proc.poll() is None

    @property
    def name(self) -> str:
        """The engine's reported id (e.g. 'Stockfish 18')."""
        return getattr(self, "_name", "Stockfish")

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- io ----------------------------------------------------------------
    def _send(self, cmd: str) -> None:
        if not self._proc or self._proc.poll() is not None:
            raise EngineError("engine process is not running")
        assert self._proc.stdin is not None
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _readlines(self, until_prefix: str) -> list[str]:
        assert self._proc and self._proc.stdout is not None
        out: list[str] = []
        for line in self._proc.stdout:
            line = line.rstrip("\n")
            out.append(line)
            if line.startswith(until_prefix):
                return out
        # stream ended without the sentinel → process died mid-command
        raise EngineError(f"engine closed stdout before {until_prefix!r}")

    def _set(self, name: str, value) -> None:
        self._send(f"setoption name {name} value {value}")

    def _isready(self) -> None:
        self._send("isready")
        self._readlines("readyok")

    # -- analysis ----------------------------------------------------------
    def new_game(self) -> None:
        """Clear hash between unrelated positions (ucinewgame)."""
        self._send("ucinewgame")
        self._isready()

    def analyse(
        self,
        fen: str,
        *,
        movetime_ms: int | None = None,
        nodes: int | None = None,
        depth: int | None = None,
        multipv: int = 1,
    ) -> Analysis:
        """Analyse `fen`. Provide exactly one of movetime_ms / nodes / depth.

        Production: movetime_ms. Tests: nodes (with threads=1) for
        reproducibility.
        """
        limits = [("movetime", movetime_ms), ("nodes", nodes), ("depth", depth)]
        chosen = [(k, v) for k, v in limits if v is not None]
        if len(chosen) != 1:
            raise ValueError("pass exactly one of movetime_ms / nodes / depth")

        return self._with_restart(lambda: self._analyse_once(fen, chosen[0], multipv))

    def _analyse_once(self, fen: str, limit: tuple[str, int], multipv: int) -> Analysis:
        if multipv != self._cur_multipv:
            self._set("MultiPV", multipv)
            self._cur_multipv = multipv
            self._isready()
        self._send(f"position fen {fen}")
        self._send(f"go {limit[0]} {limit[1]}")
        raw = self._readlines("bestmove")
        return _parse_analysis(fen, raw, multipv)

    def raw_eval(self, fen: str) -> list[str]:
        """Raw text of the `eval` command (NNUE piece-value grid + totals).

        Debug-format output — parsed by nnue.py, which is why the engine
        version is pinned. Returns the lines from the command.
        """
        def run():
            self._send(f"position fen {fen}")
            self._send("eval")
            return self._readlines("Final evaluation")
        return self._with_restart(run)

    def _with_restart(self, fn):
        with self._lock:
            return self._with_restart_locked(fn)

    def _with_restart_locked(self, fn):
        try:
            return fn()
        except EngineError:
            if self._restarts >= 1:
                raise
            self._restarts += 1
            self._proc = None
            self._start()
            return fn()


def _parse_analysis(fen: str, raw: list[str], multipv: int) -> Analysis:
    # Keep the LAST info line per multipv slot (deepest completed iteration).
    by_rank: dict[int, Line] = {}
    for line in raw:
        if not line.startswith("info ") or " score " not in line or " pv " not in line:
            continue
        toks = line.split()
        try:
            rank = int(toks[toks.index("multipv") + 1]) if "multipv" in toks else 1
            si = toks.index("score")
            score = parse_score(toks[si + 1], int(toks[si + 2]))
            pv = toks[toks.index("pv") + 1:]
        except (ValueError, IndexError):
            continue
        by_rank[rank] = Line(rank=rank, score=score, pv=pv)

    if not by_rank:
        # No scored info: terminal position (mate/stalemate) → bestmove (none).
        bestmove = next((l.split()[1] for l in raw if l.startswith("bestmove")), None)
        if bestmove in (None, "(none)"):
            raise EngineError(f"no legal moves / terminal position: {fen}")
        raise EngineError(f"engine returned no scored lines for {fen}")

    lines = [by_rank[r] for r in sorted(by_rank)]
    return Analysis(fen=fen, lines=lines)
