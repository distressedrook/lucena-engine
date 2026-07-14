"""Maia-3 — the human-move *predictor* (the difficulty lens), NOT a truth source.

Maia answers one question: *"what would a player rated R most likely play here?"*
It never grounds anything — Stockfish owns eval, legality, and quality; Maia only
informs **what to teach, to whom, how hard** (docs/content-architecture.md §6,
backlog 2026-07-07). Same firewall as the fact sheet: a `MaiaEngine` result is a
list of human-likely moves, never a centipawn number anyone trusts.

Runtime: `maia3-uci` (CSSLab/maia3, AGPL-3.0) — a PyTorch model exposed over UCI,
so it runs as a subprocess exactly like Stockfish. Point `LUCENA_MAIA` at the
command (default `maia3-5m`, the CPU/GUI-friendly variant). It speaks a different
UCI dialect than Stockfish — `SelfElo`/`OppoElo` for rating, `MultiPV` for the
top-N human moves, `Temperature`/`TopP` for sampling — so it gets its own class,
not the Stockfish `Engine`.

Determinism: one forward pass at `Temperature 0` (argmax ordering) → the ranked
top-N is reproducible; no search, no thread/nodes non-determinism to pin.
"""

from __future__ import annotations

import os
import subprocess
import threading


class MaiaError(RuntimeError):
    """Maia subprocess failed or is misconfigured (names the recovery)."""


class MaiaEngine:
    """A live `maia3-uci` process. Not thread-safe; the caller serializes access
    (the MCP server already holds one lock). Predictor only — no eval, no grounding."""

    def __init__(self, path: str | None = None):
        # `LUCENA_MAIA` may be a bare command ("maia3-5m") or a full path; split on
        # spaces so "python -m maia3.uci --model maia3-5m" also works.
        self._cmd = (path or os.environ.get("LUCENA_MAIA", "maia3-5m")).split()
        self._proc: subprocess.Popen | None = None
        self._name = ""
        self._cur_multipv = 0
        self._cur_self = self._cur_oppo = None
        # Serializes the stateful UCI conversation in `top_human_moves`: the coach and the parallel
        # poisoned-line-detection thread now share this one subprocess, so access must be atomic.
        self._lock = threading.Lock()
        self._start()

    # -- lifecycle ---------------------------------------------------------
    def _start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                self._cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1,
            )
        except FileNotFoundError as e:
            raise MaiaError(
                f"maia3 engine not found: {self._cmd!r}. Install it "
                f"(pip install 'git+https://github.com/CSSLab/maia3') and set "
                f"LUCENA_MAIA to the command (e.g. maia3-5m)."
            ) from e
        self._name = self._handshake()
        # Deterministic ranked output: argmax ordering, no nucleus filtering — we want
        # the full ranked list, not a sampled move.
        self._set("Temperature", 0)
        self._set("TopP", 1.0)
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
        return self._proc is not None and self._proc.poll() is None

    def __enter__(self) -> "MaiaEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- io ----------------------------------------------------------------
    def _send(self, cmd: str) -> None:
        if not self._proc or self._proc.poll() is not None:
            raise MaiaError("maia3 process is not running")
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
        raise MaiaError(f"maia3 closed stdout before {until_prefix!r}")

    def _set(self, name: str, value) -> None:
        self._send(f"setoption name {name} value {value}")

    def _isready(self) -> None:
        self._send("isready")
        self._readlines("readyok")

    # -- the one primitive: predict what a human plays ---------------------
    def top_human_moves(self, fen: str, rating: int, *, n: int = 5,
                        oppo_rating: int | None = None) -> list[dict]:
        """The `n` moves a player rated `rating` is most likely to play in `fen`,
        ranked most-likely first. `oppo_rating` defaults to `rating` (even game).

        Returns `[{"uci", "rank", "policy"?, "wdl"?, "cp"?}, …]` — `rank` 1 is the top
        human pick. `policy` is the move's *real* probability (float in `[0, 1]`) from
        the model's policy head — present only when the wire emits it (the
        policy-emitting `tools/maia_policy_uci.py` wrapper does; stock `maia3-uci` does
        not). This is a *prediction of behaviour*, not a judgement of quality: never
        surface `cp`/`wdl` as an evaluation; ground the move with Stockfish for that.
        """
        oppo = oppo_rating if oppo_rating is not None else rating
        with self._lock:                            # atomic: coach + poisoned-line thread share this process
            if self._cur_self != rating:
                self._set("SelfElo", rating); self._cur_self = rating
            if self._cur_oppo != oppo:
                self._set("OppoElo", oppo); self._cur_oppo = oppo
            if self._cur_multipv != n:
                self._set("MultiPV", n); self._cur_multipv = n
            self._send(f"position fen {fen}")
            self._send("go nodes 1")                # Maia is a single forward pass
            lines = self._readlines("bestmove")
            return _parse_multipv(lines)


def _parse_multipv(lines: list[str]) -> list[dict]:
    """Pull the ranked human moves out of `info … multipv K … pv <uci> …` lines.
    Keeps the last info line per `multipv` rank (engines re-emit as they refine).
    Tolerant to which fields are present (wdl/score may or may not appear)."""
    by_rank: dict[int, dict] = {}
    for line in lines:
        if not line.startswith("info ") or " pv " not in line or "multipv" not in line:
            continue
        toks = line.split()
        entry: dict = {}
        try:
            for i, t in enumerate(toks):
                if t == "multipv":
                    entry["rank"] = int(toks[i + 1])
                elif t == "pv":
                    entry["uci"] = toks[i + 1]
                elif t == "wdl":
                    entry["wdl"] = [int(toks[i + 1]), int(toks[i + 2]), int(toks[i + 3])]
                elif t == "policy":
                    entry["policy"] = float(toks[i + 1])
                elif t == "score" and toks[i + 1] in ("cp", "mate"):
                    entry[toks[i + 1]] = int(toks[i + 2])
        except (IndexError, ValueError):
            continue
        if "rank" in entry and "uci" in entry:
            by_rank[entry["rank"]] = entry
    return [by_rank[k] for k in sorted(by_rank)]
