"""Puzzle solution TREE builder — engine-computed (not a single scripted line).

chess.com/Lichess puzzles play ONE opponent reply — the move the game happened
to feature — so you only ever refute one defense and can "solve" a tactic you
don't actually understand. This builds the whole tree instead: at each of *your*
turns there must be a single clearly-best move (the "only move"); at each
*opponent* turn we enumerate every reasonable defense (within a win% band of
their best), and you must solve them all. The engine is what makes this possible;
it's the rigor the grounded architecture buys us over the big sites.

Board-core + engine only (GPL hygiene holds). Local Stockfish, so the extra
tree search is metered-free (frugality governs Claude tokens, not compute).
"""

from __future__ import annotations

from .board import Board
from .evalmodel import win_pct_from_score

# Tunables (win%, from the relevant side's POV).
SOLVE_FLOOR = 55.0        # below this the solver isn't winning — not a puzzle node
UNIQUE_MARGIN = 15.0      # best must beat 2nd-best by this to be "the only move"
DEFENSE_BAND = 12.0       # opponent replies within this of their best are real tries
MAX_DEFENSES = 3          # cap the branching so sharp positions don't explode
MAX_DEPTH = 6             # solver moves deep


def build_puzzle_tree(fen, engine, *, nodes=None, movetime_ms=None,
                      max_depth=MAX_DEPTH) -> dict:
    """Build the solution tree for the side to move in `fen`. Determinism split:
    tests pass `nodes=`, production `movetime_ms=` (as `Engine.analyse`)."""
    board = Board(fen)
    root = _solver_node(board, engine, 0, max_depth, nodes, movetime_ms)
    return {"schema": 1, "fen": fen, "side_to_solve": board.side_to_move, "root": root}


def _solver_node(board, engine, depth, max_depth, nodes, mv) -> dict:
    """The solver is to move: return the one required move + what follows, or a
    terminal node when the tactic is over (no unique winning move left)."""
    a = engine.analyse(board.fen, nodes=nodes, movetime_ms=mv, multipv=2)
    if not a.best.pv:
        return {"kind": "done", "reason": "terminal"}
    best = a.best.pv[0]
    best_win = win_pct_from_score(a.best.score)
    second_win = win_pct_from_score(a.lines[1].score) if len(a.lines) > 1 else 0.0

    if best_win < SOLVE_FLOOR:
        return {"kind": "done", "fen": board.fen, "reason": "not_winning", "win_pct": round(best_win, 1)}
    if depth >= max_depth:
        return {"kind": "done", "fen": board.fen, "reason": "depth", "win_pct": round(best_win, 1)}
    if (best_win - second_win) < UNIQUE_MARGIN:
        # more than one good move — nothing unique to "find"; you've converted.
        return {"kind": "done", "fen": board.fen, "reason": "converted", "win_pct": round(best_win, 1)}

    after = board.apply(best)
    return {
        "kind": "solve",
        "fen": board.fen,
        "expect_uci": best,
        "expect_san": board.san(best),
        "win_pct": round(best_win, 1),
        "after": _opponent_node(after, engine, depth, max_depth, nodes, mv),
    }


def _opponent_node(board, engine, depth, max_depth, nodes, mv) -> dict:
    """The opponent is to move (after the solver's required move): enumerate the
    reasonable defenses; each recurses to another solver node. No reasonable
    reply (mate/stalemate) ends the branch — the solver's move finished it."""
    if not board.legal_moves():
        return {"kind": "done", "fen": board.fen,
                "reason": "mate" if board.in_check else "stalemate"}

    a = engine.analyse(board.fen, nodes=nodes, movetime_ms=mv, multipv=MAX_DEFENSES + 2)
    opp_best = win_pct_from_score(a.best.score)  # opponent POV
    defenses = []
    for line in a.lines:
        if not line.pv:
            continue
        w = win_pct_from_score(line.score)
        if opp_best - w > DEFENSE_BAND:
            break  # lines are best-first; the rest are worse than a real try
        uci = line.pv[0]
        defenses.append({
            "uci": uci,
            "san": board.san(uci),
            "then": _solver_node(board.apply(uci), engine, depth + 1, max_depth, nodes, mv),
        })
        if len(defenses) >= MAX_DEFENSES:
            break
    return {"kind": "reply", "fen": board.fen, "defenses": defenses}


def count_leaves(node: dict) -> int:
    """How many distinct solver-lines the tree forces (for logging / a progress bar)."""
    if node.get("kind") == "solve":
        return count_leaves(node["after"])
    if node.get("kind") == "reply":
        return sum(count_leaves(d["then"]) for d in node["defenses"]) or 1
    return 1
