"""Grounded hint-ladder derivation (coach-contract "Socratic core").

A probe beat carries a `hints` ladder — 2–3 nudges ordered vague → specific,
none of which states the answer. The danger is an LLM *inventing* those nudges
(the exact confident-wrong failure the fact sheet exists to kill). So the ladder
is derived here, server-side, from grounded truth: the engine's own principal
variation plus the board core's attacker geometry. Every rung is a partial
reveal of the winning line — **true by construction**, because the PV is already
the best line and the geometry is computed, not guessed. The LLM only phrases
what this returns; it never adds a fact.

Scope: the ladder grounds a *tactic* — a fork / king-hunt (a mover move whose
piece attacks two-plus targets at once, or a check that wins a piece), or a
direct material-winning capture. When the line has no such handle (quiet /
positional best moves), `derive_hints` returns `[]` and the coach falls back to
a mastery-tiered question rather than a fabricated hint. Board-core only besides
the engine PV it is handed — GPL hygiene holds (LLD §2.0).
"""

from __future__ import annotations

from dataclasses import dataclass

from .detectors._util import (
    PIECE_CP,
    PIECE_NAME,
    forked_targets,
    king_square,
    occupancy,
    opponent,
)

# A target worth forking is a real piece, not a pawn; the king counts only as
# the check half of a royal fork (you never *win* it).
_MINOR = PIECE_CP["N"]


@dataclass(frozen=True)
class Hint:
    """One rung of a grounded hint ladder. `text` is a terse, *withholding*
    nudge (the LLM phrases it warmer); `squares` paint the same arrow the hint
    talks about; `provenance` traces it to the PV/geometry it came from."""

    rung: int              # 1 = vaguest, ascending toward the answer
    text: str
    squares: list[str]
    provenance: str


def _valuable_targets(after, to_sq: str, mover: str) -> list[tuple[str, str]]:
    """Enemy pieces (value ≥ knight, king excluded) the piece now on `to_sq`
    attacks — the shared fork geometry, minus the defender counts the hint
    ladder does not need."""
    return [(sq, pc) for sq, pc, _ in forked_targets(after, to_sq, mover)]


def _find_tactic(board, pv: list[str], mover: str) -> dict | None:
    """Walk the mover's moves down the PV; return the first that forks (≥2
    valuable targets, or a check plus a valuable target) or, failing any fork,
    a direct material-winning capture at the head of the line.

    A fork deep in the PV is only usable as a hint if its targets are pieces the
    player can see on the board *now* — a piece that has drifted to a different
    square by the fork ply would be a phantom in the nudge (the exact grounding
    bug this guards). So targets are filtered to those on the current position."""
    root_occ = {p.square: p.piece for p in board.piece_list()}
    cur = board
    direct: dict | None = None
    for i, u in enumerate(pv):
        nxt = cur.apply(u)
        if i % 2 == 0:  # a mover move (mover is to-move at ply 0)
            to_sq = u[2:4]
            targets = [(sq, pc) for sq, pc in _valuable_targets(nxt, to_sq, mover)
                       if root_occ.get(sq) == pc]  # only pieces that are there NOW
            is_check = nxt.in_check
            piece = _piece_letter(cur, u[:2])
            if len(targets) >= 2 or (is_check and targets):
                return {
                    "uci": u, "to": to_sq, "piece": piece,
                    "targets": targets, "is_check": is_check, "ply": i,
                }
            if direct is None and i == 0 and board.see(u) > 0:
                victim = occupancy(board).get(to_sq)
                if victim is not None and PIECE_CP.get(victim.piece, 0) >= _MINOR:
                    direct = {"uci": u, "to": to_sq, "piece": piece,
                              "victim": victim.piece, "is_check": is_check}
        cur = nxt
    return direct


def _piece_letter(board, square: str) -> str:
    for p in board.piece_list():
        if p.square == square:
            return p.piece
    return "?"


def _entry_rung(board, pv: list[str], mover: str) -> Hint:
    """Rung 3 — the forcing entry move's *nature*, one step short of the move."""
    first = pv[0]
    fp = _piece_letter(board, first[:2])
    if board.see(first) < 0:
        text = "you may have to give up material to open the way"
    elif board.apply(first).in_check:
        text = f"it starts with a forcing {PIECE_NAME.get(fp, 'piece')} check"
    else:
        text = f"the {PIECE_NAME.get(fp, 'piece')} on {first[:2]} makes the first move"
    return Hint(3, text, [first[:2]], f"pv:{first}")


def _mate_ladder(board, pv, mover, tac, mate) -> list[Hint]:
    """A mate is a king hunt, not a material grab — orient on the trapped king,
    name the piece that lands the blow if there is a clean one, then the entry."""
    ek = king_square(board, opponent(mover))
    ksq = [ek] if ek else []
    rungs = [Hint(1, f"there's a forced mate here — the target is the king on {ek}",
                  ksq, f"mate:{pv[0]}")]
    if tac and "victim" not in tac:
        kn = PIECE_NAME.get(tac["piece"], "piece")
        rungs.append(Hint(2, f"your {kn} lands the decisive blow", [tac["to"]],
                          f"mate:{tac['uci']}"))
    else:
        rungs.append(Hint(2, "it's a forcing run of checks — the king can't get away",
                          ksq, f"mate:{pv[0]}"))
    rungs.append(_entry_rung(board, pv, mover))
    return rungs


def derive_hints(board, analysis, *, max_rungs: int = 3) -> list[Hint]:
    """Derive a grounded hint ladder for the best move of `analysis` on
    `board`. Returns `[]` when the line offers no tactical handle to ground a
    nudge on (never a fabricated rung)."""
    pv = analysis.best.pv if analysis.lines else []
    if not pv:
        return []
    mover = board.side_to_move
    tac = _find_tactic(board, pv, mover)

    # A forced mate is not a material win — frame it as a king hunt, or the
    # ladder mislabels "mate in 3" as "win three pieces".
    mate = analysis.best.score.mate
    if mate is not None and mate > 0:
        return _mate_ladder(board, pv, mover, tac, mate)[:max_rungs]

    if tac is None:
        return []

    first = pv[0]
    rungs: list[Hint] = []

    if "victim" in tac:  # direct material win — a short, gentle ladder
        vn = PIECE_NAME.get(tac["victim"], "piece")
        rungs.append(Hint(1, f"your opponent's {vn} on {tac['to']} may not be safe",
                          [tac["to"]], f"pv-win:{tac['uci']}"))
        rungs.append(Hint(2, "check what defends it before you commit",
                          [tac["to"]], f"pv-win:{tac['uci']}"))
        rungs.append(Hint(3, "there is a capture that wins material",
                          [tac["to"], first[:2]], f"pv:{first}"))
        return rungs[:max_rungs]

    # -- fork / king-hunt ladder ------------------------------------------
    win_targets = tac["targets"]  # value-sorted, king already excluded
    prize = win_targets[0]
    pn = PIECE_NAME.get(prize[1], "piece")
    kn = PIECE_NAME.get(tac["piece"], "piece")
    n_hit = len(win_targets) + (1 if tac["is_check"] else 0)

    # Rung 1 — the target (vaguest): name the prize square, never the move.
    if tac["is_check"]:
        rungs.append(Hint(1, f"the enemy king and the {pn} on {prize[0]} share a weakness",
                          [prize[0]], f"fork:{tac['to']}"))
    else:
        rungs.append(Hint(1, f"your opponent's {pn} on {prize[0]} is looser than it looks",
                          [prize[0]], f"fork:{tac['to']}"))

    # Rung 2 — the resource: the key piece and that it hits several at once.
    rungs.append(Hint(2, f"your {kn} can attack {n_hit} things at once",
                      [sq for sq, _ in win_targets] + [tac["to"]],
                      f"fork:{tac['to']}"))

    # Rung 3 — the forcing entry (one step short): its nature, not its square.
    rungs.append(_entry_rung(board, pv, mover))

    return rungs[:max_rungs]
