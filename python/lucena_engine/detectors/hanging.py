"""hanging-piece detector — SEE-based, board-core only (LLD §2.3).

A piece is *hanging* when the opposite side has a capture that wins material by
static exchange (SEE > 0). We report both directions:

- **opportunities**: enemy pieces the side to move can win now — enumerated
  directly from the side-to-move's legal captures.
- **dangers**: the side-to-move's own pieces the opponent can win — surfaced by
  passing (`null_move`) and reading the opponent's legal captures. This is the
  static, engine-free floor under the null-move *threat* probe (`null_move.py`),
  which adds real search on top.

The material verdict is SEE, never a piece-value table; `PIECE_CP` here only
shapes salience and the human sentence. Concept: `hanging-pieces`.
"""

from __future__ import annotations

from ..facts import Fact
from ._util import PIECE_CP, PIECE_NAME, captures_by_side_to_move, occupancy

CONCEPT = "hanging-pieces"


def _salience(see_cp: int, *, danger: bool) -> float:
    base = 0.60 + min(0.35, see_cp / 2000.0)
    if danger:  # a piece *you* are about to lose is more urgent to mention
        base = min(0.98, base + 0.05)
    return round(base, 3)


def _best_capture_per_square(board, occ):
    """square -> (best_uci, best_see, victim) keeping the max-SEE capture; ties
    broken toward the least valuable attacker (LVA), then lowest uci. Only
    material-winning captures (SEE > 0) survive."""
    cands: dict[str, list] = {}
    for uci, tgt, victim in captures_by_side_to_move(board, occ):
        see = board.see(uci)
        if see <= 0:
            continue
        attacker = occ.get(uci[:2])
        atk_val = PIECE_CP.get(attacker.piece, 0) if attacker else 0
        # lower key = better: max see, then cheapest attacker, then lowest uci
        cands.setdefault(tgt, []).append(((-see, atk_val, uci), uci, see, victim))
    return {
        sq: (best[1], best[2], best[3])
        for sq, entries in cands.items()
        for best in [min(entries, key=lambda e: e[0])]
    }


def detect_hanging(board) -> list[Fact]:
    facts: list[Fact] = []
    occ = occupancy(board)
    # Name the mover in the fact text — a bare SAN ("Bxg2 wins the bishop") carries no side, and a
    # reader (esp. freeform, which strips "you/your") then mis-assigns it. An opportunity is the
    # mover's capture; a danger is the OPPONENT's capture of the mover's own piece.
    mover = "White" if board.side_to_move == "white" else "Black"
    opp = "Black" if mover == "White" else "White"

    # -- opportunities: what the side to move can win -----------------------
    for sq, (uci, see, victim) in _best_capture_per_square(board, occ).items():
        name = PIECE_NAME.get(victim.piece, "piece")
        san = board.san(uci)
        facts.append(
            Fact(
                kind="hanging",
                squares=[sq, uci[:2]],
                text=f"{mover} can play {san}, winning the {name} on {sq}",
                provenance=f"see:{uci}",
                salience=_salience(see, danger=False),
                concept_id=CONCEPT,
            )
        )

    # -- dangers: what the opponent can win if you do nothing ---------------
    try:
        passed = board.null_move()
    except ValueError:
        passed = None  # side to move is in check; cannot pass
    if passed is not None:
        pocc = occupancy(passed)
        for sq, (uci, see, victim) in _best_capture_per_square(passed, pocc).items():
            name = PIECE_NAME.get(victim.piece, "piece")
            san = passed.san(uci)
            facts.append(
                Fact(
                    kind="hanging",
                    squares=[sq, uci[:2]],
                    text=f"{opp} threatens {san}, winning {mover}'s {name} on {sq}",
                    provenance=f"nullsee:{uci}",
                    salience=_salience(see, danger=True),
                    concept_id=CONCEPT,
                )
            )
    return facts
