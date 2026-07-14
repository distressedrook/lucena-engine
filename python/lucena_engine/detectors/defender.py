"""defender-removed detector — the "remove the guard" motif (LLD §2.3).

Static, board-core only. Fires when an enemy piece E that the side to move
already attacks is held up by *exactly one* defender D, and that defender is
itself attackable by the side to move. Removing D leaves E hanging — the
classic "the knight is c6's only defender" fact.

This is deliberately a two-ply *hint*, not a verdict: it says a guard is
removable, not that the whole combination is sound (the engine / SEE decide
that). Concept: `removing-the-defender`.
"""

from __future__ import annotations

from ..facts import Fact
from ._util import PIECE_CP, PIECE_NAME, occupancy, opponent

CONCEPT = "removing-the-defender"


def _salience(victim_cp: int) -> float:
    return round(0.50 + min(0.30, victim_cp / 1000.0), 3)


def detect_defender_removed(board) -> list[Fact]:
    facts: list[Fact] = []
    stm = board.side_to_move
    opp = opponent(stm)
    occ = occupancy(board)

    for sq, piece in occ.items():
        if piece.color != opp or piece.piece == "K":
            continue  # only enemy non-king pieces are winnable targets
        stm_attackers = board.attackers(sq, stm)
        if not stm_attackers:
            continue
        defenders = board.attackers(sq, opp)
        if len(defenders) != 1:
            continue  # only the single-guard case is a clean "only defender"
        guard = defenders[0]
        # the guard must itself be attackable by the side to move to be removed
        if not board.attackers(guard, stm):
            continue
        gpiece = occ.get(guard)
        vname = PIECE_NAME.get(piece.piece, "piece")
        gname = PIECE_NAME.get(gpiece.piece, "piece") if gpiece else "piece"
        facts.append(
            Fact(
                kind="defender-removed",
                squares=[guard, sq],
                text=(
                    f"the {gname} on {guard} is the only defender of "
                    f"the {vname} on {sq}"
                ),
                provenance=f"static:{guard}->{sq}",
                salience=_salience(PIECE_CP.get(piece.piece, 0)),
                concept_id=CONCEPT,
            )
        )
    return facts
