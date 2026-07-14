"""brilliant (!!) move classification — a sound sacrifice (backlog, founder HIGH).

A brilliant move is the one a player is tempted *not* to make: it gives up
material for real, the engine (near-)endorses it, it was a genuine choice (not
forced), and it did not come from an already-won position. Precision over recall
is the whole point — the credibility win over chess.com's false-positive rate is
firing *only* on engine-confirmed sound sacrifices. Contract:
`docs/contracts/brilliant-classification.md`.

It is an overlay on the glyph classification: it upgrades an otherwise
`ok`/`only_move` move and never masks a real error.
"""

from __future__ import annotations

# win-% thresholds (from the mover's POV); named + tunable per the contract.
SOUND_MARGIN = 5.0            # played may trail best by at most this (if not itself best)
FORCED_FLOOR = 20.0          # a non-losing alternative must beat this — else the sac was forced (!)
ALREADY_WINNING_CEIL = 70.0  # if the best *alternative* already wins, the sac isn't the point
SOUND_FLOOR = 50.0           # the played sac must itself reach ~equality/advantage


def is_material_sacrifice(board, uci: str) -> bool:
    """True iff `uci` gives up material for real — board-core only, and not
    fooled by SEE-literalism (a pinned/illegal recapture).

    `board.see(uci) < 0` (static exchange loses material on the square) AND, in
    the position after the move, the opponent has at least one *legal* move
    landing on the destination square (the sacrificed piece can actually be
    taken). `Nd5+` in case-02 fails the second clause — `exd5` is illegal (pin),
    so the knight can't be legally captured: it is a fork, not a sacrifice.

    v1 recognizes a *direct* sacrifice (the move itself loses material by SEE);
    a quiet/offered sacrifice (a move leaving a different piece en prise) is
    deferred and returns False.
    """
    if board.see(uci) >= 0:
        return False
    dest = uci[2:4]
    after = board.apply(uci)
    return any(m[2:4] == dest for m in after.legal_moves())


def classify_brilliant(*, is_sacrifice: bool, best_win: float, played_win: float,
                       second_win: float | None, played_is_best: bool) -> bool:
    """Pure decision over win-% (mover POV). `second_win` is the engine's
    second-best line, or None when only one legal move exists. All five
    conditions are necessary."""
    if not is_sacrifice:
        return False
    if not (played_is_best or (best_win - played_win) <= SOUND_MARGIN):  # sound
        return False
    if second_win is None or not (second_win > FORCED_FLOOR):            # a real choice
        return False
    if not (second_win < ALREADY_WINNING_CEIL):                         # not already winning
        return False
    return played_win >= SOUND_FLOOR                                    # worthwhile


def is_brilliant(board, uci: str, *, best_win: float, played_win: float,
                 second_win: float | None, played_is_best: bool) -> bool:
    """Convenience: the sacrifice check plus the pure win-% decision."""
    return classify_brilliant(
        is_sacrifice=is_material_sacrifice(board, uci),
        best_win=best_win, played_win=played_win,
        second_win=second_win, played_is_best=played_is_best,
    )
