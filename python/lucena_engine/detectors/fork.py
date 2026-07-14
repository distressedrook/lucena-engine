"""fork / double-attack detector — the "one move, two targets" motif (LLD §2.3).

Static, board-core only. Fires when the side to move has a legal move whose
piece, from its destination, attacks **two or more things worth winning** — two
loose pieces (a queen forking two rooks), or a check plus a loose piece (a royal
knight fork). The defender cannot parry both, so material falls.

This is the motif the M3 trio missed: the fact sheet surfaced only distractors
on the fork positions (case-02 Nd5+, case-03 Qd4) and left the winning double
attack to the coach's unaided vision. It is *deterministic* — pure
`attackers()`/`defenders()` geometry — so grounding it removes that gap.

Like every detector this is a salience *hint*, not a verdict: it flags a
plausible double attack, and `build_fact_sheet` reconciles it against the engine
(a fork the engine rejects — the forking piece is itself lost, or something is
simply better — is dropped, the §2.3 rule *a SEE fact that contradicts the
engine loses*). Board-only, it is best-effort. Concept: `forks-double-attack`.
"""

from __future__ import annotations

from ..facts import Fact
from ._util import PIECE_CP, PIECE_NAME, forked_targets, king_square, occupancy

CONCEPT = "forks-double-attack"


def _salience(winnable_cp: int, *, check: bool) -> float:
    base = 0.70 + min(0.26, winnable_cp / 2000.0)
    if check:  # a checking fork is forcing — the opponent can't just walk away
        base = min(0.98, base + 0.03)
    return round(base, 3)


def _phrase(mover_piece: str, targets: list[tuple[str, str, int]],
            king_checked: bool, discovered_check: bool) -> str:
    kn = PIECE_NAME.get(mover_piece, "piece")
    named = [f"the {PIECE_NAME.get(pc, 'piece')} on {sq}" for sq, pc, _ in targets]
    joined = named[0] if len(named) == 1 else " and ".join(
        [", ".join(named[:-1]), named[-1]] if len(named) > 2 else named)
    if king_checked:
        return f"the {kn} forks the king and {joined}"
    if discovered_check:
        return f"the {kn} hits {joined} while the check lands"
    return f"the {kn} forks {joined}"


def detect_fork(board) -> list[Fact]:
    facts: list[Fact] = []
    stm = board.side_to_move
    occ = occupancy(board)
    enemy_king = king_square(board, "white" if stm == "black" else "black")

    for m in board.legal_moves():
        src, to = m[:2], m[2:4]
        mover = occ.get(src)
        if mover is None:
            continue
        mover_cp = PIECE_CP.get(mover.piece, 0)
        after = board.apply(m)

        targets = forked_targets(after, to, stm)          # loose/valuable prongs
        king_checked = enemy_king is not None and to in after.attackers(enemy_king, stm)
        gives_check = after.in_check                       # incl. discovered checks

        # A prong is *winnable* when its piece is undefended, or worth more than
        # the forker (so the capture nets material even through the defender).
        winnable = [t for t in targets
                    if t[2] == 0 or PIECE_CP.get(t[1], 0) > mover_cp]
        is_fork = len(winnable) >= 2 or (gives_check and len(winnable) >= 1)
        if not is_fork:
            continue
        # Anti-spam floor for the no-engine path: a non-checking move that simply
        # loses the forker by SEE is not a real fork (the engine gate catches the
        # rest, incl. pinned-defender cases where SEE lies — hence the check let-off).
        if board.see(m) < 0 and not gives_check:
            continue

        winnable_cp = sum(PIECE_CP.get(pc, 0) for _, pc, _ in winnable)
        facts.append(
            Fact(
                kind="fork",
                squares=[to] + [sq for sq, _, _ in winnable],
                text=_phrase(mover.piece, winnable, king_checked,
                             gives_check and not king_checked),
                provenance=f"fork:{m}",
                salience=_salience(winnable_cp, check=gives_check),
                concept_id=CONCEPT,
            )
        )
    return facts
