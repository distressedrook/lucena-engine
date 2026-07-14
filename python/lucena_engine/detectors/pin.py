"""pin detector — absolute pins (a piece stuck in front of its own king).

Pure board geometry: walk each king's rays; a friendly piece with an aligned
enemy slider behind it is pinned to the king — it cannot move off that line, and
some of its moves (including captures) are *illegal*. This grounds two things the
coach otherwise eyeballs and gets wrong: "your knight is pinned, it can't move",
and "that recapture is illegal — the pawn is pinned" (exactly case-02's `exd5`).

An absolute pin is a fact of geometry, always true, so unlike the salience-hint
detectors it needs no engine reconciliation. Board-core only (GPL hygiene holds).
Concept: `pins-skewers`.
"""

from __future__ import annotations

from ..facts import Fact
from ._util import PIECE_CP, PIECE_NAME, king_square, occupancy

CONCEPT = "pins-skewers"

# (drank, dfile) rays and the enemy sliders that pin along them.
_ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]     # rook / queen
_DIAG = [(1, 1), (1, -1), (-1, 1), (-1, -1)]    # bishop / queen


def _square(file: int, rank: int) -> str:
    return chr(97 + file) + str(rank + 1)


def _filerank(square: str) -> tuple[int, int]:
    return ord(square[0]) - 97, int(square[1]) - 1


def _salience(piece: str) -> float:
    return round(0.55 + min(0.25, PIECE_CP.get(piece, 0) / 2000.0), 3)


def detect_pin(board) -> list[Fact]:
    occ = occupancy(board)
    stm = board.side_to_move
    facts: list[Fact] = []
    for kcolor in ("white", "black"):
        ksq = king_square(board, kcolor)
        if ksq is None:
            continue
        kf, kr = _filerank(ksq)
        for rays, sliders in ((_ORTHO, ("R", "Q")), (_DIAG, ("B", "Q"))):
            for drank, dfile in rays:
                pinned = None
                f, r = kf + dfile, kr + drank
                while 0 <= f < 8 and 0 <= r < 8:
                    p = occ.get(_square(f, r))
                    if p is not None:
                        if pinned is None:
                            if p.color != kcolor:
                                break                       # first piece is enemy → no pin
                            pinned = (_square(f, r), p)      # candidate friendly pinned piece
                        else:
                            if p.color != kcolor and p.piece in sliders:
                                facts.append(_pin_fact(pinned, _square(f, r), p, stm))
                            break                            # second piece decides it either way
                    f += dfile
                    r += drank
    return facts


def _pin_fact(pinned_pair, slider_sq, slider, stm) -> Fact:
    psq, pp = pinned_pair
    pn = PIECE_NAME.get(pp.piece, "piece")
    sn = PIECE_NAME.get(slider.piece, "piece")
    if pp.color == stm:
        text = (f"your {pn} on {psq} is pinned to your king by the {sn} on {slider_sq} "
                f"— it can't move off that line")
    else:
        text = (f"the {pn} on {psq} is pinned to the king by the {sn} on {slider_sq} "
                f"— it can't legally move off that line")
    return Fact(
        kind="pin",
        squares=[psq, slider_sq],
        text=text,
        provenance=f"pin:{psq}",
        salience=_salience(pp.piece),
        concept_id=CONCEPT,
    )
