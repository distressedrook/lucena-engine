"""Shared primitives for the motif detectors.

Everything here is board-core only (lucena_board via `lucena.board.Board`) —
no engine, no python-chess. GPL hygiene holds (LLD §2.0).
"""

from __future__ import annotations

# Detectors receive a `lucena.board.Board` instance (duck-typed) — nothing here
# constructs one, so there is no import to make and no cycle to worry about.

# Piece letters are SF/FEN convention: upper = white, lower = black. Detectors
# receive `Board.PieceOn.piece` which is already colour-stripped (P N B R Q K).
PIECE_NAME = {
    "P": "pawn",
    "N": "knight",
    "B": "bishop",
    "R": "rook",
    "Q": "queen",
    "K": "king",
}

# Rough material values in centipawns — used ONLY for salience shaping and human
# text ("wins the knight"). The material *verdict* is always SEE, never this map.
PIECE_CP = {"P": 100, "N": 300, "B": 300, "R": 500, "Q": 900, "K": 0}


def opponent(color: str) -> str:
    return "black" if color == "white" else "white"


def piece_at(board, square: str):
    """Return the PieceOn on `square`, or None if empty."""
    for p in board.piece_list():
        if p.square == square:
            return p
    return None


def occupancy(board) -> dict[str, object]:
    """square -> PieceOn for every occupied square (one piece_list scan)."""
    return {p.square: p for p in board.piece_list()}


def king_square(board, color: str) -> str | None:
    for p in board.piece_list():
        if p.piece == "K" and p.color == color:
            return p.square
    return None


def forked_targets(after, to_sq: str, mover: str) -> list[tuple[str, str, int]]:
    """The valuable enemy pieces the piece now on `to_sq` attacks in the
    position `after` its move — the geometry under a fork/double attack.

    Returns `(square, piece_letter, n_defenders)` for every enemy piece of
    value ≥ knight (the king is excluded — it is the *check* prong, never a
    won piece), most-valuable first. Shared by the fork detector and the hint
    ladder so there is exactly one definition of "what this piece hits."
    """
    enemy = opponent(mover)
    hits: list[tuple[str, str, int]] = []
    for p in after.piece_list():
        if p.color != enemy or p.piece == "K":
            continue
        if PIECE_CP.get(p.piece, 0) < PIECE_CP["N"]:
            continue
        if to_sq in after.attackers(p.square, mover):
            hits.append((p.square, p.piece, len(after.defenders(p.square))))
    hits.sort(key=lambda t: -PIECE_CP.get(t[1], 0))
    return hits


def captures_by_side_to_move(board, occ: dict | None = None):
    """Yield (uci, target_square, victim PieceOn) for every legal *capture*.

    A legal move is a capture when its destination holds an enemy piece, or it
    is an en-passant pawn capture (pawn changes file onto an empty square).
    """
    if occ is None:
        occ = occupancy(board)
    stm = board.side_to_move
    for m in board.legal_moves():
        src, dst = m[:2], m[2:4]
        victim = occ.get(dst)
        if victim is not None and victim.color != stm:
            yield m, dst, victim
            continue
        # en passant: a pawn moving diagonally onto an empty square
        mover = occ.get(src)
        if (
            mover is not None
            and mover.piece == "P"
            and src[0] != dst[0]
            and dst not in occ
        ):
            ep_sq = dst[0] + src[1]  # the captured pawn sits on the mover's rank
            yield m, ep_sq, occ.get(ep_sq)
