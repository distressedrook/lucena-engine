"""FEN / board-truth string helpers lifted from the former coach layer (mcp/tools.py).

Pure functions over FEN text — position identity, rank expansion, the piece a move
captures (board truth, not the coach's guess), move-number/side, and PGN-numbered
line rendering. This is the ONE home for `_norm_fen`; the former duplicates in
`openings`/`poisoned_line_detector` import it from here.
"""

from __future__ import annotations

_PIECE_NAMES = {"p": "pawn", "n": "knight", "b": "bishop", "r": "rook", "q": "queen", "k": "king"}


def norm_fen(fen: str) -> str:
    """A position's identity ignoring clocks — placement + side + castling + ep (first 4 FEN fields).
    Lets "is the board at this move's position?" ignore halfmove/fullmove drift."""
    return " ".join(str(fen).split()[:4])


def row_squares(fen_row: str) -> list[str | None]:
    """Expand one FEN rank ('r1b1') into 8 squares (piece char or None). Board-core truth."""
    out: list[str | None] = []
    for ch in fen_row:
        if ch.isdigit():
            out.extend([None] * int(ch))
        else:
            out.append(ch)
    return out


def captured_piece(fen: str, uci: str) -> str | None:
    """Name the piece a move captures — 'bishop', 'pawn', … — from board-core truth, NOT the coach's
    guess (SAN like 'Qxf2' names the square, never the captured piece, so the model hallucinates it,
    usually 'pawn'). Handles en passant. Returns None when the move captures nothing."""
    try:
        rows = fen.split()[0].split("/")            # rows[0] = rank 8
        tgt_file = ord(uci[2]) - ord("a")           # 0..7
        tgt_rank = int(uci[3])                       # 1..8
        piece = row_squares(rows[8 - tgt_rank])[tgt_file]
        if piece:
            return _PIECE_NAMES.get(piece.lower())
        # En passant: a pawn moving diagonally onto an empty square captures the passed pawn.
        src_file = ord(uci[0]) - ord("a")
        mover = row_squares(rows[8 - int(uci[1])])[src_file]
        if mover and mover.lower() == "p" and src_file != tgt_file:
            return "pawn"
        return None
    except Exception:
        return None


def num(fen: str) -> tuple[int, bool]:
    """(move number, white_moved) for a move whose RESULTING position is `fen` — the mover is the
    side NOT to move; White's number is the fullmove, Black's is one less."""
    parts = str(fen).split()
    full = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1
    white_moved = (parts[1] if len(parts) > 1 else "w") == "b"
    return (full if white_moved else full - 1), white_moved


def pgn_line(moves: list[dict]) -> str:
    """Render a run of view moves ({san, fen}) as PGN-numbered text — e.g. "17.Rxd6 cxd6 18.Bxd6+".
    The number + side come from each move's resulting fen (fullmove ticks after Black; the mover is
    the side NOT to move). Black-led runs and mid-run Black moves read "17…" / bare figure."""
    out: list[str] = []
    for i, m in enumerate(moves):
        san = m.get("san")
        if not san:
            continue
        parts = str(m.get("fen", "")).split()
        full = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else 1
        white_moved = (parts[1] if len(parts) > 1 else "w") == "b"   # side-to-move is the NON-mover
        num_ = full if white_moved else full - 1
        if white_moved:
            out.append(f"{num_}.{san}")
        elif i == 0:
            out.append(f"{num_}...{san}")
        else:
            out.append(san)
    return " ".join(out)
