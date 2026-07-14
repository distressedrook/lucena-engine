"""Ergonomic wrapper over the Rust board core (lucena_board).

The one place product code touches board logic. GPL hygiene: python-chess is
never imported here or anywhere under python/lucena/ (see LICENSE).
"""

from __future__ import annotations

from dataclasses import dataclass

from . import lucena_board as _rs

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass(frozen=True)
class PieceOn:
    square: str
    piece: str  # P N B R Q K
    color: str  # white | black


class Board:
    """Immutable position; every mutation returns a new Board.

    Public API includes `.fen` (canonical FEN; en-passant square is emitted
    whenever a double push just occurred, cozy-chess convention). Moves are
    accepted as UCI or SAN everywhere; detection is by shape (4-5 chars of
    coordinates = UCI), so UCI-shaped-but-illegal strings raise the UCI
    error, not the SAN no-match error.
    """

    def __init__(self, fen: str = STARTPOS):
        # Validate eagerly so a bad FEN fails at construction, not first use.
        _rs.side_to_move(fen)
        self.fen = fen

    # -- reads ------------------------------------------------------------
    @property
    def side_to_move(self) -> str:
        return _rs.side_to_move(self.fen)

    @property
    def in_check(self) -> bool:
        return _rs.is_check(self.fen)

    def piece_list(self) -> list[PieceOn]:
        return [PieceOn(sq, p, c) for sq, p, c in _rs.piece_list(self.fen)]

    def legal_moves(self) -> list[str]:
        """Standard UCI, castling as e1g1."""
        return _rs.legal_moves(self.fen)

    def attackers(self, square: str, color: str) -> list[str]:
        """Squares of `color` pieces attacking `square` (full occupancy).

        Sorted for determinism. Full occupancy means batteries do NOT appear:
        a rook behind a rook is not an attacker here (x-rays are a SEE
        concept, not an attackers() concept).
        """
        return sorted(_rs.attackers(self.fen, square, color))

    def defenders(self, square: str) -> list[str]:
        """Squares of same-color pieces defending the piece on `square`."""
        occupant_color = next(
            (p.color for p in self.piece_list() if p.square == square), None
        )
        if occupant_color is None:
            raise ValueError(f"no piece on {square}")
        return self.attackers(square, occupant_color)

    # -- conversions -------------------------------------------------------
    def san(self, uci: str) -> str:
        return _rs.uci_to_san(self.fen, uci)

    def uci(self, san: str) -> str:
        return _rs.san_to_uci(self.fen, san)

    # -- judgments ----------------------------------------------------------
    def see(self, move: str) -> int:
        """Static exchange eval in centipawns; accepts UCI or SAN."""
        return _rs.see_move(self.fen, self._as_uci(move))

    # -- mutations (returning new boards) -----------------------------------
    def apply(self, move: str) -> "Board":
        return Board(_rs.apply_uci(self.fen, self._as_uci(move)))

    def null_move(self) -> "Board":
        """Side to move passes. Raises ValueError if in check."""
        return Board(_rs.null_move_fen(self.fen))

    # -- internals -----------------------------------------------------------
    def _as_uci(self, move: str) -> str:
        if (
            len(move) in (4, 5)
            and move[0] in "abcdefgh"
            and move[1] in "12345678"
            and move[2] in "abcdefgh"
            and move[3] in "12345678"
        ):
            return move
        return self.uci(move)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Board({self.fen!r})"
