"""Deterministic detection of a position pasted into free text — FEN or PGN.

The board must never move to a position the *model* produced (one wrong character is a
hallucinated board). So a player who pastes a FEN or PGN into chat has it recognised and
validated HERE, deterministically, by the board core and the PGN parser — never by an LLM
`set_board`. Detection is structure (regex candidates) + semantics (the parser rejects the
impossible): a rank that doesn't sum to 8, an illegal SAN, a king-less board all fail.

Lives in the open core because it uses `board.Board` + `pgn.parse_pgn`; the closed runner
reaches it only over the protocol.
"""
from __future__ import annotations

import re

from .board import Board
from .pgn import Game, PgnError, parse_pgn

# ---- FEN -------------------------------------------------------------------

_FEN_RE = re.compile(
    r'([pnbrqkPNBRQK1-8]{1,8}(?:/[pnbrqkPNBRQK1-8]{1,8}){7})'   # placement (8 ranks)
    r'\s+([wb])'                                                 # side to move
    r'\s+(-|[KQkq]{1,4})'                                        # castling
    r'\s+(-|[a-h][36])'                                          # en passant
    r'(?:\s+(\d+)\s+(\d+))?'                                     # halfmove / fullmove (optional)
)


def detect_fens(text: str) -> list[str]:
    """Every legal FEN in `text`, normalised (default move counters filled in). A FEN-shaped
    run that isn't a legal position (bad rank sum, no king, …) is dropped by the board core."""
    out: list[str] = []
    for m in _FEN_RE.finditer(text or ""):
        placement, side, castle, ep, half, full = m.groups()
        fen = f"{placement} {side} {castle} {ep} {half or '0'} {full or '1'}"
        try:
            Board(fen)
            if fen not in out:
                out.append(fen)
        except Exception:
            pass
    return out


# ---- PGN -------------------------------------------------------------------

# One PGN movetext token: a move number ("12." / "12..."), a SAN move (with optional
# check/mate and !/? annotations), a game result, or a NAG.
_SAN = (r'(?:O-O-O|O-O|'
        r'[KQRBN][a-h1-8]?x?[a-h][1-8]|'          # piece move
        r'[a-h]x?[a-h]?[1-8](?:=[QRBN])?)'         # pawn move / promotion
        r'[+#]?[!?]*')
_PGN_TOKEN = re.compile(r'^(?:\d+\.+|' + _SAN + r'|1-0|0-1|1/2-1/2|\*|\$\d+|\.\.\.)$')
_HAS_HEADER = re.compile(r'\[\s*\w+\s+"[^"]*"\s*\]')


def _movetext_span(text: str) -> str | None:
    """The longest contiguous run of whitespace-separated PGN tokens in `text` — the pasted
    movetext with the surrounding prose ("what's best after …?") stripped off."""
    toks = text.split()
    best: list[str] = []
    cur: list[str] = []
    for t in toks:
        if _PGN_TOKEN.match(t):
            cur.append(t)
            if len(cur) > len(best):
                best = cur[:]
        else:
            cur = []
    # need at least a move number + one move to be worth parsing
    return " ".join(best) if len(best) >= 2 else None


def detect_pgn(text: str) -> Game | None:
    """The game pasted in `text`, or None. A full PGN (with `[Tag "…"]` headers) parses
    directly; bare movetext is extracted first. Legality is the parser's job — an illegal
    move makes it None, never a wrong board."""
    text = text or ""
    if _HAS_HEADER.search(text):
        try:
            g = parse_pgn(text)
            if g.plies:
                return g
        except PgnError:
            pass
    span = _movetext_span(text)
    if span:
        try:
            g = parse_pgn(span)
            if g.plies:
                return g
        except PgnError:
            pass
    return None
