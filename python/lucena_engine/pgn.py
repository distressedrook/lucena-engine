"""PGN parsing — pure-Python envelope over the board core (LLD §3.1 step 1).

GPL hygiene (LLD §2.0): python-chess is never imported. We tokenize the PGN
*text* ourselves (headers, movetext, comments, variations, NAGs) and hand every
SAN move to the board core (`Board.uci` / `Board.apply`) for the actual chess —
so all legality/legibility comes from the MIT engine, never a GPL library.

Scope: one game per call (multi-game intake is orchestrated a level up). A
`[FEN]` + `[SetUp "1"]` header pair sets the start position; otherwise the game
starts from the standard position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .board import Board, STARTPOS


class PgnError(ValueError):
    """Malformed PGN, or a move that is illegal in its position."""


@dataclass(frozen=True)
class Ply:
    """One half-move, with the positions on either side of it."""

    ply: int          # 1-based half-move index
    move_no: int      # full-move number (1, 1, 2, 2, …)
    side: str         # "w" | "b"
    san: str          # as it should be rendered (board-core canonical SAN)
    uci: str          # "e2e4", "e7e8q"
    fen_before: str
    fen_after: str


@dataclass(frozen=True)
class Game:
    headers: dict[str, str]
    plies: list[Ply]
    result: str        # "1-0" | "0-1" | "1/2-1/2" | "*"
    start_fen: str


_HEADER_RE = re.compile(r'^\s*\[\s*(\w+)\s+"((?:[^"\\]|\\.)*)"\s*\]\s*$')
_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}


def _split_headers(text: str) -> tuple[dict[str, str], str]:
    """Pull `[Tag "value"]` lines off the front; return (headers, movetext)."""
    headers: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = _HEADER_RE.match(line)
        if not m:
            break
        headers[m.group(1)] = m.group(2).replace('\\"', '"').replace("\\\\", "\\")
        i += 1
    return headers, "\n".join(lines[i:])


def _strip_movetext(movetext: str) -> str:
    """Remove comments `{…}`, rest-of-line comments `;…`, variations `(…)`,
    and NAG glyphs `$n`, leaving a flat token stream. Nested braces/parens are
    handled by counting depth."""
    out = []
    depth_brace = 0
    depth_paren = 0
    i = 0
    n = len(movetext)
    while i < n:
        c = movetext[i]
        if depth_brace:
            if c == "}":
                depth_brace -= 1
            i += 1
            continue
        if c == "{":
            depth_brace += 1
            i += 1
            continue
        if c == ";":  # rest-of-line comment
            j = movetext.find("\n", i)
            i = n if j == -1 else j + 1
            continue
        if c == "(":
            depth_paren += 1
            i += 1
            continue
        if c == ")":
            if depth_paren == 0:
                raise PgnError("unbalanced ')' in movetext")
            depth_paren -= 1
            i += 1
            continue
        if depth_paren:
            i += 1
            continue
        out.append(c)
        i += 1
    if depth_brace:
        raise PgnError("unterminated comment '{' in movetext")
    if depth_paren:
        raise PgnError("unterminated variation '(' in movetext")
    return "".join(out)


# A move token: strip a leading move number ("12." / "12...") if fused, then the
# SAN itself. NAGs ($n) and move-number-only tokens are dropped by the caller.
_MOVENUM_RE = re.compile(r"^\d+\.(\.\.)?")
_NAG_RE = re.compile(r"^\$\d+$")


def _san_tokens(flat: str) -> list[str]:
    tokens: list[str] = []
    for raw in flat.split():
        if raw in _RESULTS or _NAG_RE.match(raw):
            continue
        tok = _MOVENUM_RE.sub("", raw)  # "12.Nf3" -> "Nf3"; "12." -> ""
        if not tok:
            continue
        tokens.append(tok)
    return tokens


def parse_pgn(text: str) -> Game:
    """Parse a single PGN game. Raises `PgnError` on malformed text or an
    illegal move. The returned `plies` walk the game from `start_fen`."""
    if not text or not text.strip():
        raise PgnError("empty PGN")

    headers, movetext = _split_headers(text)

    start_fen = STARTPOS
    if headers.get("SetUp") == "1" and "FEN" in headers:
        start_fen = headers["FEN"]
    try:
        board = Board(start_fen)
    except Exception as e:
        raise PgnError(f"invalid start position: {e}") from e

    tokens = _san_tokens(_strip_movetext(movetext))

    plies: list[Ply] = []
    for idx, san in enumerate(tokens):
        side = "w" if board.side_to_move == "white" else "b"
        move_no = _move_number(start_fen, idx)
        try:
            uci = board.uci(san)
            after = board.apply(uci)
        except Exception as e:
            raise PgnError(
                f"illegal move #{idx + 1} {san!r} at ply {idx + 1}: {e}"
            ) from e
        plies.append(
            Ply(
                ply=idx + 1,
                move_no=move_no,
                side=side,
                san=board.san(uci),
                uci=uci,
                fen_before=board.fen,
                fen_after=after.fen,
            )
        )
        board = after

    result = _result(headers, movetext)
    return Game(headers=headers, plies=plies, result=result, start_fen=start_fen)


def _move_number(start_fen: str, ply_index: int) -> int:
    """Full-move number for the ply_index-th half-move (0-based) from start."""
    parts = start_fen.split()
    start_no = int(parts[5]) if len(parts) >= 6 and parts[5].isdigit() else 1
    white_to_move = (len(parts) >= 2 and parts[1] == "w")
    # offset of the first ply within its full move: 0 if white starts, else 1
    first_offset = 0 if white_to_move else 1
    return start_no + (ply_index + first_offset) // 2


def _result(headers: dict[str, str], movetext: str) -> str:
    for tok in reversed(movetext.split()):
        if tok in _RESULTS:
            return tok
    return headers.get("Result", "*")
