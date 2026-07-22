"""Parser for Stockfish 18's `eval` command — per-square NNUE piece values.

Classical evaluation terms were removed in SF 16.1; the `eval` command's
per-square NNUE piece-value board is their replacement and the source of
piece-quality facts ("the trade walled your bishop in"). See LLD-analysis
§2.3. The output is a debug format with no stability guarantee, so the
engine version is pinned (uci.PINNED_MAJOR) and the parser is snapshot-tested.

Grid layout (verified SF 18): the block after "NNUE derived piece values:"
prints 8 board ranks top-down (rank 8 first), files a-h left-to-right, each
rank as a piece-letter row then a value row, cells `|`-delimited. Values are
in pawns from White's perspective (positive = good for White).
"""

from __future__ import annotations

from dataclasses import dataclass

_HEADER = "NNUE derived piece values"
_FILES = "abcdefgh"


@dataclass(frozen=True)
class PieceValue:
    square: str          # e.g. "e5"
    piece: str           # upper = white, lower = black (as SF prints it)
    value: float | None  # pawns, White's POV (+ = good for White); None for kings


def _split_cells(row: str) -> list[str]:
    # "|   r   |       | ..." → 8 trimmed cell strings
    parts = row.split("|")[1:-1]  # drop before-first and after-last
    return [p.strip() for p in parts]


def parse_piece_values(eval_lines: list[str]) -> list[PieceValue]:
    """Extract per-square NNUE piece values from `eval` output.

    Raises ValueError if the grid is absent or malformed (a version drift or
    a non-`eval` input) — never returns partial garbage.
    """
    # Locate the grid.
    start = next((i for i, l in enumerate(eval_lines) if _HEADER in l), None)
    if start is None:
        raise ValueError("no NNUE piece-value grid found (wrong SF version / not eval output)")

    # Content rows start with '|'; separator rows start with '+' (a value
    # cell may itself contain '+', e.g. "+0.30", so filter on the first
    # non-space char, not on the presence of '+'). Stop at the first blank
    # line after the grid so the later "network contributions" table (also
    # '|'-delimited) is never consumed.
    content: list[str] = []
    for line in eval_lines[start + 1:]:
        s = line.strip()
        if s.startswith("|"):
            content.append(line)
            if len(content) == 16:
                break
        elif not s and content:
            break  # blank line ends the grid block
    # Two rows per rank (pieces, values); 8 ranks → 16 rows.
    if len(content) != 16:
        raise ValueError(f"expected 16 grid rows, found {len(content)}")

    out: list[PieceValue] = []
    for rank_idx in range(8):
        rank_no = 8 - rank_idx  # first printed rank is 8
        pieces = _split_cells(content[rank_idx * 2])
        values = _split_cells(content[rank_idx * 2 + 1])
        if len(pieces) != 8 or len(values) != 8:
            raise ValueError(f"malformed grid row at rank {rank_no}")
        for file_idx in range(8):
            p = pieces[file_idx]
            v = values[file_idx]
            if not p and not v:
                continue  # empty square
            square = f"{_FILES[file_idx]}{rank_no}"
            if p in ("K", "k") and not v:
                # Kings carry no NNUE piece value by design; record as None so
                # callers can place the king but never treat it as tradeable.
                out.append(PieceValue(square=square, piece=p, value=None))
                continue
            if not p or not v:
                raise ValueError(f"piece/value mismatch at rank {rank_no} file {file_idx}")
            out.append(PieceValue(square=square, piece=p, value=float(v)))
    return out


def piece_value_map(eval_lines: list[str]) -> dict[str, PieceValue]:
    """Same data keyed by square, for O(1) diffing across a move."""
    return {pv.square: pv for pv in parse_piece_values(eval_lines)}
