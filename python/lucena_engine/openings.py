"""Opening-name lookup — normalized-FEN → name, from the shipped table (built by
`tools/build_openings.py` from the Lichess chess-openings TSVs, CC0). A position that isn't a known
opening (e.g. any drill starting mid-game) returns None; the panel then simply shows no name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "data" / "openings.tsv"


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    table: dict[str, str] = {}
    try:
        for line in _PATH.read_text(encoding="utf-8").splitlines():
            fen, _, name = line.partition("\t")
            if name:
                table[fen] = name
    except OSError:
        pass
    return table


from ._fen import norm_fen as _norm   # the one home for position identity


def name_for(fen: str) -> str | None:
    """The opening name for a position, or None if it isn't a known opening position."""
    return _table().get(_norm(fen))


def deepest(fens) -> str | None:
    """The name of the deepest position in a game line that's a known opening, or None."""
    best = None
    for fen in fens:
        n = name_for(fen)
        if n:
            best = n
    return best
