"""Opening-name lookup — normalized-FEN → name, from the shipped table (built by
`tools/build_openings.py` from the Lichess chess-openings TSVs, CC0). A position that isn't a known
opening (e.g. any drill starting mid-game) returns None; the panel then simply shows no name.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# The table sits INSIDE the package (`lucena_engine/data/`), so it must be resolved relative to this
# module's own directory — `parent.parent` pointed one level above the package, at a path that exists
# in no install. `openings.tsv` is packaged, so the miss was pure path arithmetic.
_PATH = Path(__file__).resolve().parent / "data" / "openings.tsv"


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    table: dict[str, str] = {}
    try:
        for line in _PATH.read_text(encoding="utf-8").splitlines():
            fen, _, name = line.partition("\t")
            if name:
                table[fen] = name
    except OSError:
        # A missing table degrades to "no opening names" rather than breaking the coach — but say so.
        # Swallowing this silently is what let a wrong path ship: every lookup returned None, which is
        # indistinguishable from "not a known opening", so the feature was simply absent and nothing
        # ever complained.
        print(f"[openings] table not readable at {_PATH} — opening names disabled", flush=True)
    return table


@lru_cache(maxsize=1)
def _family_sizes() -> dict[str, int]:
    """Table rows per top-level family ("Ruy Lopez: Morphy Defense, Closed" -> "Ruy Lopez").

    A proxy for how much real continuation theory an opening actually has. Ruy Lopez has 234
    rows in the shipped table, Sicilian Defense 385 — deep trees where a position briefly
    dropping out of the table (the table names POSITIONS, not lines) is a coverage artifact, not
    a sign the players left theory. 57 of the table's 148 families have only 1-2 rows: a single
    catalogued position with no documented tree beyond it. `book_name`'s caller uses this to
    decide whether an unnamed-but-recent position should still speak with book-voice confidence.
    """
    counts: dict[str, int] = {}
    for name in _table().values():
        top = name.split(":", 1)[0].strip()
        counts[top] = counts.get(top, 0) + 1
    return counts


def family_size(name: str | None) -> int:
    """Table rows sharing `name`'s top-level family. 0 for a falsy or unrecognised name."""
    if not name:
        return 0
    return _family_sizes().get(name.split(":", 1)[0].strip(), 0)


from ._fen import norm_fen as _norm   # the one home for position identity


@lru_cache(maxsize=1)
def _ep_stripped_table() -> dict[str, str]:
    """A fallback index keyed WITHOUT the en-passant square.

    The table (and our board core) record ep on ANY double push, legal capture or not — 778 of 3733
    rows carry one, and `1.e4` is keyed with `e3`. Most of the chess world (Lichess, chess.com,
    python-chess) writes ep ONLY when a capture is actually available, so a FEN pasted from anywhere
    else misses every one of those rows — silently, because a miss is indistinguishable from "not an
    opening". Pasting a FEN is a first-class flow here, so this is a real loss, not a corner.

    Ambiguous keys are EXCLUDED rather than guessed: exactly one position is reachable under two
    names (a genuine transposition — Van Geet: Nowokunski Gambit / King's Gambit Accepted:
    Mason-Keres Gambit). Naming it from a stripped key would be a coin flip, so the fallback declines
    and that position stays exact-match-only.
    """
    by_stripped: dict[str, list[str]] = {}
    for key, name in _table().items():
        by_stripped.setdefault(" ".join(key.split()[:3]), []).append(name)
    return {k: v[0] for k, v in by_stripped.items() if len(v) == 1}


def name_for(fen: str) -> str | None:
    """The opening name for a position, or None if it isn't a known opening position.

    Exact first, then ep-stripped: our own FENs hit the exact index, and a foreign FEN written with
    the legal-only ep convention still resolves. See `_ep_stripped_table`.
    """
    key = _norm(fen)
    hit = _table().get(key)
    if hit is not None:
        return hit
    return _ep_stripped_table().get(" ".join(key.split()[:3]))


def _path(name: str) -> list[str]:
    """A name's ancestry, coarse→fine: "Ruy Lopez: Morphy Defense, Closed" →
    ["Ruy Lopez", "Morphy Defense", "Closed"]."""
    parts: list[str] = []
    for chunk in name.split(":"):
        parts += [p.strip() for p in chunk.split(",") if p.strip()]
    return parts


def _is_ancestor(candidate: str, current: str) -> bool:
    """Is `candidate` the same as, or a coarser form of, `current`?

    "Sicilian Defense" is an ancestor of "Sicilian Defense: Modern Variations". Used to ignore a
    name that would walk BACKWARDS up the tree — see `book_name`.
    """
    a, b = _path(candidate), _path(current)
    return len(a) <= len(b) and b[:len(a)] == a


def book_name(fens) -> str | None:
    """The opening this line is in, folded over the whole line — or None if it never entered the book.

    Not simply "the last named position": the table re-attaches COARSER names at deeper positions, so
    naive last-wins walks backwards up the tree. Measured on real mainlines:

        Najdorf: d6 -> "Sicilian Defense: Modern Variations", then d4 -> "Sicilian Defense"
        QGD:     Bg5 -> "...: Modern Variation",              then Be7 -> "Queen's Gambit Declined"
        French:  d4 -> "French Defense: Normal Variation",    then d5 -> "French Defense"

    A caller narrating on name-change would announce a LESS specific name than the one it just used —
    it reads as the coach forgetting. So a new name only REPLACES the current one when it is not an
    ancestor of it; a coarsening is ignored, a genuine branch (Ruy Lopez -> Morphy Defense, Indian ->
    Nimzo-Indian) replaces.

    Unnamed plies are sticky for free: the fold simply keeps the current name, so a line that dips out
    of the table for a ply or two (Ruy: Ba4, Nf6) stays in its opening.

    A pure function of the line, which is the point: callers detect a change with
    `book_name(hist) != book_name(hist[:-1])` and need no stored "last narrated" state — nothing to
    persist, nothing to resync after a restart.
    """
    current: str | None = None
    for fen in fens:
        n = name_for(fen)
        if n and not (current and _is_ancestor(n, current)):
            current = n
    return current


def plies_since_named(fens) -> int | None:
    """How many plies since the line was last in the book, or None if it never was.

    0 = the current position is itself named. A caller announces "end of book" when this reaches a
    threshold, which fires exactly once per exit with no latch state — and naturally re-arms if the
    line transposes back into the book.

    None (never named) means a line that did not start from the opening position — a pasted midgame
    FEN, a drill — where "you have left the book" is meaningless and must not be announced.
    """
    last = None
    for i, fen in enumerate(fens):
        if name_for(fen):
            last = i
    if last is None:
        return None
    return len(fens) - 1 - last
