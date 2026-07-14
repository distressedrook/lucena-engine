"""combination detector — "there's a forcing win here" (LLD §2.3, multi-move).

The M3 trio and the fork detector are 1–2 ply: they surface *immediate* motifs.
A multi-move combination — a forced mate, or a forcing sequence that wins
decisively — lives past their horizon, in the engine's principal variation. On
those positions the sheet used to lead with a shallow distractor ("the opponent
threatens X") while the actual win sat unmentioned in `lines[0]`.

This detector reads the engine verdict directly and, when the best line is a
**forced mate** or a **forcing decisive shot** (a check or a sound sacrifice that
is clearly better than every alternative), emits one high-salience fact naming
the entry move — so the coach leads with the combination, not the distractor. It
does *not* spell the whole line (the PV/arrow carry that); it flags that a
forcing win exists and where it starts. Being the engine's own verdict, it needs
no reconciliation. Concept: `tactical-signals`.
"""

from __future__ import annotations

from ..evalmodel import win_pct_from_score
from ..facts import Fact

CONCEPT = "tactical-signals"

_DECISIVE_WIN = 80.0     # win% at which a forcing line counts as "winning"
_GAP_TO_SECOND = 15.0    # how much better than the alternative it must be to be *the* shot


def detect_combination(board, engine, *, nodes=None, movetime_ms=None) -> list[Fact]:
    a = engine.analyse(board.fen, nodes=nodes, movetime_ms=movetime_ms, multipv=2)
    best = a.best
    if not best.pv:
        return []
    first = best.pv[0]
    san = board.san(first)
    squares = [first[:2], first[2:4]]

    # -- forced mate for the side to move ---------------------------------
    mate = best.score.mate
    if mate is not None and mate > 0:
        return [Fact(
            kind="combination", squares=squares,
            text=f"there's a forced mate in {mate} — it starts with {san}",
            provenance=f"mate:{first}", salience=0.99, concept_id=CONCEPT,
        )]

    # -- forcing decisive (non-mate) combination --------------------------
    win = win_pct_from_score(best.score)
    second = win_pct_from_score(a.lines[1].score) if len(a.lines) > 1 else 0.0
    gives_check = board.apply(first).in_check
    is_sac = board.see(first) < 0
    forcing = gives_check or is_sac  # a plain winning capture is the hanging fact, not this
    if forcing and win >= _DECISIVE_WIN and (win - second) >= _GAP_TO_SECOND:
        how = "sacrifices material but wins" if is_sac else "wins by force"
        return [Fact(
            kind="combination", squares=squares,
            text=f"a forcing sequence starting with {san} {how}",
            provenance=f"combo:{first}", salience=0.95, concept_id=CONCEPT,
        )]
    return []
