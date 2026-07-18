"""null-move threat detector — the killer Socratic fact (LLD §2.3).

"What if you just passed?" We compare the side-to-move's best reply against what
the *opponent* gets if handed a free move (a null move). The gap is the cost of
passing — i.e. the threat the side to move must answer:

    threat_swing = stm_win% + opp_free_win% - 100

where `stm_win%` is the side-to-move's win% with their best move and
`opp_free_win%` is the opponent's win% after the null move. A large positive
swing means passing is expensive: something is threatened. This isolates a
*threat* from merely *being worse* — a lost position has a high `opp_free_win%`
but so does its real best line, so the gap stays small.

This is the one M3 detector that needs the engine (two analyses: the real
position and the passed one). Search finds threats a single static capture
misses (a fork setup, a mate net, a defender deflected first) and prices them by
win%, not just SEE.

Determinism split: callers pass `nodes=` (tests, reproducible) or
`movetime_ms=` (production), forwarded verbatim to both `Engine.analyse` calls.
The pair is reproducible under the same rules as `Engine.analyse` (fresh engine
or `new_game()` first; nodes + threads=1).
"""

from __future__ import annotations

from ..evalmodel import win_pct_from_score
from ..facts import Fact
from ._util import PIECE_CP, PIECE_NAME, occupancy

CONCEPT = "tactical-signals"

# Minimum cost-of-passing (win% points) for the opponent's free move to count as
# a threat worth a fact.
MIN_WIN_PCT_SWING = 10.0

# Severity of a threat, measured by what happens to YOU if you ignore it: if
# passing drops your win% below _LOSING_CEIL while you are otherwise at least
# _SAFE_FLOOR (OK/winning), the threat flips the game — surface it loudly.
_LOSING_CEIL = 35.0
_SAFE_FLOOR = 45.0


def _salience(win_pct_swing: float, material_cp: int) -> float:
    # material threats lead; a purely positional swing still registers
    if material_cp > 0:
        base = 0.55 + min(0.40, material_cp / 2000.0)
    else:
        base = 0.50 + min(0.30, win_pct_swing / 100.0)
    return round(min(0.97, base), 3)


def detect_null_move_threat(
    board, engine, *, nodes: int | None = None, movetime_ms: int | None = None
) -> list[Fact]:
    try:
        passed = board.null_move()
    except ValueError:
        return []  # in check: passing is illegal, there is no null-move probe

    if not passed.legal_moves():
        return []  # passing would stalemate/leave the opponent no move — no threat

    real = engine.analyse(board.fen, nodes=nodes, movetime_ms=movetime_ms, multipv=1)
    stm_win = win_pct_from_score(real.best.score)  # side-to-move POV

    passed_an = engine.analyse(
        passed.fen, nodes=nodes, movetime_ms=movetime_ms, multipv=1
    )
    reply = passed_an.best
    if not reply.pv:
        return []
    opp_free_win = win_pct_from_score(reply.score)  # opponent POV after the pass

    swing = stm_win + opp_free_win - 100.0
    if swing < MIN_WIN_PCT_SWING:
        return []

    # is the threatening move a material grab? price it with SEE on `passed`.
    first = reply.pv[0]
    occ = occupancy(passed)
    dst = first[2:4]
    victim = occ.get(dst)
    material = 0
    victim_name = None
    if victim is not None and victim.color == board.side_to_move:
        if passed.see(first) > 0:
            material = PIECE_CP.get(victim.piece, 0)
            victim_name = PIECE_NAME.get(victim.piece, "piece")

    san = passed.san(first)
    mate = reply.score.mate                 # opponent's mate distance after the pass
    pass_win = 100.0 - opp_free_win         # YOUR win% if you ignore the threat

    # Absolute colours, never "you"/"the opponent" — relative words invert with side-to-move and were
    # the source of perspective flips (a fact carried onto the wrong player). _mover is the side that
    # would pass; _threat is the side that gets the free move.
    _mover = "White" if board.side_to_move == "white" else "Black"
    _threat = "Black" if _mover == "White" else "White"
    if mate is not None and mate > 0:       # a mate is brewing — the loudest threat
        if mate == 1:
            text = f"{_threat} threatens mate: {san}"
        elif mate <= 5:
            text = f"{_threat} threatens mate in {mate} — it starts with {san}"
        else:                               # too far out to recite a count — it's simply winning
            text = f"{_threat} is easily winning — the attack starts with {san}"
        salience = 0.98
    elif victim_name:                       # concrete material threat
        text = f"after a pass, {_threat} plays {san}, winning {_mover}'s {victim_name} on {dst}"
        salience = _salience(swing, material)
    elif pass_win <= _LOSING_CEIL and stm_win >= _SAFE_FLOOR:
        # not material, but ignoring it flips the game from OK/winning to losing
        standing = "winning" if stm_win >= 62.0 else "holding"
        text = f"warning — if {_mover} ignores {san}, {_mover} goes from {standing} to losing"
        salience = 0.95
    else:                                   # a real but non-flipping positional threat
        text = f"after a pass, {san} is strong for {_threat}"
        salience = _salience(swing, 0)

    return [
        Fact(
            kind="threat",
            squares=[first[:2], dst],
            text=text,
            provenance=f"nullmove:{first}",
            salience=salience,
            concept_id=CONCEPT,
        )
    ]
