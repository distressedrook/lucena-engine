"""Lichess-verbatim evaluation semantics — pure, no engine dependency.

The win-probability model and move-classification thresholds are copied from
MIT-licensed scalachess (`core/.../eval.scala`, `Advice.scala`), not
approximated. See docs/research/lichess-ecosystem-reuse.md for provenance.

Design note on mate handling: a mate score collapses to ±1000 cp regardless
of distance, and classification then uses uniform win%-deltas. This single
rule reproduces every "special mate table" behavior:
  * missed mate  (±1000 → +300)  → large drop → blunder
  * allowed mate (+200 → ∓1000)  → large drop → blunder
  * delayed mate (mate-5 → mate-8, both ±1000) → zero drop → NOT punished
  * faster mate  (mate-5 → mate-2, both ±1000) → zero drop → not flagged
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

# scalachess constants
_MULTIPLIER = -0.00368208
_CP_CEIL = 1000  # cp clamp AND mate-collapse magnitude

# Glyph thresholds in win-% points (winningChances deltas .05/.10/.15 → these,
# since winningChances ∈ [-1,1] is 2×(win%/100 − .5), so a .10 delta = 5 pts).
_DUBIOUS = 5.0   # ?!  (inaccuracy)
_MISTAKE = 10.0  # ?
_BLUNDER = 15.0  # ??
_ONLY_MOVE_GAP = 15.0  # best vs 2nd-best drop for a "!" only-move


class Glyph(str, Enum):
    OK = ""
    DUBIOUS = "?!"
    MISTAKE = "?"
    BLUNDER = "??"
    ONLY_MOVE = "!"


@dataclass(frozen=True)
class Score:
    """An engine score from the side-to-move's perspective.

    Exactly one of `cp` / `mate` is set. `mate` is signed moves-to-mate:
    positive = side to move delivers mate, negative = side to move is mated.
    `mate=0` means the side to move is already checkmated (terminal).
    """

    cp: int | None = None
    mate: int | None = None

    def __post_init__(self):
        if (self.cp is None) == (self.mate is None):
            raise ValueError("Score needs exactly one of cp / mate")

    @property
    def is_mate(self) -> bool:
        return self.mate is not None

    def negated(self) -> "Score":
        """Same position from the other side's perspective."""
        if self.cp is not None:
            return Score(cp=-self.cp)
        return Score(mate=-self.mate)

    def to_ceiled_cp(self) -> int:
        """Collapse to a bounded centipawn value (mate → ±1000)."""
        if self.mate is not None:
            if self.mate == 0:
                return -_CP_CEIL  # side to move is mated
            return _CP_CEIL if self.mate > 0 else -_CP_CEIL
        return max(-_CP_CEIL, min(_CP_CEIL, self.cp))


def win_pct(cp: int) -> float:
    """Win% [0,100] for a centipawn eval from the scored side's POV.

    cp is ceiled to ±1000 before the logistic (lichess convention).
    """
    c = max(-_CP_CEIL, min(_CP_CEIL, cp))
    return 50.0 + 50.0 * (2.0 / (1.0 + math.exp(_MULTIPLIER * c)) - 1.0)


def win_pct_from_score(score: Score) -> float:
    """Win% [0,100] from the scored side's POV, mate-aware."""
    return win_pct(score.to_ceiled_cp())


def _win_pct_mover(best_from_mover: Score, played_result: Score) -> tuple[float, float]:
    """(best win%, played win%) both from the MOVER's perspective.

    `best_from_mover` is the eval before the move (mover to move).
    `played_result` is the eval AFTER the move, as reported for the new side
    to move (the opponent) — so it is negated to the mover's POV.
    """
    return win_pct_from_score(best_from_mover), win_pct_from_score(played_result.negated())


def classify(
    best_from_mover: Score,
    played_result: Score,
    *,
    second_best_from_mover: Score | None = None,
    played_is_best: bool = False,
) -> tuple[Glyph, float]:
    """Classify a played move by win-% drop from the mover's perspective.

    Returns (glyph, win_pct_drop). Drop is clamped at 0 (a move that improves
    on the engine's line — possible from shallow pre-move analysis — is OK,
    never negative-drop noise).

    `!` (only-move) is awarded when the played move IS the engine best AND the
    second-best drops by ≥ 15 win-% points (i.e. everything else meaningfully
    loses). `!` takes precedence over OK but never masks a real error.
    """
    best_wp, played_wp = _win_pct_mover(best_from_mover, played_result)
    drop = max(0.0, best_wp - played_wp)

    if drop >= _BLUNDER:
        return Glyph.BLUNDER, drop
    if drop >= _MISTAKE:
        return Glyph.MISTAKE, drop
    if drop >= _DUBIOUS:
        return Glyph.DUBIOUS, drop

    if played_is_best and second_best_from_mover is not None:
        second_wp = win_pct_from_score(second_best_from_mover)
        if (best_wp - second_wp) >= _ONLY_MOVE_GAP:
            return Glyph.ONLY_MOVE, drop
    return Glyph.OK, drop


def parse_score(token: str, value: int) -> Score:
    """Build a Score from a UCI `score` field: token ∈ {'cp','mate'}."""
    if token == "cp":
        return Score(cp=value)
    if token == "mate":
        return Score(mate=value)
    raise ValueError(f"unknown score token: {token}")


# Public alias (2026-07-23, core-extraction Phase 2): consumers outside this
# package (lucena_core, lucena-tactics' facts port) need the mistake threshold;
# reaching for the private name was always a wart (facts.py did it).
MISTAKE = _MISTAKE
