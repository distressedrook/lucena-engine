"""Black-box tests for lucena_engine.evalmodel (M2 contract).

PURE: no engine, no I/O. Every numeric expectation is derived from the
contract's win% formula alone:

    win_pct(cp) = 50 + 50*(2/(1+exp(-0.00368208*cp)) - 1)      (cp clamped ±1000)

Boundary Scores for `classify` are built from a single identity. With
`best_from_mover = Score(cp=0)` (win% 50) and `played_result = Score(cp=P)`
(reported from the OPPONENT's POV, so classify negates it to mover-POV cp=-P):

    drop = best_wp - played_wp = 50 - win_pct(-P) = win_pct(P) - 50   (symmetry)

so the drop is a pure function of P. The exact bracketing values of P used
below were computed from the formula (arithmetic shown inline).
"""

import math

import pytest

from lucena_engine import Score, Glyph, classify, win_pct, win_pct_from_score

# parse_score is documented as importable from `lucena.engine` but is only
# exported from the submodule today (see test_parse_score_toplevel_export_gap).
from lucena_engine.evalmodel import parse_score


# --- an INDEPENDENT reference reimplementation of the contract formula, used
#     only to cross-check anchor points; never imported from the product.
def _ref_win_pct(cp: int) -> float:
    cp = max(-1000, min(1000, cp))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * cp)) - 1)


# ============================ win_pct ============================


def test_win_pct_zero_is_exactly_fifty():
    assert win_pct(0) == 50.0


def test_win_pct_positive_anchor():
    # formula: 100/(1+exp(-3.68208)) = 97.5447...
    assert win_pct(1000) == pytest.approx(97.5447, abs=1e-3)


def test_win_pct_negative_anchor():
    assert win_pct(-1000) == pytest.approx(2.4553, abs=1e-3)


def test_win_pct_symmetric_about_fifty():
    for x in (-777, -300, -1, 0, 1, 250, 999, 5000):
        assert win_pct(x) + win_pct(-x) == pytest.approx(100.0, abs=1e-9)


def test_win_pct_monotonic_increasing():
    vals = [win_pct(cp) for cp in range(-1200, 1201, 25)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    # strictly increasing inside the unclamped band
    inner = [win_pct(cp) for cp in range(-900, 901, 25)]
    assert all(b > a for a, b in zip(inner, inner[1:]))


def test_win_pct_clamps_beyond_1000():
    assert win_pct(5000) == win_pct(1000)
    assert win_pct(-5000) == win_pct(-1000)
    assert win_pct(10_000_000) == win_pct(1000)


def test_win_pct_matches_reference_across_range():
    for cp in range(-1500, 1501, 37):
        assert win_pct(cp) == pytest.approx(_ref_win_pct(cp), abs=1e-9)


def test_win_pct_bounded_in_unit_interval():
    for cp in (-999999, -1000, -1, 0, 1, 1000, 999999):
        assert 0.0 <= win_pct(cp) <= 100.0


# ============================ Score ============================


def test_score_requires_exactly_one_field_none():
    with pytest.raises(ValueError):
        Score()


def test_score_requires_exactly_one_field_both():
    with pytest.raises(ValueError):
        Score(cp=5, mate=2)


def test_score_cp_only_ok():
    s = Score(cp=42)
    assert s.cp == 42 and s.mate is None and s.is_mate is False


def test_score_mate_only_ok():
    s = Score(mate=3)
    assert s.mate == 3 and s.cp is None and s.is_mate is True


def test_score_mate_zero_is_valid_and_is_mate():
    # mate=0 means side-to-move is already checkmated (contract).
    s = Score(mate=0)
    assert s.is_mate is True
    assert s.mate == 0


def test_score_negated_cp():
    assert Score(cp=30).negated().cp == -30
    assert Score(cp=-30).negated().cp == 30


def test_score_negated_mate():
    assert Score(mate=3).negated().mate == -3
    assert Score(mate=-2).negated().mate == 2


def test_score_negated_mate_zero():
    # -0 == 0; a side already mated, flipped, is the side delivering mate-in-0.
    assert Score(mate=0).negated().mate == 0


def test_to_ceiled_cp_clamps_cp():
    assert Score(cp=5000).to_ceiled_cp() == 1000
    assert Score(cp=-5000).to_ceiled_cp() == -1000
    assert Score(cp=1000).to_ceiled_cp() == 1000
    assert Score(cp=-1000).to_ceiled_cp() == -1000
    assert Score(cp=20).to_ceiled_cp() == 20
    assert Score(cp=0).to_ceiled_cp() == 0


def test_to_ceiled_cp_mate_positive():
    assert Score(mate=1).to_ceiled_cp() == 1000
    assert Score(mate=99).to_ceiled_cp() == 1000


def test_to_ceiled_cp_mate_negative_and_zero():
    # mate<=0 (including 0) → -1000 per contract.
    assert Score(mate=-1).to_ceiled_cp() == -1000
    assert Score(mate=-50).to_ceiled_cp() == -1000
    assert Score(mate=0).to_ceiled_cp() == -1000


# ==================== win_pct_from_score ====================


def test_win_pct_from_score_cp():
    assert win_pct_from_score(Score(cp=0)) == 50.0
    assert win_pct_from_score(Score(cp=5000)) == win_pct(1000)


def test_win_pct_from_score_mate_positive():
    assert win_pct_from_score(Score(mate=3)) == pytest.approx(97.5447, abs=1e-3)


def test_win_pct_from_score_mate_negative():
    assert win_pct_from_score(Score(mate=-1)) == pytest.approx(2.4553, abs=1e-3)


def test_win_pct_from_score_mate_zero_is_lost():
    # mate=0 → to_ceiled_cp -1000 → ~2.46% (side is checkmated).
    assert win_pct_from_score(Score(mate=0)) == pytest.approx(2.4553, abs=1e-3)


# ============================ classify ============================
#
# helper: mover-POV `best` cp and opponent-POV `played` cp.
def _c(best_cp, played_cp, **kw):
    return classify(Score(cp=best_cp), Score(cp=played_cp), **kw)


def test_classify_glyph_enum_values():
    assert Glyph.OK.value == ""
    assert Glyph.DUBIOUS.value == "?!"
    assert Glyph.MISTAKE.value == "?"
    assert Glyph.BLUNDER.value == "??"
    assert Glyph.ONLY_MOVE.value == "!"


def test_classify_returns_glyph_and_float_drop():
    g, drop = _c(0, 0)
    assert isinstance(g, Glyph)
    assert isinstance(drop, float)


# --- DUBIOUS threshold: drop >= 5 --------------------------------
# win_pct(54)-50 = 4.95446  (< 5)  → OK
# win_pct(55)-50 = 5.04563  (>= 5) → DUBIOUS
def test_classify_below_dubious_boundary_is_ok():
    g, drop = _c(0, 54)
    assert g is Glyph.OK
    assert drop == pytest.approx(4.9545, abs=1e-3)


def test_classify_at_dubious_boundary():
    g, drop = _c(0, 55)
    assert g is Glyph.DUBIOUS
    assert drop == pytest.approx(5.0456, abs=1e-3)
    assert drop >= 5.0


# --- MISTAKE threshold: drop >= 10 -------------------------------
# win_pct(110)-50 = 9.98953 (< 10, this is the "9.99" case) → DUBIOUS
# win_pct(111)-50 = 10.07787 (>= 10)                        → MISTAKE
def test_classify_just_below_mistake_boundary_is_dubious():
    g, drop = _c(0, 110)
    assert g is Glyph.DUBIOUS
    assert drop == pytest.approx(9.9895, abs=1e-3)
    assert drop < 10.0


def test_classify_at_mistake_boundary():
    g, drop = _c(0, 111)
    assert g is Glyph.MISTAKE
    assert drop == pytest.approx(10.0779, abs=1e-3)
    assert drop >= 10.0


# --- BLUNDER threshold: drop >= 15 -------------------------------
# win_pct(168)-50 = 14.98977 (< 15) → MISTAKE
# win_pct(169)-50 = 15.07350 (>= 15) → BLUNDER
def test_classify_just_below_blunder_boundary_is_mistake():
    g, drop = _c(0, 168)
    assert g is Glyph.MISTAKE
    assert drop == pytest.approx(14.9898, abs=1e-3)
    assert drop < 15.0


def test_classify_at_blunder_boundary():
    g, drop = _c(0, 169)
    assert g is Glyph.BLUNDER
    assert drop == pytest.approx(15.0735, abs=1e-3)
    assert drop >= 15.0


def test_classify_negative_drop_clamped_to_zero():
    # played is BETTER for the mover than "best": opponent-POV cp=-100 →
    # mover-POV +100 → played_wp>50 → raw drop negative → clamp to 0, glyph OK.
    g, drop = _c(0, -100)
    assert g is Glyph.OK
    assert drop == 0.0


def test_classify_perspective_opponent_winning_is_blunder():
    # played_result reported from opponent POV as clearly winning FOR the
    # opponent (cp=+800). classify negates → mover-POV -800 → huge drop.
    g, drop = _c(0, 800)
    assert g is Glyph.BLUNDER
    # win_pct(800)-50 = 45.0058
    assert drop == pytest.approx(45.0058, abs=1e-3)


def test_classify_mate_played_result_is_blunder():
    # opponent-POV mate in 2 (opponent mates mover) → catastrophic drop.
    g, drop = classify(Score(cp=0), Score(mate=2))
    assert g is Glyph.BLUNDER
    assert drop == pytest.approx(win_pct(0) - win_pct(-1000), abs=1e-3)


# --- ONLY_MOVE (!) rule -----------------------------------------
# best cp=0 (wp 50). second_best mover-POV cp=-169 → wp 34.9265;
# gap = 50 - 34.9265 = 15.0735 (>= 15) → ! eligible.
# second_best cp=-168 → gap 14.9898 (< 15) → not eligible.
def test_only_move_fires():
    g, drop = classify(
        Score(cp=0),
        Score(cp=0),  # best move played → drop 0
        second_best_from_mover=Score(cp=-169),
        played_is_best=True,
    )
    assert g is Glyph.ONLY_MOVE
    assert drop == 0.0


def test_only_move_not_fired_when_gap_under_15():
    g, drop = classify(
        Score(cp=0),
        Score(cp=0),
        second_best_from_mover=Score(cp=-168),  # gap 14.99 < 15
        played_is_best=True,
    )
    assert g is Glyph.OK


def test_only_move_suppressed_by_real_error():
    # A genuine error (drop >= 5) always outranks !. Here played_result
    # opponent-POV cp=169 → drop 15.07 (BLUNDER) despite the wide gap.
    g, drop = classify(
        Score(cp=0),
        Score(cp=169),
        second_best_from_mover=Score(cp=-169),
        played_is_best=True,
    )
    assert g is Glyph.BLUNDER
    assert drop == pytest.approx(15.0735, abs=1e-3)


def test_only_move_suppressed_when_not_best():
    g, _ = classify(
        Score(cp=0),
        Score(cp=0),
        second_best_from_mover=Score(cp=-169),
        played_is_best=False,  # must be True for !
    )
    assert g is Glyph.OK


def test_only_move_suppressed_when_second_best_none():
    g, _ = classify(
        Score(cp=0),
        Score(cp=0),
        second_best_from_mover=None,  # required for !
        played_is_best=True,
    )
    assert g is Glyph.OK


# ============================ parse_score ============================


def test_parse_score_cp():
    s = parse_score("cp", 42)
    assert s.cp == 42 and s.mate is None


def test_parse_score_mate():
    s = parse_score("mate", -3)
    assert s.mate == -3 and s.cp is None


def test_parse_score_bad_token():
    with pytest.raises(ValueError):
        parse_score("centipawns", 5)
    with pytest.raises(ValueError):
        parse_score("", 5)


def test_parse_score_toplevel_export():
    # Reconciled: the export gap Agent A flagged is fixed — parse_score is
    # now re-exported at the package level as the contract promises.
    from lucena_engine import parse_score as _ps

    assert _ps("cp", 5) == Score(cp=5)
