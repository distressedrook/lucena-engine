"""build_analysis — the deterministic natural-language grounded briefing.

Engine-backed (eval + fact sheet), so nodes-limit determinism, threads=1; skipped
without Stockfish. These check the briefing's shape and grounding, not exact
wording of engine-derived facts.
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import pytest

from lucena_engine.board import Board
from lucena_engine import Engine
from lucena_engine.analysis import build_analysis

NODES = 60_000

# White has Rb8+ forcing win despite being down a pawn (eval and material disagree).
SESSION = "6k1/pR4p1/7p/4N2K/P6P/8/5b1r/8 w - - 1 2"
# Black is up a bishop for a pawn; d5 is the isolated queen pawn.
IQP = "r1bq1rk1/pp2bppp/2n2n2/3p4/3P4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 11"
# White to move but Black has a forced mate in 1.
MATE1 = "r4rk1/ppp2p2/3p3p/4p3/4P1nq/2NP4/PPP2PP1/R3QRK1 w - - 0 15"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _have_sf():
    return bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")


requires_engine = pytest.mark.skipif(not _have_sf(), reason="no stockfish")


@pytest.fixture
def engine():
    with Engine(threads=1) as e:
        e.new_game()
        yield e


@requires_engine
def test_returns_nonempty_list_of_nl_lines(engine):
    lines = build_analysis(Board(SESSION), engine, nodes=NODES)
    assert isinstance(lines, list) and lines
    assert all(isinstance(s, str) and s.strip() for s in lines)


@requires_engine
def test_first_line_is_eval_with_turn_and_numbers(engine):
    first = build_analysis(Board(SESSION), engine, nodes=NODES)[0]
    assert "White to move" in first          # side to move, grounded
    assert "White is winning" in first        # White POV verdict
    assert "%" in first                        # carries numbers (coach's input, not player-facing)


@requires_engine
def test_deterministic(engine):
    a = build_analysis(Board(IQP), engine, nodes=NODES)
    engine.new_game()
    b = build_analysis(Board(IQP), engine, nodes=NODES)
    assert a == b


@requires_engine
def test_material_line_is_white_pov_and_names_the_leader(engine):
    # IQP: Black is up material — the material line must say so (White's POV, absolute).
    lines = build_analysis(Board(IQP), engine, nodes=NODES)
    assert any("Black is up" in s for s in lines)


@requires_engine
def test_forced_mate_is_named(engine):
    first = build_analysis(Board(MATE1), engine, nodes=NODES)[0]
    assert "forced mate in 1" in first


@requires_engine
def test_startpos_collapses_to_nothing_sharp(engine):
    # A flat position states the eval and ONE "nothing sharp" line — not five filler term sentences,
    # and never the always-on "least active piece" nudge.
    lines = build_analysis(Board(START), engine, nodes=NODES)
    assert "roughly equal" in lines[0].lower()
    assert any("nothing sharp" in s.lower() for s in lines)
    assert not any("least active piece" in s.lower() for s in lines)
    assert not any("material is even" in s.lower() for s in lines), "no filler term line at balance"


@requires_engine
def test_tactics_line_present_when_the_fact_sheet_has_facts(engine):
    # SESSION has a forcing sequence + a pin + a defender-removed fact.
    lines = build_analysis(Board(SESSION), engine, nodes=NODES)
    assert any(s.startswith("Tactics:") for s in lines)


# -- assemble_analysis: pure ranking + compensation (no engine, no Stockfish) ----------------------

from lucena_engine.analysis import assemble_analysis, _SHEET_THRESHOLD


class _Score:
    def __init__(self, cp, mate=None):
        self._cp, self.is_mate, self.mate = cp, mate is not None, (mate or 0)
    def to_ceiled_cp(self):
        return self._cp


class _Bd:
    def __init__(self, stm="white"):
        self.side_to_move = stm


def test_long_mate_reads_as_easily_winning():
    # A mate ≤5 recites the count; a mate too far out (>5) is called what it is — 'easily winning' —
    # rather than an uninstructive 'mate in 13'.
    from lucena_engine.analysis import _eval_line
    assert "forced mate in 5" in _eval_line(_Bd("white"), _Score(0, mate=5))
    assert "easily winning" in _eval_line(_Bd("white"), _Score(0, mate=6))
    assert "mate in" not in _eval_line(_Bd("black"), _Score(0, mate=13)), "long mates don't recite a count"


def _pos(**overrides):
    """A positional dict with the five terms; each override is term=(cp, standing)."""
    base = {"material": (0, "material is even"),
            "king_safety": (0, "both kings are reasonably safe"),
            "activity": (0, "piece activity is roughly balanced"),
            "pawns": (0, "both pawn structures are healthy"),
            "center": (0, "the centre is contested")}
    base.update(overrides)
    terms = {k: {"cp": cp, "standing": st} for k, (cp, st) in base.items()}
    leads = [k for k in sorted(terms, key=lambda k: -abs(terms[k]["cp"])) if abs(terms[k]["cp"]) >= 25][:2]
    return {"phase": 1.0, "terms": terms, "leads": leads}


def test_flat_position_says_nothing_sharp_once():
    lines = assemble_analysis(_Bd(), _Score(30), _pos(), [])
    assert sum("Nothing sharp" in s for s in lines) == 1
    assert not any("Main factor" in s for s in lines)
    assert not any("balanced across the board" in s and "activity" in s for s in lines)


def test_decisive_eval_names_a_main_factor_and_no_compensation():
    # Black up a bishop, White a bit more active; eval decisive → NOT compensation.
    pos = _pos(material=(-362, "Black is up a bishop"), activity=(119, "White's pieces are the more active"))
    lines = assemble_analysis(_Bd(), _Score(-321), pos, [])
    assert any(s.startswith("Main factor — Black is up a bishop") for s in lines)
    assert not any("compensation" in s.lower() for s in lines)
    assert any("more active" in s for s in lines)          # secondary context, plainly stated


def test_level_eval_despite_big_gap_is_compensation():
    # Danish-gambit shape: Black up two pawns, White big activity + centre; eval ≈ level.
    pos = _pos(material=(-164, "Black is up two pawns"),
               activity=(101, "White's pieces are the more active"),
               center=(60, "White controls the centre"))
    lines = assemble_analysis(_Bd(stm="black"), _Score(47), pos, [])   # black to move, +47 stm ≈ -47 W? see note
    joined = " ".join(lines)
    assert "Main factor — Black is up two pawns" in joined
    assert "White's compensation:" in joined
    assert "more active" in joined and "centre" in joined


def test_sub_threshold_terms_are_dropped():
    # 40 < 50 threshold → not stated; 60 ≥ 50 → stated.
    pos = _pos(material=(-300, "Black is up a piece"),
               center=(60, "White controls the centre"),
               pawns=(40, "White's pawns are slightly better"))
    lines = " ".join(assemble_analysis(_Bd(), _Score(-250), pos, []))
    assert "controls the centre" in lines
    assert "pawns are slightly better" not in lines
    assert _SHEET_THRESHOLD == 50


def test_no_least_active_nudge_ever_reaches_the_sheet():
    pos = _pos(activity=(120, "White's pieces are the more active"), material=(-200, "Black is up the exchange"))
    lines = " ".join(assemble_analysis(_Bd(), _Score(-150), pos, []))
    assert "least active piece" not in lines


def test_eval_line_stays_first():
    lines = assemble_analysis(_Bd(), _Score(-321), _pos(material=(-362, "Black is up a bishop")), [])
    assert lines[0].startswith("White to move")
