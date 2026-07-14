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
def test_startpos_is_equal_and_even(engine):
    lines = build_analysis(Board(START), engine, nodes=NODES)
    assert "roughly equal" in lines[0].lower()
    assert any("material is even" in s.lower() for s in lines)


@requires_engine
def test_tactics_line_present_when_the_fact_sheet_has_facts(engine):
    # SESSION has a forcing sequence + a pin + a defender-removed fact.
    lines = build_analysis(Board(SESSION), engine, nodes=NODES)
    assert any(s.startswith("Tactics:") for s in lines)
