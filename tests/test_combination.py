"""combination detector (detectors.combination.detect_combination) — the
multi-move "there's a forcing win here" fact, and its lead on the fact sheet.

Needs the engine (it reads the PV/eval verdict), so all tests are engine-marked.
Determinism split: nodes-limited.
"""

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from lucena_engine.board import Board
from lucena_engine.detectors import detect_combination
from lucena_engine.facts import build_fact_sheet
from lucena_engine.uci import Engine

MATE1 = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"                              # Ra8# in one
DEFLECT = "5rk1/R4pp1/1p5p/3Q4/1PPp2q1/3P2P1/5P2/4K3 b - - 0 34"        # Re8+ forcing win
QUIET = "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"
HANG = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"                        # plain 1-move win

_have_engine = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
requires_engine = pytest.mark.skipif(not _have_engine, reason="no stockfish")
NODES = 1_500_000


@pytest.fixture
def engine():
    with Engine(threads=1) as e:
        e.new_game()
        yield e


@pytest.mark.engine
@requires_engine
def test_forced_mate_becomes_a_combination_fact(engine):
    facts = detect_combination(Board(MATE1), engine, nodes=NODES)
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == "combination"
    assert "mate in 1" in f.text and "Ra8" in f.text
    assert f.provenance == "mate:a1a8"
    assert f.salience >= 0.98            # a mate leads everything


@pytest.mark.engine
@requires_engine
def test_mate_leads_the_fact_sheet(engine):
    sheet = build_fact_sheet(Board(MATE1), engine, nodes=NODES)
    assert sheet[0].kind == "combination"   # not a distractor


@pytest.mark.engine
@requires_engine
def test_forcing_decisive_line_is_a_combination(engine):
    # Re8+ is a check that wins by force — a multi-move combination, surfaced as
    # the lead fact instead of the old "opponent threatens Ra8" distractor.
    sheet = build_fact_sheet(Board(DEFLECT), engine, nodes=NODES)
    assert sheet[0].kind == "combination"
    assert "Re8" in sheet[0].text


@pytest.mark.engine
@requires_engine
def test_quiet_position_has_no_combination(engine):
    assert detect_combination(Board(QUIET), engine, nodes=NODES) == []


@pytest.mark.engine
@requires_engine
def test_plain_winning_capture_is_not_a_combination(engine):
    # A one-move win (Rxe5 wins the hanging knight) is the hanging fact, not a
    # forcing multi-move combination — the detector must not double-count it.
    assert detect_combination(Board(HANG), engine, nodes=NODES) == []
