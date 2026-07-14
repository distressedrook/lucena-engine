"""fork / double-attack detector (detectors.fork.detect_fork) + its engine
reconciliation gate in build_fact_sheet.

The detector is pure board geometry, so its own tests need no engine and are
fully deterministic. The reconciliation gate (a fork the engine rejects is
dropped) needs the engine and is marked accordingly.
"""

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from lucena_engine.board import Board
from lucena_engine.detectors import detect_fork
from lucena_engine.facts import build_fact_sheet
from lucena_engine.uci import Engine

QD4 = "r2q1rk1/pp3ppp/2p1b3/2R5/2P5/3P1Q1P/P1P2PP1/R1B3K1 b - - 2 15"  # Qd4 forks Ra1+Rc5
ND5 = "2r4r/4kpp1/p3p2p/7n/Bq3P2/2N2Q1P/1PP3P1/4R1K1 w - - 2 28"        # Nd5+ royal fork
HANG = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"                       # plain hang, no fork
QUIET = "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3"


def _forks(fen):
    return detect_fork(Board(fen))


# -- board-only detection (deterministic, no engine) ---------------------
def test_detects_queen_fork_of_two_loose_rooks():
    forks = _forks(QD4)
    assert any(f.kind == "fork" for f in forks)
    f = next(f for f in forks if f.provenance == "fork:d8d4")
    assert f.squares[0] == "d4"                 # the forking square first
    assert set(f.squares[1:]) == {"a1", "c5"}   # both loose rooks
    assert "a1" in f.text and "c5" in f.text
    assert f.concept_id == "forks-double-attack"


def test_detects_royal_knight_fork_with_check():
    f = next(f for f in _forks(ND5) if f.provenance == "fork:c3d5")
    assert "king" in f.text and "b4" in f.text   # king + the queen on b4
    assert f.squares == ["d5", "b4"]
    assert f.salience > 0.9                       # a checking queen-fork ranks high


def test_no_fork_on_a_simple_single_target():
    assert _forks(HANG) == []                     # Rxe5 wins one piece — not a fork


def test_no_fork_in_a_quiet_opening():
    assert _forks(QUIET) == []


def test_fork_squares_start_with_the_landing_square():
    for fen in (QD4, ND5):
        for f in _forks(fen):
            assert len(f.squares) >= 2            # landing + ≥1 target
            assert f.provenance.startswith("fork:")


# -- engine reconciliation gate ------------------------------------------
_have_engine = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
requires_engine = pytest.mark.skipif(not _have_engine, reason="no stockfish")


@pytest.mark.engine
@requires_engine
def test_reconciliation_keeps_only_the_engine_endorsed_fork():
    # Board geometry finds two forks on the Nd5+ position (the sound knight fork
    # and an inferior queen fork); the engine gate must keep only the knight.
    board_only = [f for f in _forks(ND5) if f.kind == "fork"]
    assert len(board_only) >= 2                    # geometry is liberal

    with Engine(threads=1) as e:
        e.new_game()
        sheet = build_fact_sheet(Board(ND5), e, nodes=1_500_000)
    forks = [f for f in sheet if f.kind == "fork"]
    assert len(forks) == 1                          # the inferior one was reconciled away
    assert forks[0].provenance == "fork:c3d5"       # the knight fork survives
    assert forks[0].id == "F1"                       # and leads the sheet


@pytest.mark.engine
@requires_engine
def test_fork_leads_the_sheet_over_the_old_distractor():
    # Regression for case-03: the winning fork must outrank the Rh5/Rg5 threat
    # the M3 sheet used to surface alone.
    with Engine(threads=1) as e:
        e.new_game()
        sheet = build_fact_sheet(Board(QD4), e, nodes=1_500_000)
    assert sheet[0].kind == "fork"
    assert set(sheet[0].squares[1:]) == {"a1", "c5"}
