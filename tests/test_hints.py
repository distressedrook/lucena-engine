"""Grounded hint-ladder derivation (engine.hints.derive_hints).

The derivation is board-core over an engine PV it is *handed*, so these tests
feed synthetic `Analysis` objects with real PVs — deterministic, no live engine.
The contract under test: every rung is grounded in the PV/geometry (provenance
present), the ladder is ordered vague → specific, and the deriver *abstains*
(returns []) rather than fabricate a nudge when the line has no tactical handle.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from lucena_engine.board import Board
from lucena_engine.hints import Hint, derive_hints
from lucena_engine.uci import Analysis, Line
from lucena_engine.evalmodel import Score


def _analysis(fen, pv, cp=700):
    return Analysis(fen=fen, lines=[Line(rank=1, score=Score(cp=cp, mate=None), pv=pv)])


# Real positions + their real best-line PVs (from the puzzle DB / eval cases).
Nd5 = "2r4r/4kpp1/p3p2p/7n/Bq3P2/2N2Q1P/1PP3P1/4R1K1 w - - 2 28"
Nd5_PV = ["c3d5", "e7f8", "d5b4", "g7g6", "b4c6"]

RXG7 = "r4rk1/2p3bR/p2p4/1p1Pp1N1/2P2q2/8/PP1Q4/2K1R3 w - - 0 28"
RXG7_PV = ["h7g7", "g8h8", "e1h1", "h8g7", "g5e6"]

HANG = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"
HANG_PV = ["e1e5", "g8f8", "e5e1"]

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# -- king-fork: an immediate royal fork (Nd5+ forks Ke7 and Qb4) ----------
def test_king_fork_ladder_targets_the_prize_then_the_piece():
    hints = derive_hints(Board(Nd5), _analysis(Nd5, Nd5_PV))
    assert [h.rung for h in hints] == [1, 2, 3]
    # rung 1 names the won piece's square (the queen on b4), not the move
    assert "b4" in hints[0].text and "queen" in hints[0].text
    assert hints[0].squares == ["b4"]
    # rung 2 is the resource: the knight, hitting several at once
    assert "knight" in hints[1].text and "at once" in hints[1].text
    assert hints[1].provenance.startswith("fork:")
    # rung 3 is the forcing entry, traced to pv[0]
    assert hints[2].provenance.startswith(("pv:", "fork:"))


# -- deep king-hunt: fork is Ne6+ four plies down the sac line ------------
def test_deep_kinghunt_reads_the_fork_downstream_and_the_sac_entry():
    hints = derive_hints(Board(RXG7), _analysis(RXG7, RXG7_PV))
    assert len(hints) == 3
    # the Ne6 fork hits queen + rook + gives check => "3 things"
    assert "knight" in hints[1].text and "3 things" in hints[1].text
    # the entry move (Rxg7) is a sacrifice by SEE -> rung 3 says give up material
    assert "give up material" in hints[2].text
    assert hints[2].squares == ["h7"]  # the entry square, not the fork square


# -- direct material win: a plain winning capture, gentle ladder ----------
def test_direct_win_ladder_points_at_the_loose_piece():
    hints = derive_hints(Board(HANG), _analysis(HANG, HANG_PV))
    assert len(hints) >= 2
    assert "e5" in hints[0].text  # the knight that can be won sits on e5
    assert all(h.provenance for h in hints)  # everything traced
    assert any("capture" in h.text for h in hints)


# -- abstain: no tactical handle => no fabricated hints -------------------
def test_quiet_line_returns_no_hints():
    assert derive_hints(Board(START), _analysis(START, ["e2e4", "e7e5"], cp=20)) == []


def test_empty_pv_returns_no_hints():
    assert derive_hints(Board(START), _analysis(START, [])) == []


# -- mate awareness: a forced mate is a king hunt, not a material grab ----
MATE1 = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"  # Ra8# — back-rank mate in one


def _mate_analysis(fen, pv, n):
    return Analysis(fen=fen, lines=[Line(rank=1, score=Score(mate=n), pv=pv)])


def test_forced_mate_is_framed_as_a_king_hunt_not_a_material_win():
    hints = derive_hints(Board(MATE1), _mate_analysis(MATE1, ["a1a8"], 1))
    assert hints, "a mate should still produce a ladder"
    assert "mate" in hints[0].text.lower()          # rung 1 names the mate
    assert hints[0].provenance.startswith("mate:")
    assert "g8" in hints[0].text                     # points at the hunted king
    # never mislabels a mate as "win N things at once"
    assert all("at once" not in h.text for h in hints)


def test_mate_ladder_entry_rung_flags_the_forcing_check():
    hints = derive_hints(Board(MATE1), _mate_analysis(MATE1, ["a1a8"], 1))
    assert "check" in hints[-1].text                 # Ra8 is a forcing check


def test_no_lines_returns_no_hints():
    assert derive_hints(Board(START), Analysis(fen=START, lines=[])) == []


# -- shape + ordering invariants -----------------------------------------
def test_rungs_are_hints_ordered_and_capped():
    hints = derive_hints(Board(Nd5), _analysis(Nd5, Nd5_PV), max_rungs=2)
    assert len(hints) == 2
    assert all(isinstance(h, Hint) for h in hints)
    assert [h.rung for h in hints] == sorted(h.rung for h in hints)


def test_no_rung_states_the_move_verbatim():
    # withholding: no rung spells the answer as a from->to move string
    for fen, pv in [(Nd5, Nd5_PV), (RXG7, RXG7_PV)]:
        hints = derive_hints(Board(fen), _analysis(fen, pv))
        answer = pv[0]  # e.g. "c3d5"
        joined = " ".join(h.text for h in hints).lower()
        assert answer not in joined  # never the raw uci


def test_deep_fork_on_a_moved_piece_is_not_a_phantom_hint():
    # Regression (live get_hints bug): the PV drifts a bishop from h3 to c8, where
    # Ne7+ forks it — but that bishop is NOT on c8 in the current position, so
    # naming "the bishop on c8" would be a phantom. derive_hints must reference
    # only pieces on the board now, so here it abstains rather than hallucinate.
    fen = "6k1/8/2N5/8/8/7b/8/6K1 w - - 0 1"
    pv = ["g1f2", "h3c8", "c6e7"]   # Kf2, ...Bc8, Ne7+ (forks Kg8 + the drifted bishop)
    assert derive_hints(Board(fen), _analysis(fen, pv)) == []
