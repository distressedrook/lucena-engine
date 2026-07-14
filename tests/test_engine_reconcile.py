"""Reconciliation additions from the Agent B (critic) pass on the M2 engine
suite. Each fills a black-box blind spot the critic proved could hide a real
bug behind Agent A's green suite. Verified against the contract formula and
real Stockfish 18; arithmetic shown inline.
"""

import os
import stat

import pytest

from lucena_engine import (
    Engine,
    EngineError,
    Glyph,
    Score,
    classify,
    parse_piece_values,
    win_pct,
)

pytestmark = []


# --- Finding #1: classify with a NON-ZERO best (Agent A always used cp=0, so
# a bug that negated best_from_mover was invisible; Score(0).negated()==0). ---

def test_classify_nonzero_best_catches_negate_bugs():
    # best_wp = win_pct(200) = 67.62; played negated to +100 → win_pct(100)
    # = 59.10; drop = 8.52 → DUBIOUS.
    # If best were negated: win_pct(-200)=32.38 → drop clamps to 0 → OK (fail).
    # If played negate were dropped: win_pct(-100)=40.90 → drop 26.72 →
    # BLUNDER (fail). Only the correct perspective yields DUBIOUS.
    glyph, drop = classify(Score(cp=200), Score(cp=-100))
    assert glyph is Glyph.DUBIOUS
    assert 8.0 < drop < 9.0


def test_classify_nonzero_best_equal_positions_is_ok():
    # best +200 (mover POV), played leaves opponent at -200 → mover POV +200
    # → drop 0 → OK. A negated-best bug would make this a huge drop.
    glyph, drop = classify(Score(cp=200), Score(cp=-200))
    assert glyph is Glyph.OK
    assert drop == pytest.approx(0.0, abs=1e-9)


def test_classify_mate_on_best_side():
    # best = mate (win% ≈ 97.545); played leaves opponent at -300 → mover POV
    # +300 → win_pct(300)=74.63; drop ≈ 22.9 → BLUNDER (a missed mate).
    glyph, drop = classify(Score(mate=3), Score(cp=-300))
    assert glyph is Glyph.BLUNDER
    # best=mate, played that KEEPS a winning edge close to mate → small drop.
    # played leaves opponent at -900 → mover POV +900 → win_pct(900)=96.44;
    # drop ≈ 1.1 → OK.
    glyph2, drop2 = classify(Score(mate=3), Score(cp=-900))
    assert glyph2 is Glyph.OK
    assert drop2 < 5.0


# --- Finding #2: NNUE sign must come from the PRINTED value, not the piece
# letter case. Agent A's positions all had sign==case, so a sign-from-case
# parser passed. Real SF18 prints negative values for White pieces in some
# positions (verified: 7k/8/8/8/8/8/6PP/6BK → white pawns/bishop negative). ---

_SEP = "+-------+-------+-------+-------+-------+-------+-------+-------+"


def _cell(s):
    return f" {s:^5} "


def _vfmt(c):
    if not c or c[1] is None:
        return ""
    v = c[1]
    return f"{'+' if v >= 0 else '-'}{abs(v):.2f}"


def _build_grid(ranks):
    """ranks: 8 lists (rank8 first) of 8 cells; cell = None | (letter, value)."""
    lines = [" NNUE derived piece values:", _SEP]
    for row in ranks:
        lines.append("|" + "|".join(_cell(c[0] if c else "") for c in row) + "|")
        lines.append("|" + "|".join(_cell(_vfmt(c)) for c in row) + "|")
        lines.append(_SEP)
    lines += ["", "Final evaluation       +0.00 (white side)"]
    return lines


def _rank(**kw):
    row = [None] * 8
    for f, cell in kw.items():
        row[ord(f) - 97] = cell
    return row


def test_nnue_sign_from_printed_value_not_letter_case():
    # A WHITE rook (upper case) printed with a NEGATIVE value must parse as
    # negative. A sign-from-case parser would return +5.0 and fail.
    grid = _build_grid(
        [
            _rank(a=("r", 5.0), h=("k", None)),   # rank 8: black rook printed POSITIVE
            *([_rank()] * 6),
            _rank(a=("R", -5.0), h=("K", None)),  # rank 1: white rook printed NEGATIVE
        ]
    )
    by_sq = {pv.square: pv for pv in parse_piece_values(grid)}
    assert by_sq["a1"].piece == "R" and by_sq["a1"].value == -5.0
    assert by_sq["a8"].piece == "r" and by_sq["a8"].value == 5.0
    assert by_sq["h1"].value is None and by_sq["h8"].value is None  # both kings


# --- Finding #3: the "fewer than multipv" branch (Agent A only tested a
# high-mobility startpos where len==multipv exactly). ---

@pytest.mark.engine
def test_multipv_fewer_lines_when_few_legal_moves():
    # Black king h8 in check from Rh2 with exactly two legal replies
    # (Kg8, Kg7); requesting multipv=5 must yield 1..2 contiguous-rank lines,
    # never a padded 5.
    fen = "7k/8/8/8/8/8/7R/1R5K b - - 0 1"
    with Engine(threads=1) as eng:
        a = eng.analyse(fen, nodes=50_000, multipv=5)
    assert 1 <= len(a.lines) <= 2
    assert [ln.rank for ln in a.lines] == list(range(1, len(a.lines) + 1))


# --- Finding #5: version-mismatch is black-box reachable via a UCI stub that
# reports a non-SF18 id name (Agent A had declared it untestable). ---

_STUB = """#!/usr/bin/env python3
import sys
for line in sys.stdin:
    line = line.strip()
    if line == "uci":
        print("id name Stockfish 17"); print("uciok"); sys.stdout.flush()
    elif line == "isready":
        print("readyok"); sys.stdout.flush()
    elif line == "quit":
        break
"""


@pytest.mark.engine
def test_version_mismatch_raises(tmp_path):
    stub = tmp_path / "fake_sf.py"
    stub.write_text(_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXUSR)
    with pytest.raises(EngineError, match="Stockfish 18"):
        Engine(path=str(stub))


@pytest.mark.engine
def test_version_mismatch_bypassed_when_assert_off(tmp_path):
    # With assert_version=False the stub's wrong version is tolerated through
    # the handshake (it will fail later on real analysis, but construction and
    # option-setting must not raise on the version check itself).
    stub = tmp_path / "fake_sf.py"
    stub.write_text(_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC | stat.S_IXUSR)
    # The stub answers isready but ignores setoption; construction should reach
    # readyok without a version error.
    eng = Engine(path=str(stub), assert_version=False)
    eng.close()
