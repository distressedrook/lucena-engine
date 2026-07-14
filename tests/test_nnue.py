"""Black-box tests for lucena_engine.nnue (M2 contract).

Two kinds of test:

  * PURE parser tests (no engine): grids are built by hand with `_build_grid`,
    whose output format was verified to round-trip through the real parser.
    Mutations of it exercise every ValueError path in the contract.
  * ENGINE snapshot tests (marked `engine`): capture real `raw_eval` output
    and snapshot-assert specific squares. NNUE piece values are Stockfish-18
    derived; asserted with a tolerance to survive minor net revisions, while
    sign/case/structure are asserted strictly.
"""

import os
import shutil

import pytest

from lucena_engine import Engine, parse_piece_values, piece_value_map, PieceValue


# ============================================================
#  Hand-built grid tooling (PURE — no engine).
#  Format mirrors Stockfish's "NNUE derived piece values" block:
#  first printed rank is rank 8, files a-h left→right; king cells blank.
# ============================================================

_SEP = "+-------+-------+-------+-------+-------+-------+-------+-------+"


def _cell(s):
    return f" {s:^5} "


def _build_grid(ranks):
    """ranks: 8 lists (rank8 first) of 8 cells; a cell is None (empty) or
    (piece_letter, value_or_None)."""
    lines = [" NNUE derived piece values:", _SEP]
    for row in ranks:
        letters = "|" + "|".join(_cell(c[0] if c else "") for c in row) + "|"

        def vfmt(c):
            if not c or c[1] is None:
                return ""
            v = c[1]
            return f"{'+' if v >= 0 else '-'}{abs(v):.2f}"

        vals = "|" + "|".join(_cell(vfmt(c)) for c in row) + "|"
        lines += [letters, vals, _SEP]
    lines += [
        "",
        " NNUE evaluation        +0.12 (white side)",
        "Final evaluation       +0.15 (white side)",
    ]
    return lines


_EMPTY = [None] * 8


def _rank(**kw):
    row = [None] * 8
    for f, cell in kw.items():
        row[ord(f) - 97] = cell
    return row


def _minimal_valid():
    # black rook a8 + black king e8, white rook a1 + white king e1.
    return _build_grid(
        [
            _rank(a=("r", -5.0), e=("k", None)),  # rank 8
            _EMPTY,
            _EMPTY,
            _EMPTY,
            _EMPTY,
            _EMPTY,
            _EMPTY,
            _rank(a=("R", 5.0), e=("K", None)),  # rank 1
        ]
    )


# -------------------- PURE parser: happy path --------------------


def test_hand_built_grid_parses():
    pvs = parse_piece_values(_minimal_valid())
    by_sq = {p.square: p for p in pvs}
    assert set(by_sq) == {"a8", "e8", "a1", "e1"}
    assert by_sq["a8"] == PieceValue(square="a8", piece="r", value=-5.0)
    assert by_sq["a1"] == PieceValue(square="a1", piece="R", value=5.0)
    # kings carry value None regardless of case.
    assert by_sq["e8"].piece == "k" and by_sq["e8"].value is None
    assert by_sq["e1"].piece == "K" and by_sq["e1"].value is None


def test_hand_built_rank_file_mapping():
    # place a lone white knight on h5 (rank 5, file h) and confirm the square
    # decodes correctly: rank index 3 from top (rank 8,7,6,5), file 'h'.
    grid = [_EMPTY] * 3 + [_rank(h=("N", 3.0))] + [_EMPTY] * 4
    pvs = parse_piece_values(_build_grid(grid))
    assert len(pvs) == 1
    assert pvs[0].square == "h5"
    assert pvs[0].piece == "N"


def test_piece_value_map_keys_by_square():
    m = piece_value_map(_minimal_valid())
    assert set(m) == {"a8", "e8", "a1", "e1"}
    assert m["a1"].piece == "R"
    assert m["e8"].value is None


# -------------------- PURE parser: malformed → ValueError --------------------


def test_empty_input_raises():
    with pytest.raises(ValueError):
        parse_piece_values([])


def test_non_eval_input_raises():
    with pytest.raises(ValueError):
        parse_piece_values(["hello", "world", "not an eval dump at all"])


def test_header_only_no_rows_raises():
    with pytest.raises(ValueError):
        parse_piece_values([" NNUE derived piece values:", _SEP])


def test_truncated_rows_raises():
    # header + first separator + one rank only (wrong row count).
    full = _minimal_valid()
    with pytest.raises(ValueError):
        parse_piece_values(full[:5])


def test_non_king_piece_missing_value_raises():
    # A white ROOK on a1 with a blank value cell — only kings may be valueless.
    bad = _build_grid(
        [
            _rank(a=("r", -5.0), e=("k", None)),
            _EMPTY,
            _EMPTY,
            _EMPTY,
            _EMPTY,
            _EMPTY,
            _EMPTY,
            _rank(a=("R", None), e=("K", None)),  # R with no value → invalid
        ]
    )
    with pytest.raises(ValueError):
        parse_piece_values(bad)


def test_garbage_raises():
    with pytest.raises(ValueError):
        parse_piece_values(["%%%%", "| ? | ? |", "not#a#grid", "+++"])


# ============================================================
#  ENGINE snapshot tests (real raw_eval output).
# ============================================================

_HAVE_SF = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
engine_only = pytest.mark.skipif(not _HAVE_SF, reason="no stockfish")

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Italian-game middlegame, White to move.
MIDDLEGAME_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"


@pytest.fixture(scope="module")
def start_pvs():
    with Engine(threads=1) as e:
        return parse_piece_values(e.raw_eval(START_FEN))


@pytest.mark.engine
@engine_only
def test_startpos_has_32_entries(start_pvs):
    assert len(start_pvs) == 32


@pytest.mark.engine
@engine_only
def test_startpos_exactly_two_kings_valued_none(start_pvs):
    kings = [p for p in start_pvs if p.value is None]
    assert len(kings) == 2
    assert {p.square for p in kings} == {"e1", "e8"}
    assert {p.piece for p in kings} == {"K", "k"}


@pytest.mark.engine
@engine_only
def test_startpos_case_convention(start_pvs):
    by_sq = {p.square: p for p in start_pvs}
    # rank 1 & 2 = white (UPPER), rank 7 & 8 = black (lower).
    for sq in ("a1", "d1", "a2", "h2"):
        assert by_sq[sq].piece.isupper(), sq
    for sq in ("a8", "d8", "a7", "h7"):
        assert by_sq[sq].piece.islower(), sq


@pytest.mark.engine
@engine_only
def test_startpos_white_pov_sign(start_pvs):
    # White's-POV values: white pieces positive, black pieces negative.
    for p in start_pvs:
        if p.value is None:
            continue
        if p.piece.isupper():
            assert p.value > 0, f"white {p.piece}@{p.square} should be +"
        else:
            assert p.value < 0, f"black {p.piece}@{p.square} should be −"


@pytest.mark.engine
@engine_only
def test_middlegame_square_snapshot():
    # SF18-derived snapshot; values in pawns from White's POV.
    # Captured via raw_eval on MIDDLEGAME_FEN (tolerant to net revisions,
    # strict on sign/piece/case).
    with Engine(threads=1) as e:
        m = piece_value_map(e.raw_eval(MIDDLEGAME_FEN))
    expected = {
        "e4": ("P", +1.59),
        "e5": ("p", -1.59),
        "c4": ("B", +4.15),
        "c6": ("n", -3.82),
        "f3": ("N", +3.97),
        "f6": ("n", -3.90),
        "d1": ("Q", +6.59),
        "d8": ("q", -7.07),
    }
    for sq, (piece, val) in expected.items():
        pv = m[sq]
        assert pv.piece == piece, f"{sq}: piece {pv.piece} != {piece}"
        assert pv.value == pytest.approx(val, abs=0.5), f"{sq}: {pv.value} vs {val}"
        # sign is load-bearing (White's POV), assert it strictly.
        assert (pv.value > 0) == piece.isupper(), sq


@pytest.mark.engine
@engine_only
def test_black_piece_negative_against_real_capture():
    # In a balanced-ish, symmetric-material position a black pawn's NNUE value
    # is negative (bad for White) while its white counterpart is positive —
    # the White's-POV sign convention, verified against real engine output.
    with Engine(threads=1) as e:
        m = piece_value_map(e.raw_eval(MIDDLEGAME_FEN))
    assert m["e5"].value < 0 < m["e4"].value
    assert m["c6"].value < 0  # black knight
    assert m["f3"].value > 0  # white knight
