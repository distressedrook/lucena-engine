"""Detecting a pasted position in free text — FEN + PGN — validated by the board core.

The board must never move to a model-produced position; a pasted FEN/PGN is recognised and
VALIDATED here, deterministically. These tests pin that the regex finds candidates and the
parser rejects the impossible (bad rank sum, no king, illegal SAN)."""

from lucena_engine.detect import detect_fens, detect_pgn


def test_fen_extracted_from_a_sentence():
    s = "coach, what's best in r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4?"
    assert detect_fens(s) == ["r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4"]


def test_fen_without_move_counters_is_normalised():
    assert detect_fens("analyze 8/8/8/8/8/8/4k3/4K2R w K -") == ["8/8/8/8/8/8/4k3/4K2R w K - 0 1"]


def test_two_fens_in_one_message():
    s = "compare 8/8/8/8/8/8/4k3/4K2R w K - 0 1 and 8/8/8/8/8/8/4k3/R3K3 b Q - 0 1"
    assert len(detect_fens(s)) == 2


def test_fen_false_positives_rejected():
    assert detect_fens("we met on 1/2/3 last week") == []       # a date, not a FEN
    assert detect_fens("8/8/8/8/8/8/8/8 w - - 0 1") == []       # legal shape, but no kings
    assert detect_fens("rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") == []  # rank sums to 9


def test_pgn_from_bare_embedded_movetext():
    g = detect_pgn("hey, what's best after 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6?")
    assert g is not None and len(g.plies) == 6 and g.plies[-1].san == "a6"


def test_pgn_with_headers():
    g = detect_pgn('[Event "x"]\n[White "A"]\n[Black "B"]\n\n1. d4 d5 2. c4 e6 *')
    assert g is not None and len(g.plies) == 4


def test_pgn_with_annotations():
    g = detect_pgn("I played 1. e4 e5 2. Ke2?? — bad?")
    assert g is not None and len(g.plies) == 3


def test_pgn_illegal_move_is_none():
    assert detect_pgn("look at 1. e4 e5 2. Ke7 what next") is None   # Ke7 is illegal here


def test_no_pgn_returns_none():
    assert detect_pgn("can you help me understand forks?") is None
