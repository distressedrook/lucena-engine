"""Adversarial board/SAN/constructor cases from the black-box test critique.

Pins the contract clauses the smoke tests skipped: constructor validation
depth, in_check, error paths, disambiguation (double / pinned-twin
suppression), suffixes (stalemate ≠ mate, castling-with-check), en passant
and underpromotion notation, FEN state through apply, and the UCI/SAN
auto-detector boundary.
"""

import pytest

from lucena_engine import Board

EP_FEN = "4k3/2p5/8/3pP3/8/8/8/3RK3 w - d6 0 3"


# ---- constructor validation depth ------------------------------------------

def test_constructor_rejects_opposite_side_in_check():
    # Re2 checks e8 with Black NOT to move → invalid position.
    with pytest.raises(ValueError):
        Board("4k3/8/8/8/8/8/4R3/4K3 w - - 0 1")


def test_constructor_rejects_structurally_broken_boards():
    with pytest.raises(ValueError):
        Board("8/8/8/8/8/8/8/4K3 w - - 0 1")  # no black king
    with pytest.raises(ValueError):
        Board("P3k3/8/8/8/8/8/8/4K3 w - - 0 1")  # pawn on rank 8
    with pytest.raises(ValueError):
        Board("")


# ---- in_check / null_move error path ----------------------------------------

def test_in_check_true_and_null_move_rejected():
    b = Board("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1")  # Re2 checks Ke1
    assert b.in_check is True
    with pytest.raises(ValueError):
        b.null_move()


def test_in_check_false_at_startpos():
    assert Board().in_check is False


def test_null_move_clears_ep_square():
    after_e4 = Board().apply("e4")
    assert " e3 " in after_e4.fen
    assert " - " in after_e4.null_move().fen


# ---- attackers/defenders: pieces, direction, occlusion ----------------------

def test_attackers_pawns_forward_only():
    b = Board()
    assert b.attackers("e3", "white") == ["d2", "f2"]
    assert b.attackers("e3", "black") == []  # wrong direction + blocked sliders


def test_attackers_king_knight_bishop_queen_together():
    assert Board().attackers("e2", "white") == ["d1", "e1", "f1", "g1"]


def test_attackers_battery_rear_slider_hidden_at_full_occupancy():
    # Doubled rooks e1/e2: only the FRONT rook attacks e5 through full
    # occupancy — x-rays are a SEE concept, not an attackers() concept.
    b = Board("4k3/8/3p4/4p3/8/8/4R3/4R1K1 w - - 0 1")
    assert b.attackers("e5", "white") == ["e2"]


def test_attackers_bad_inputs_raise():
    with pytest.raises(ValueError):
        Board().attackers("z9", "white")
    with pytest.raises(ValueError):
        Board().attackers("e4", "purple")


def test_defenders_nonempty_and_empty_square_error():
    assert Board().defenders("f2") == ["e1"]  # king as the sole defender
    with pytest.raises(ValueError):
        Board().defenders("e4")


def test_defenders_key_off_occupant_not_side_to_move():
    # White to move, but d5 is occupied by a BLACK pawn → black defenders
    # (only Qd8 down the open d-file in this position).
    b = Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    assert b.defenders("d5") == ["d8"]


# ---- SAN: disambiguation hard cases ------------------------------------------

TRIPLE_QUEENS = "1k6/8/8/8/4Q2Q/8/8/K6Q w - - 0 1"


def test_double_and_single_disambiguation_three_queens():
    b = Board(TRIPLE_QUEENS)
    assert b.san("h4e1") == "Qh4e1"  # both file and rank needed
    assert b.san("e4e1") == "Qee1"   # file suffices
    assert b.san("h1e1") == "Q1e1"   # rank suffices
    for uci in ("h4e1", "e4e1", "h1e1"):
        assert b.uci(b.san(uci)) == uci


def test_pinned_twin_suppresses_disambiguation():
    # Ne4 is absolutely pinned by Re8: it cannot reach d2, so Nb1-d2 is
    # unambiguous — SAN must be "Nd2", NOT "Nbd2" (legal-moves-only clause).
    b = Board("4r2k/8/8/8/4N3/8/8/1N2K3 w - - 0 1")
    assert b.san("b1d2") == "Nd2"
    with pytest.raises(ValueError):
        b.san("e4d2")  # the pinned knight's move is illegal


# ---- SAN: suffixes and special notation --------------------------------------

def test_castling_with_check_and_queenside():
    assert Board("5k2/8/8/8/8/8/8/4K2R w K - 0 1").san("e1g1") == "O-O+"
    b = Board("r3k3/8/8/8/8/8/8/R3K3 w Qq - 0 1")
    assert b.san("e1c1") == "O-O-O"
    assert b.uci("O-O-O") == "e1c1"
    assert b.uci("0-0-0") == "e1c1"  # zeros tolerance


def test_stalemating_move_gets_no_suffix():
    # Qc7 stalemates Black (Ka8 has no moves, not in check) — a '#' detector
    # implemented as "opponent has no legal moves" fails here.
    assert Board("k7/8/1K6/8/8/8/8/2Q5 w - - 0 1").san("c1c7") == "Qc7"


def test_underpromotion_san_roundtrip():
    b = Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1")
    assert b.san("e7e8n") == "e8=N"
    assert b.uci("e8=N") == "e7e8n"


def test_en_passant_san_both_directions():
    b = Board(EP_FEN)
    assert b.san("e5d6") == "exd6"
    assert b.uci("exd6") == "e5d6"
    assert b.uci("exd6 e.p.") == "e5d6"  # tolerance clause


def test_san_suffix_tolerance():
    assert Board().uci("Nf3!?") == "g1f3"
    b = Board().apply("f3").apply("e5").apply("g4")
    assert b.uci("Qh4#") == "d8h4"


def test_roundtrip_over_notation_rich_positions():
    for fen in (TRIPLE_QUEENS, EP_FEN, "3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1"):
        pos = Board(fen)
        for uci in pos.legal_moves():
            assert pos.uci(pos.san(uci)) == uci, (fen, uci)


# ---- apply(): UCI form, immutability, errors, FEN state ----------------------

def test_apply_accepts_uci_and_promotion_uci():
    assert Board().apply("e2e4").side_to_move == "black"
    after = Board("7k/4P3/8/8/8/8/8/4K3 w - - 0 1").apply("e7e8q")
    assert any(p.square == "e8" and p.piece == "Q" and p.color == "white"
               for p in after.piece_list())


def test_apply_is_immutable():
    b = Board()
    b2 = b.apply("e4")
    assert b.side_to_move == "white" and b2.side_to_move == "black"
    assert b.fen != b2.fen


def test_apply_illegal_raises():
    with pytest.raises(ValueError):
        Board().apply("e2e5")


def test_fen_state_through_apply():
    b = Board().apply("e4")
    assert " e3 " in b.fen  # ep square emitted after double push (pinned convention)
    b = b.apply("e5").apply("Ke2")
    parts = b.fen.split()
    assert parts[2] == "kq"      # white rights gone after the king move
    assert parts[3] == "-"       # ep square cleared
    assert parts[4] == "1"       # halfmove clock: king move increments
    assert parts[5] == "2"       # fullmove number


def test_fen_roundtrip_is_stable():
    for fen in (Board().fen, EP_FEN, TRIPLE_QUEENS):
        assert Board(fen).fen == fen


def test_piece_count_drops_after_capture():
    b = Board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2")
    assert len(b.apply("exd5").piece_list()) == 31
