"""SEE against a hand-verified exchange table.

Every expected value below was derived by playing out the exchange
least-valuable-attacker-first by hand; comments show the accounting.
Values: P=100 N=B=300 R=500 Q=900.
"""

from lucena_engine import Board


def see(fen: str, move: str) -> int:
    return Board(fen).see(move)


def test_pawn_takes_pawn_queen_recaptures_is_equal():
    # 1.e4 d5: exd5 wins a pawn, Qxd5 takes the pawn back → net 0.
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    assert see(fen, "e4d5") == 0


def test_rook_takes_free_pawn():
    fen = "4k3/8/8/4p3/8/8/8/4R1K1 w - - 0 1"
    assert see(fen, "e1e5") == 100


def test_rook_takes_pawn_defended_by_pawn():
    # Rxe5 dxe5: +100 −500 = −400.
    fen = "4k3/8/3p4/4p3/8/8/8/4R1K1 w - - 0 1"
    assert see(fen, "e1e5") == -400


def test_xray_doubled_rooks_still_losing():
    # Rxe5 dxe5 Rxe5: +100 −500 +100 = −300 (x-ray rook joins via e-file).
    fen = "4k3/8/3p4/4p3/8/8/4R3/4R1K1 w - - 0 1"
    assert see(fen, "e2e5") == -300


def test_queen_grab_with_rook_battery_behind():
    # Qxe5 dxe5 Rxe5: +100 −900 +100 = −700.
    fen = "4k3/8/3p4/4p3/8/8/4Q3/4R1K1 w - - 0 1"
    assert see(fen, "e2e5") == -700


def test_knight_takes_defended_pawn():
    # Nxe5 Nxe5: +100 −300 = −200.
    fen = "4k3/8/2n5/4p3/8/5N2/8/4K3 w - - 0 1"
    assert see(fen, "f3e5") == -200


def test_equal_minor_exchange_on_defended_knight():
    # Bxc6 bxc6: bishop takes knight (+300), pawn takes bishop (−300) → 0.
    fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    assert see(fen, "b5c6") == 0


def test_en_passant_pawn_for_pawn():
    # exd6 e.p. cxd6: +100 −100 = 0.
    fen = "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3"
    assert see(fen, "e5d6") == 0


def test_king_cannot_capture_defended_pawn():
    # Kxe5 with Kd6 defending is an *illegal* move (into check); the board
    # core rejects it before SEE ever runs — the right behavior, since facts
    # are only ever computed over legal candidates. (The in-SEE king guard
    # covers kings appearing mid-exchange as recapturers.)
    import pytest

    fen = "8/8/3k4/4p3/5K2/8/8/8 w - - 0 1"
    with pytest.raises(ValueError, match="illegal"):
        see(fen, "f4e5")


def test_king_takes_free_pawn():
    fen = "8/8/8/4p3/5K2/8/3k4/8 w - - 0 1"
    assert see(fen, "f4e5") == 100


def test_winning_capture_overloaded_square():
    # Nxe5: pawn e5 defended only by Nc6-knight; after Nxe5 Nxe5 Rxe5:
    # +100 −300 +300 = +100 for White.
    fen = "4k3/8/2n5/4p3/8/5N2/8/4RK2 w - - 0 1"
    assert see(fen, "f3e5") == 100


def test_quiet_move_to_defended_square_is_negative():
    # Not a capture: rook steps onto a square attacked by a pawn → −500.
    # (Black king on h8 keeps the e-file check-free so the FEN is valid.)
    fen = "7k/8/3p4/8/8/8/8/4R1K1 w - - 0 1"
    assert see(fen, "e1e5") == -500
