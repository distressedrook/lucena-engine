"""Adversarial SEE cases from the black-box test critique.

Every expectation was independently hand-minimaxed (trees in comments) and
the positions verified legal against a reference implementation. These pin
the branches where swap algorithms rot: stand-pat/decline, LVA ordering,
mixed-value batteries, en-passant occupancy, the king-join guard, and both
documented limitations (pins invisible; promotions valued as the pawn).
"""

import pytest

from lucena_engine import Board


def see(fen: str, move: str) -> int:
    return Board(fen).see(move)


def test_opponent_declines_mid_sequence():
    # Nxe5 +100. If Rxe5, Bxe5 wins the rook (net +300 for White) — so Black
    # STANDS PAT and White simply keeps the pawn.
    fen = "4r2k/8/8/4p3/8/5N2/1B6/6K1 w - - 0 1"
    assert see(fen, "f3e5") == 100


def test_mixed_battery_bishop_in_front_of_queen():
    # Bxe5 +100; dxe5 −200; Qxe5 (x-ray through b2) −100, undefended.
    # Black recaptures: min(+100, −100) → −100.
    fen = "7k/8/3p4/4p3/8/8/1B6/Q5K1 w - - 0 1"
    assert see(fen, "b2e5") == -100


def test_mixed_battery_queen_in_front_of_bishop():
    # Same diagonal, pieces swapped: Qxe5 +100; dxe5 −800; Bxe5 −700.
    fen = "7k/8/3p4/4p3/8/8/1Q6/B5K1 w - - 0 1"
    assert see(fen, "b2e5") == -700


def test_least_valuable_attacker_ordering():
    # White can recapture on d5 with bishop (b3) or queen (d1). Correct LVA
    # (bishop first): Nxd5 +100; exd5 −200; Bxd5 −100; Black stops (Rxd5
    # would lose the rook to Qxd5). Queen-first bugs yield −200.
    fen = "3r3k/8/4p3/3p4/5N2/1B6/8/3Q2K1 w - - 0 1"
    assert see(fen, "f4d5") == -100


def test_en_passant_vacates_victim_square_for_xray():
    # Rd1 is blocked by the d5 pawn UNTIL exd6 e.p. removes it — the rook
    # joins through the *victim's* square, not the mover's origin.
    # exd6 +100; cxd6 0; Rxd6 +100, undefended → +100.
    fen = "4k3/2p5/8/3pP3/8/8/8/3RK3 w - d6 0 3"
    assert see(fen, "e5d6") == 100


def test_king_joins_when_square_becomes_undefended():
    # Rxe5 +100; Nxe5 −400; the knight was e5's only defender, so Kxe5 is
    # allowed: −100. Black recaptures: min(+100, −100) → −100.
    fen = "7k/3n4/8/4p3/5K2/8/8/4R3 w - - 0 1"
    assert see(fen, "e1e5") == -100


def test_king_excluded_while_square_still_defended():
    # Same but with a second minor (Bg7): after either recapture the other
    # minor still guards e5, so the king may never join → White stuck at −400.
    fen = "7k/3n2b1/8/4p3/5K2/8/8/4R3 w - - 0 1"
    assert see(fen, "e1e5") == -400


def test_documented_limitation_pinned_defender_still_counts():
    # Nd7 is absolutely pinned (zero legal moves) yet SEE must count it —
    # pins are invisible by contract. Rxe5 +100; "Nxe5" −400.
    # If SEE is ever made pin-aware, this test forces a contract change.
    fen = "3k4/3n4/8/4p3/8/8/8/3RR1K1 w - - 0 1"
    assert see(fen, "e1e5") == -400


def test_documented_limitation_capture_promotion_valued_as_pawn():
    # exd8=Q wins the rook (+500); Kxd8 recaptures a piece valued at 100
    # (the promotion delta is ignored by contract) → +400.
    fen = "3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1"
    assert see(fen, "e7d8q") == 400


def test_documented_limitation_quiet_promotion_on_attacked_square():
    # Non-capture promotion; Rxe8 takes a 100-valued piece → −100.
    fen = "r6k/4P3/8/8/8/8/8/4K3 w - - 0 1"
    assert see(fen, "e7e8q") == -100


def test_quiet_move_to_attacked_but_defended_square():
    # Rook steps into the enemy rook's fire but the d4 pawn defends:
    # 0; Rxe5 −500... no: Rxe5 then dxe5 recovers the rook → net 0.
    # Pins that quiet-move SEE runs the full swap, not just "is it attacked".
    fen = "4r2k/8/8/8/3P4/8/8/4R1K1 w - - 0 1"
    assert see(fen, "e1e5") == 0


def test_see_accepts_san():
    fen = "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2"
    assert see(fen, "exd5") == 0


def test_see_castling_is_zero_both_notations():
    fen = "5k2/8/8/8/8/8/8/4K2R w K - 0 1"
    assert see(fen, "e1g1") == 0
    assert see(fen, "O-O") == 0


def test_see_garbage_inputs_raise():
    with pytest.raises(ValueError):
        see(Board().fen, "zzzz")
    with pytest.raises(ValueError):
        see(Board().fen, "")
