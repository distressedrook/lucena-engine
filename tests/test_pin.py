"""pin detector (detectors.pin.detect_pin) — absolute pins, pure board geometry.

Deterministic and always true (no engine, no reconciliation), so every test is a
plain board assertion.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from lucena_engine.board import Board
from lucena_engine.detectors.pin import detect_pin

CASE02 = "2r4r/4kpp1/p3p2p/7n/Bq3P2/2N2Q1P/1PP3P1/4R1K1 w - - 2 28"   # e6 pawn pinned to Ke7 by Re1
OWN_PIN = "7k/b7/8/8/8/8/5N2/6K1 w - - 0 1"                          # Nf2 pinned to Kg1 by Ba7
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
NON_SLIDER_BEHIND = "6k1/8/8/8/8/4n3/5P2/6K1 w - - 0 1"              # f2 on the ray, but a KNIGHT behind


def test_enemy_pawn_pinned_to_king_grounds_case02_illegal_recapture():
    f = next(f for f in detect_pin(Board(CASE02)) if f.provenance == "pin:e6")
    assert f.kind == "pin"
    assert f.squares == ["e6", "e1"]
    assert "pinned to the king" in f.text and "rook on e1" in f.text
    assert "can't legally move" in f.text            # the exd5-is-illegal grounding
    assert f.concept_id == "pins-skewers"


def test_own_piece_pin_phrased_as_your_king():
    f = next(f for f in detect_pin(Board(OWN_PIN)) if f.provenance == "pin:f2")
    assert "your knight on f2 is pinned to your king" in f.text
    assert f.squares == ["f2", "a7"]                 # [pinned, pinning slider]


def test_no_pins_at_start():
    assert detect_pin(Board(START)) == []


def test_no_pin_when_the_piece_behind_is_not_a_slider():
    # f2 sits on the king's diagonal, but the piece behind it is a knight, not a
    # rook/bishop/queen — no pin.
    assert detect_pin(Board(NON_SLIDER_BEHIND)) == []


def test_pin_is_deterministic():
    assert detect_pin(Board(CASE02)) == detect_pin(Board(CASE02))
