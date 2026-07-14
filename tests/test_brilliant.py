"""Black-box tests for `lucena_engine.brilliant` (!!) classification.

Written from docs/contracts/brilliant-classification.md ONLY. The implementation
(python/lucena/engine/brilliant.py) was NOT read, opened, grepped, or run — it did
not exist when these tests were authored. Chess expectations were grounded with the
product board core (lucena_engine.board.Board: see / legal_moves / apply) and, for the
engine-marked surfacing tests, cross-checked against the M5 MCP surface. python-chess
is NEVER imported here (GPL hygiene).

Sections:
  * is_material_sacrifice  — board-core only, deterministic, no engine.
  * classify_brilliant     — pure decision over win-%, band edges from the constants.
  * is_brilliant           — convenience combiner (board + signals).
  * surfacing              — engine-marked: MCP evaluate_move / gamepass emit
                             "brilliant" only for a sound sacrifice, never over an
                             error, never on a fork. Skipped without Stockfish;
                             determinism split -> nodes-limited, threads=1.

Constants (v1 values, from the contract §"Constants"):
  SOUND_MARGIN = 5.0 · FORCED_FLOOR = 20.0 · ALREADY_WINNING_CEIL = 70.0 · SOUND_FLOOR = 50.0
Bands are half-open exactly as the contract writes them (`>`, `<`, `>=`, `<=`).
"""

import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from lucena_engine.board import Board
from lucena_engine.brilliant import (
    classify_brilliant,
    is_brilliant,
    is_material_sacrifice,
)

# --------------------------------------------------------------------------
# Positions (all grounded with the board core / MCP surface below)
# --------------------------------------------------------------------------

# Legal's Mate: after 5...Bh5, White's Nxe5 (f3e5) is a SOUND sacrifice and the
# engine's best move. see(f3e5) = -200 with legal recaptures dxe5 / Nxe5 on e5.
# Engine (nodes): best Nxe5 ~66.7% win, second best ~60.2% (inside the 20..70 band).
LEGAL_FEN = "r2qkbnr/ppp2ppp/2np4/4p2b/2B1P3/2N2N1P/PPPP1PP1/R1BQK2R w KQkq - 1 6"
LEGAL_SAC_UCI = "f3e5"
LEGAL_SAC_SAN = "Nxe5"

# case-02 fork: Nd5+ (c3d5) has see = -200 but the only recapture (exd5) is ILLEGAL
# (e6 pawn pinned to the e7 king). A fork, NOT a sacrifice -> is_material_sacrifice False.
FORK_FEN = "2r4r/4kpp1/p3p2p/7n/Bq3P2/2N2Q1P/1PP3P1/4R1K1 w - - 2 28"
FORK_UCI = "c3d5"
FORK_SAN = "Nd5+"

# Greek gift that is UNSOUND here (Nf6 defends h7): Bxh7+ (d3h7) is a real capture-sac
# (see = -200, Kxh7 legal) but the engine scores it a blunder -> never brilliant.
GREEK_LOSING_FEN = "r1bq1rk1/pp1nbppp/4pn2/2pp2B1/3P4/2NBPN2/PPP2PPP/R2QK2R w KQ - 0 1"
GREEK_LOSING_UCI = "d3h7"
GREEK_LOSING_SAN = "Bxh7+"

# HANG: White Rxe5 (e1e5) is a plain winning capture, see = +300, no recapture on e5.
HANG_FEN = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"
HANG_WIN_UCI = "e1e5"       # Rxe5, see>0
HANG_WIN_SAN = "Rxe5"
HANG_QUIET_UCI = "g1f1"     # Kf1, a quiet non-capture, see == 0

LEGAL_PGN = """[Event "Legal"]
[White "A"]
[Black "B"]
[Result "*"]
[UserSide "w"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 d6 4. Nc3 Bg4 5. h3 Bh5 6. Nxe5 *
"""

_have_engine = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
requires_engine = pytest.mark.skipif(not _have_engine, reason="no stockfish")


# ==========================================================================
# is_material_sacrifice  (board-core only; pure/deterministic; no engine)
# ==========================================================================


def test_real_capture_sacrifice_is_true_greek_gift():
    # Grounded: Bxh7+ loses material by SEE and Kxh7 is a legal recapture of h7.
    b = Board(GREEK_LOSING_FEN)
    assert b.see(GREEK_LOSING_UCI) < 0
    after = b.apply(GREEK_LOSING_UCI)
    assert any(m.endswith("h7") for m in after.legal_moves())  # legal recapture exists
    assert is_material_sacrifice(b, GREEK_LOSING_UCI) is True


def test_real_capture_sacrifice_is_true_legal_knight():
    # Grounded: Nxe5 loses material by SEE and dxe5 / Nxe5 legally recapture e5.
    b = Board(LEGAL_FEN)
    assert b.see(LEGAL_SAC_UCI) < 0
    after = b.apply(LEGAL_SAC_UCI)
    assert any(m.endswith("e5") for m in after.legal_moves())
    assert is_material_sacrifice(b, LEGAL_SAC_UCI) is True


def test_pinned_recapture_guard_fork_is_not_a_sacrifice():
    # The contractual regression: Nd5+ has see<0 but exd5 is ILLEGAL (pinned e6 pawn),
    # so nothing can legally take the knight -> a fork, not a sacrifice -> False.
    b = Board(FORK_FEN)
    assert b.see(FORK_UCI) < 0                      # SEE-literalism would say "sac"
    after = b.apply(FORK_UCI)
    assert "e6d5" not in after.legal_moves()        # exd5 illegal (pin)
    assert not any(m.endswith("d5") for m in after.legal_moves())  # no legal recapture
    assert is_material_sacrifice(b, FORK_UCI) is False


def test_plain_winning_capture_is_not_a_sacrifice():
    # Rxe5 wins a clean knight (see > 0) -> not a sacrifice.
    b = Board(HANG_FEN)
    assert b.see(HANG_WIN_UCI) > 0
    assert is_material_sacrifice(b, HANG_WIN_UCI) is False


def test_quiet_nonloss_move_is_not_a_sacrifice():
    # A quiet non-capture with see >= 0 is not a *direct* sacrifice (the offered/
    # quiet sac form is a documented v1 non-goal).
    b = Board(HANG_FEN)
    assert b.see(HANG_QUIET_UCI) >= 0
    assert is_material_sacrifice(b, HANG_QUIET_UCI) is False


# ==========================================================================
# classify_brilliant  (pure decision over win-%; no engine, no board)
# ==========================================================================

# A baseline signal tuple that satisfies ALL five conditions -> True:
#   sacrifice True; sound (delta 2 <= 5, played not best); real choice (second 60 > 20);
#   not already winning (second 60 < 70); worthwhile (best 66 >= 50).
BASE = dict(
    is_sacrifice=True,
    best_win=66.0,
    played_win=64.0,
    second_win=60.0,
    played_is_best=False,
)


def _sig(**overrides):
    d = dict(BASE)
    d.update(overrides)
    return d


def test_baseline_all_conditions_hold_is_true():
    assert classify_brilliant(**BASE) is True


def test_flip_sacrifice_false():
    # Condition 1: not a sacrifice -> never brilliant.
    assert classify_brilliant(**_sig(is_sacrifice=False)) is False


def test_flip_sound_false_delta_beyond_margin():
    # Condition 2: not best and best-played delta > SOUND_MARGIN (5) -> not sound.
    assert classify_brilliant(
        **_sig(played_win=60.0, played_is_best=False)  # 66 - 60 = 6 > 5
    ) is False


def test_flip_forced_false_second_win_at_or_below_floor():
    # Condition 3: second_win must be strictly > FORCED_FLOOR (20). Exactly 20 -> forced.
    assert classify_brilliant(**_sig(second_win=20.0)) is False


def test_flip_forced_false_second_win_none():
    # Condition 3: only one legal move (second_win None) -> forced, not a choice.
    assert classify_brilliant(**_sig(second_win=None)) is False


def test_flip_already_winning_false_second_win_at_or_above_ceil():
    # Condition 4: second_win must be strictly < ALREADY_WINNING_CEIL (70). Exactly 70
    # means a non-sac alternative was already winning -> not a brilliancy.
    assert classify_brilliant(
        **_sig(best_win=75.0, played_win=73.0, second_win=70.0)
    ) is False


def test_flip_worthwhile_false_best_win_below_floor():
    # Condition 5: best_win must be >= SOUND_FLOOR (50). Just below -> a mere defensive
    # resource, not a brilliancy.
    assert classify_brilliant(
        **_sig(best_win=49.0, played_win=49.0, played_is_best=True, second_win=30.0)
    ) is False


# ---- exact band edges implied by the constants -----------------------------


def test_sound_edge_delta_exactly_margin_is_sound():
    # best - played == 5.0 == SOUND_MARGIN, played not best -> `<= 5` holds -> sound.
    assert classify_brilliant(
        **_sig(best_win=66.0, played_win=61.0, played_is_best=False)
    ) is True


def test_sound_edge_delta_just_over_margin_not_sound():
    # 66 - 60.9 = 5.1 > 5, played not best -> not sound -> False.
    assert classify_brilliant(
        **_sig(best_win=66.0, played_win=60.9, played_is_best=False)
    ) is False


def test_sound_via_played_is_best_overrides_large_delta():
    # played_is_best True satisfies "sound" regardless of the delta.
    assert classify_brilliant(
        **_sig(best_win=66.0, played_win=50.0, played_is_best=True, second_win=60.0)
    ) is True


def test_forced_floor_edge_just_above_twenty_is_a_choice():
    # second_win 20.5 > 20 -> a real choice (other conditions still hold) -> True.
    assert classify_brilliant(**_sig(second_win=20.5)) is True


def test_already_winning_ceil_edge_just_below_seventy_is_ok():
    # second_win 69.9 < 70 -> not already winning -> True.
    assert classify_brilliant(
        **_sig(best_win=75.0, played_win=73.0, second_win=69.9)
    ) is True


def test_worthwhile_edge_best_win_exactly_floor_is_worthwhile():
    # best_win 50.0 == SOUND_FLOOR -> `>= 50` holds -> worthwhile -> True.
    assert classify_brilliant(
        **_sig(best_win=50.0, played_win=50.0, played_is_best=True, second_win=30.0)
    ) is True


def test_worthwhile_edge_best_win_just_below_floor_is_not():
    assert classify_brilliant(
        **_sig(best_win=49.9, played_win=49.9, played_is_best=True, second_win=30.0)
    ) is False


# ==========================================================================
# is_brilliant  (convenience: sac check on a board + the pure logic)
# ==========================================================================

# Plausible win% signals for Legal's Nxe5 (grounded against the engine below): the
# played move is the engine's best, best ~66.7%, second-best ~60.2% (in-band).
LEGAL_SIGNALS = dict(best_win=66.7, played_win=66.3, second_win=60.2, played_is_best=True)


def test_is_brilliant_true_on_real_sound_sacrifice():
    b = Board(LEGAL_FEN)
    assert is_brilliant(b, LEGAL_SAC_UCI, **LEGAL_SIGNALS) is True


def test_is_brilliant_agrees_with_calling_the_two_directly():
    b = Board(LEGAL_FEN)
    expected = classify_brilliant(
        is_sacrifice=is_material_sacrifice(b, LEGAL_SAC_UCI), **LEGAL_SIGNALS
    )
    assert is_brilliant(b, LEGAL_SAC_UCI, **LEGAL_SIGNALS) == expected is True


def test_is_brilliant_false_on_fork_even_with_winning_signals():
    # The fork's signals could otherwise pass classify, but the sac guard vetoes it.
    b = Board(FORK_FEN)
    assert is_material_sacrifice(b, FORK_UCI) is False
    assert is_brilliant(b, FORK_UCI, **LEGAL_SIGNALS) is False
    # ...and that equals routing the (False) sac verdict through the pure logic.
    assert is_brilliant(b, FORK_UCI, **LEGAL_SIGNALS) == classify_brilliant(
        is_sacrifice=False, **LEGAL_SIGNALS
    )


def test_is_brilliant_false_on_plain_capture():
    # Not a sacrifice -> not brilliant, whatever the win% signals say.
    b = Board(HANG_FEN)
    assert is_brilliant(b, HANG_WIN_UCI, **LEGAL_SIGNALS) is False


# ==========================================================================
# Surfacing (integration; engine-marked, nodes-limited, threads=1)
# ==========================================================================

NODES = 3_000_000


# NOTE: the ToolContext "surfacing" tests (evaluate_and_show brilliant overlay) reach into the
# backend (mcp.state/tools) and were re-homed to the backend test suite in M2. The engine keeps the
# pure classifier tests + the gamepass surfacing test below.


@pytest.mark.engine
@requires_engine
def test_gamepass_surfaces_brilliant_ply(tmp_path):
    # A full-game pass over Legal's opening: the played 6.Nxe5 (ply 11) is a sound
    # sacrifice over an otherwise-ok move -> its ply class is "brilliant".
    from lucena_engine.gamepass import run_pass
    from lucena_engine.uci import Engine

    # run_pass returns the digest directly (persistence is the backend's job, not the engine's).
    with Engine(threads=1) as engine:
        engine.new_game()
        data = run_pass(
            LEGAL_PGN, engine,
            fast_limit={"nodes": 400_000},
            deep_limit={"nodes": 1_200_000},
        )
    nxe5 = next(p for p in data["plies"] if p.get("san") == LEGAL_SAC_SAN)
    assert nxe5["class"] == "brilliant"


# ===========================================================================
# Reconciliation additions — coverage/pin points raised by the critic (Agent B).
# ===========================================================================
from lucena_engine.brilliant import (  # noqa: E402
    classify_brilliant as _cb,
    is_material_sacrifice as _ims,
)


# B#3 — "worthwhile" must gate on played_win, not best_win (the contract fix).
def test_worthwhile_gates_on_played_win_not_best_win():
    common = dict(is_sacrifice=True, best_win=52.0, second_win=45.0, played_is_best=False)
    # played sac reaches only 48% (< SOUND_FLOOR) though best_win is 52 -> not brilliant
    assert _cb(played_win=48.0, **common) is False
    # played sac itself reaches 50% -> brilliant (sound: delta 2 <= margin)
    assert _cb(played_win=50.0, **common) is True


# B#1 — a small-but-real drop sacrifice (a ?!-ish sac) is not "sound" -> not brilliant.
def test_near_best_drop_sacrifice_is_not_sound_enough():
    assert _cb(is_sacrifice=True, best_win=80.0, played_win=72.0,   # delta 8 > SOUND_MARGIN
               second_win=45.0, played_is_best=False) is False


# B#4 — a sacrifice whose ONLY legal recapture is the king is still a real sacrifice.
def test_king_only_recapture_is_a_real_sacrifice():
    # Bd3xh7+: the only reply landing on h7 is Kxh7 (verified: g8h7).
    b = Board("6k1/5ppp/8/8/8/3B4/8/6K1 w - - 0 1")
    assert _ims(b, "d3h7") is True
