"""Forcing-line tree builder (line_tree). Engine-backed, nodes-limit determinism,
Threads=1; skipped without Stockfish. Serial builder (pool.py is shelved)."""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import pytest

from lucena_engine import Engine
from lucena_engine.line_tree import build_line_tree, count_leaves, is_only_move

DISCOVER = {"nodes": 25_000}
VERIFY = {"nodes": 150_000}

# White has a forced combination: Rb8+ Kh7, Ng6 Rxh4+, Nxh4 winning.
SESSION = "6k1/pR4p1/7p/4N2K/P6P/8/5b1r/8 w - - 1 2"
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _have_sf():
    return bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")


requires_engine = pytest.mark.skipif(not _have_sf(), reason="no stockfish")


@pytest.fixture
def engine():
    with Engine(threads=1) as e:
        e.new_game()
        yield e


def _has_branching_reply(node) -> bool:
    if node.get("kind") == "reply":
        return len(node["defenses"]) > 1 or any(
            _has_branching_reply(d["then"]) for d in node["defenses"])
    if node.get("kind") == "solve":
        return _has_branching_reply(node["after"])
    return False


# A "win the queen" fork whose won position ALSO has a forced mate. The drill must stop when the
# tactic is over (many moves win), not drill the incidental mate net — that ballooned this puzzle to
# ~66 solve nodes / 23 forced moves, so it never concluded and the poisoned reveal never fired.
FORK_THEN_MATE = "1k5r/4q3/1pp5/3bNp2/6p1/P5P1/1P3P2/3QRK2 w - - 0 1"


def _solve_nodes(node) -> int:
    c = 1 if node.get("kind") in ("solve", "mate") else 0
    if node.get("after"):
        c += _solve_nodes(node["after"])
    for d in (node.get("defenses") or []):
        c += _solve_nodes(d["then"])
    for o in (node.get("options") or []):
        c += _solve_nodes(o["then"])
    return c


@requires_engine
def test_stops_when_multiple_moves_win_not_the_mate_net(engine):
    tree = build_line_tree(FORK_THEN_MATE, engine, discover=DISCOVER, verify=VERIFY)
    assert tree["root"]["kind"] == "solve" and tree["root"]["expect_san"] == "Qxd5"
    n = _solve_nodes(tree["root"])
    assert n <= 15, f"the drill should stop when multiple moves win; got {n} solve nodes (mate-net blowup)"


@requires_engine
def test_root_is_the_forced_only_move(engine):
    tree = build_line_tree(SESSION, engine, discover=DISCOVER, verify=VERIFY)
    assert tree["side_to_solve"] == "white"
    root = tree["root"]
    assert root["kind"] == "solve"          # exactly one move wins -> a FIND node
    assert root["expect_san"] == "Rb8+"     # the forced only-move


@requires_engine
def test_deterministic(engine):
    a = build_line_tree(SESSION, engine, discover=DISCOVER, verify=VERIFY)
    engine.new_game()
    b = build_line_tree(SESSION, engine, discover=DISCOVER, verify=VERIFY)
    assert a == b


# A real mate-in-5 (chess.com) the shallow discover pass scores ~0.0 for: it only
# surfaces via the verify-escalation + narrow-multipv search. Guards BOTH that the
# deep tactic is detected AND that it stays byte-reproducible (Threads=1 + fixed
# nodes) — the path a wide multipv / no-escalation build silently got wrong.
DEEP_TACTIC = "r1b1kb1Q/pp3p1p/2p3p1/3pN3/3PnP2/2B5/P3q1PP/RN3RK1 b q - 1 1"


@requires_engine
def test_deep_tactic_escalates_and_is_deterministic(engine):
    trees = []
    for _ in range(4):
        engine.new_game()
        trees.append(build_line_tree(DEEP_TACTIC, engine, discover=DISCOVER, verify=VERIFY))
    root = trees[0]["root"]
    assert root["kind"] == "mate" and root["mate_in"] == 5   # escalation found the deep win
    assert all(t == trees[0] for t in trees[1:])             # every rebuild identical


@requires_engine
def test_it_is_a_tree_not_a_line(engine):
    # the combination branches on the opponent's defenses (play through them all)
    tree = build_line_tree(SESSION, engine, discover=DISCOVER, verify=VERIFY)
    assert count_leaves(tree["root"]) >= 2
    assert _has_branching_reply(tree["root"])


@requires_engine
def test_solve_alternates_with_reply(engine):
    # a FIND node's continuation is always an opponent (reply) node
    tree = build_line_tree(SESSION, engine, discover=DISCOVER, verify=VERIFY)
    root = tree["root"]
    assert root["after"]["kind"] in ("reply", "done")


# ---- drillable slice: result-band floor (win OR draw), separation guard, reasons ----

# `drillable` is the caller's gate (tools.py): root["kind"] in ("solve", "mate").
def _drillable(root) -> bool:
    return root["kind"] in ("solve", "mate")


# A validated real HOLD puzzle (contract worked example): Black is a hair worse and
# ONLY h3 keeps the draw — every alternative drops out of the hold band into a loss.
# Its holding move h3 sits at ~45-47 win% (above HOLDABLE_FLOOR 40 + HOLD_SEPARATION 3),
# so it clears the separation guard and drills. Before this slice it read as converted.
HOLD_PUZZLE = "2b5/8/pp4p1/3kP1P1/5K1p/P7/1PB5/8 b - - 2 41"


@requires_engine
def test_hold_puzzle_is_drillable_solve(engine):
    # a defensive hold (best in the HOLD band, not the WIN band) still produces a
    # `solve` root with the single holding move to find -> drillable.
    tree = build_line_tree(HOLD_PUZZLE, engine, discover=DISCOVER, verify=VERIFY)
    assert tree["side_to_solve"] == "black"
    root = tree["root"]
    assert root["kind"] == "solve"
    assert root["expect_uci"] == "h4h3"       # the only move that holds the draw
    assert _drillable(root)                    # a HOLD solve drills with no caller change


@requires_engine
def test_winning_only_move_still_drills(engine):
    # WIN band is unchanged by the slice: exactly one move >= WIN_BAR (67) -> solve.
    tree = build_line_tree(SESSION, engine, discover=DISCOVER, verify=VERIFY)
    root = tree["root"]
    assert root["kind"] == "solve"
    assert root["expect_uci"] == "b7b8"        # Rb8+, the winning only-move
    assert root["win_pct"] >= 67.0             # sits in the WIN band (>= WIN_BAR)
    assert _drillable(root)


@requires_engine
def test_balanced_start_is_converted_not_drillable(engine):
    # regression guard: a quiet, balanced position (best in-band, several moves clear
    # the floor) is NOT a puzzle -> done/`converted`, never solve or mate. Reconciled
    # to the drillable slice: the old `not_winning` reason is now `lost` (best below
    # HOLDABLE_FLOOR); a balanced start (best ~53%) is always `converted`.
    tree = build_line_tree(START, engine, discover=DISCOVER, verify=VERIFY)
    root = tree["root"]
    assert root["kind"] == "done"
    assert root["reason"] == "converted"       # result banked, nothing unique to find
    assert not _drillable(root)


# A dead-drawn symmetric rook endgame: many moves hold the draw -> `converted`, the
# quiet-position regression guard the slice must not turn drillable.
DEAD_DRAW = "4k3/4r3/8/8/8/8/4R3/4K3 w - - 0 1"


@requires_engine
def test_dead_drawn_endgame_is_converted_not_drillable(engine):
    tree = build_line_tree(DEAD_DRAW, engine, discover=DISCOVER, verify=VERIFY)
    root = tree["root"]
    assert root["kind"] == "done"
    assert root["reason"] == "converted"       # 2+ moves clear the HOLDABLE_FLOOR
    assert not _drillable(root)


# White is in check, down a rook (K+R vs K): losing even with best play. best_win is
# well below HOLDABLE_FLOOR (40), so the position is not a puzzle.
LOST_POSITION = "4k3/8/8/8/8/8/8/r3K3 w - - 0 1"


@requires_engine
def test_lost_position_reason_is_lost_not_drillable(engine):
    # `lost` REPLACES the old `not_winning`: best move below HOLDABLE_FLOOR.
    tree = build_line_tree(LOST_POSITION, engine, discover=DISCOVER, verify=VERIFY)
    root = tree["root"]
    assert root["kind"] == "done"
    assert root["reason"] == "lost"            # doesn't hold even with best play
    assert not _drillable(root)


# White to move, Qxf7 is checkmate (the Scholar's-mate finish) — a forced mate.
SCHOLAR_MATE1 = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def _reaches_mate(node) -> bool:
    if node.get("kind") == "done":
        return node["reason"] == "mate"
    if node.get("kind") == "mate":
        return any(_reaches_mate(o["then"]) for o in node["options"])
    if node.get("kind") == "solve":
        return _reaches_mate(node["after"])
    if node.get("kind") == "reply":
        return any(_reaches_mate(d["then"]) for d in node["defenses"])
    return False


@requires_engine
def test_forced_mate_is_caught_to_checkmate(engine):
    # a forced mate must be caught to #, overriding the material rules
    tree = build_line_tree(SCHOLAR_MATE1, engine, discover=DISCOVER, verify=VERIFY)
    root = tree["root"]
    assert root["kind"] == "mate"
    assert root["mate_in"] == 1
    assert any(o["san"] == "Qxf7#" for o in root["options"])
    assert _reaches_mate(root)          # the tree walks to an actual checkmate leaf


# ---- is_only_move: the drillable rule as a pure, engine-free predicate ----
# Thresholds from the contract: WIN_BAR=67.0, HOLDABLE_FLOOR=40.0, HOLD_SEPARATION=3.0.
# win_pcts are side-to-move POV, best-first. Every expectation is traced to the
# contract's `is_only_move` section (band floors + separation guard + boundaries).


def test_is_only_move_empty_is_false():
    # [] -> False (no move, no floor).
    assert is_only_move([]) is False


@pytest.mark.parametrize("win_pcts", [[30, 10], [39.9, 5]])
def test_is_only_move_lost_best_below_floor(win_pcts):
    # best < HOLDABLE_FLOOR (40) -> False (lost even with best play).
    assert is_only_move(win_pcts) is False


@pytest.mark.parametrize("win_pcts", [[41, 20], [42.9, 10]])
def test_is_only_move_separation_guard_blocks_marginal_hold(win_pcts):
    # THE separation-guard blocker: best in [40, 43) is NOT an only-move even though
    # exactly one value clears 40 -- a hold hovering at the floor is eval-noise.
    assert is_only_move(win_pcts) is False


@pytest.mark.parametrize("win_pcts", [[43.0, 20], [44, 37]])
def test_is_only_move_hold_drills_once_clear_of_guard(win_pcts):
    # best >= 43 (floor + separation, inclusive) and exactly one >= 40 -> True.
    # 43.0 pins the inclusive separation boundary.
    assert is_only_move(win_pcts) is True


@pytest.mark.parametrize("win_pcts", [[50, 45], [44, 41]])
def test_is_only_move_hold_band_requires_uniqueness(win_pcts):
    # best clears the guard but TWO values >= HOLDABLE_FLOOR -> False (result banked,
    # nothing unique to hold).
    assert is_only_move(win_pcts) is False


def test_is_only_move_win_band_unique_no_separation_margin():
    # best >= WIN_BAR (67): True iff exactly one value >= WIN_BAR, with NO separation
    # margin (win band is exempt from the guard).
    assert is_only_move([85, 55]) is True      # one win, clean
    assert is_only_move([70, 68]) is False     # two moves >= 67 -> banked
    assert is_only_move([67, 40]) is True       # 67.0 pins the inclusive WIN_BAR floor


def test_is_only_move_band_transition_below_win_bar_is_hold():
    # Documented band-transition: best 66.9 is < WIN_BAR, so it falls into the HOLD
    # band (floor 40). It clears the separation guard (66.9 >= 43) and exactly one
    # value is >= 40 (20 is not) -> True. A drop of 0.1 below WIN_BAR does not make it
    # non-drillable; it re-evaluates under the hold rule.
    assert is_only_move([66.9, 20]) is True
