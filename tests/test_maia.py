"""Maia-3 human-move predictor (engine/maia.py). Runs the real `maia3-uci`
subprocess, so it's gated on it being installed — set `LUCENA_MAIA` to the
command (e.g. `.venv-maia/bin/maia3-5m`). Predictor only: we assert on *ranked
human moves*, never on Maia's cp/wdl as if it were truth (that's Stockfish's job).
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import pytest

from lucena_engine.maia import MaiaEngine, _parse_multipv

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
# Black to move; bxc4 wins the hanging bishop, Nxe4 is the tempting alternative.
TACTIC = "r1bq1rk1/2pn1p1p/p2b1np1/1p1Np3/2B1P3/5NB1/PPPQ1PPP/2KR3R b - - 1 1"


def _have_maia():
    return bool(os.environ.get("LUCENA_MAIA")) or shutil.which("maia3-5m")


requires_maia = pytest.mark.skipif(not _have_maia(), reason="no maia3 (set LUCENA_MAIA)")

# The policy-emitting production wrapper (tools/maia_policy_uci.py) — stock maia3-5m
# does NOT emit `policy`, so poisoned-line detection is wired specifically to this.
# Gated on the .venv-maia interpreter + the wrapper both existing, independent of
# whatever LUCENA_MAIA happens to point at, so it always tests the real production path.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_POLICY_MAIA = f"{_REPO}/.venv-maia/bin/python {_REPO}/tools/maia_policy_uci.py"


def _have_policy_maia():
    return all(os.path.exists(p) for p in _POLICY_MAIA.split())


requires_policy_maia = pytest.mark.skipif(
    not _have_policy_maia(), reason="no .venv-maia policy wrapper (tools/maia_policy_uci.py)"
)


# -- parser: pure, no subprocess (always runs) -------------------------------

def test_parse_multipv_extracts_ranked_moves():
    lines = [
        "info depth 1 multipv 1 score cp 138 wdl 556 26 418 pv e2e4",
        "info depth 1 multipv 2 score cp 142 wdl 558 26 416 pv d2d4",
        "bestmove e2e4",
    ]
    out = _parse_multipv(lines)
    assert [x["rank"] for x in out] == [1, 2]
    assert out[0]["uci"] == "e2e4" and out[1]["uci"] == "d2d4"
    assert out[0]["cp"] == 138 and out[0]["wdl"] == [556, 26, 418]


def test_parse_multipv_ignores_noise_and_partial_lines():
    lines = ["Maia3 ready", "readyok", "info string hello", "bestmove e2e4"]
    assert _parse_multipv(lines) == []


def test_parse_multipv_extracts_policy_when_present():
    # The policy-emitting wrapper adds `policy <p>` before `pv`; parse it as a float.
    lines = [
        "info depth 1 multipv 1 score cp 138 wdl 556 26 418 policy 0.642815 pv e2e4",
        "info depth 1 multipv 2 score cp 142 wdl 558 26 416 policy 0.241652 pv d2d4",
    ]
    out = _parse_multipv(lines)
    assert out[0]["policy"] == 0.642815 and out[1]["policy"] == 0.241652
    assert out[0]["uci"] == "e2e4"                            # pv still last, still parsed


def test_parse_multipv_omits_policy_when_absent():
    # Stock maia3-5m emits no policy token — the key is simply absent (not 0).
    lines = ["info depth 1 multipv 1 score cp 138 wdl 556 26 418 pv e2e4"]
    assert "policy" not in _parse_multipv(lines)[0]


# -- live engine -------------------------------------------------------------

@pytest.fixture(scope="module")
def maia():
    with MaiaEngine() as m:
        yield m


@requires_maia
def test_starts_and_identifies_as_maia(maia):
    assert "Maia" in maia._name


@requires_maia
def test_top_human_moves_are_ranked_legal_uci(maia):
    mv = maia.top_human_moves(START, 1500, n=4)
    assert len(mv) == 4
    assert [x["rank"] for x in mv] == [1, 2, 3, 4]        # contiguous, most-likely first
    assert all(len(x["uci"]) in (4, 5) for x in mv)       # uci moves
    assert mv[0]["uci"] == "e2e4"                          # 1500 humans open e4 (stable)


@requires_maia
def test_n_is_respected(maia):
    assert len(maia.top_human_moves(START, 1500, n=1)) == 1
    assert len(maia.top_human_moves(START, 1500, n=3)) == 3


@requires_maia
def test_deterministic_at_temperature_zero(maia):
    a = maia.top_human_moves(START, 1500, n=5)
    b = maia.top_human_moves(START, 1500, n=5)
    assert a == b                                          # one forward pass, argmax order


@requires_maia
def test_rating_knob_is_accepted_and_predicts(maia):
    # The rating conditions the prediction; both levels must return valid ranked
    # lists (we don't assert they differ on any single position — not all separate).
    low = maia.top_human_moves(START, 1100, n=5)
    high = maia.top_human_moves(START, 2000, n=5)
    assert len(low) == 5 and len(high) == 5
    assert all("uci" in x for x in low + high)


@requires_maia
def test_predicts_the_human_move_on_a_tactic(maia):
    # The product thesis: a mid-level human's top instinct here is the winning
    # capture, and the tempting Nxe4 is close behind — Maia knows both.
    mv = maia.top_human_moves(TACTIC, 1300, n=5)
    ucis = [x["uci"] for x in mv]
    assert "b5c4" in ucis[:3]                              # bxc4, the right move, is a top pick


# -- the policy-emitting production wrapper ----------------------------------
# Regression guard for the "green tests, dead in prod" bug: find_poisoned_lines
# needs Maia's REAL policy probability, which only tools/maia_policy_uci.py emits.
# The fake-based detector suite can't catch a wrapper that silently drops it.

@pytest.fixture(scope="module")
def policy_maia():
    with MaiaEngine(_POLICY_MAIA) as m:
        yield m


@requires_policy_maia
def test_policy_wrapper_emits_real_probabilities(policy_maia):
    mv = policy_maia.top_human_moves(START, 1500, n=5)
    assert mv, "no moves returned"
    assert all("policy" in m for m in mv), "wrapper must emit a policy per move"
    pols = [m["policy"] for m in mv]
    assert all(0.0 < p <= 1.0 for p in pols)               # real probabilities, never the 0 default
    assert pols == sorted(pols, reverse=True)              # most-likely first
    assert sum(pols) <= 1.0 + 1e-6                         # a genuine (sub)distribution over legal moves
    # NOT rank-derived — the whole point of this slice. A rank-based fake would be
    # evenly spaced; the real head is sharply peaked (e4 dominates at 1500) with
    # non-uniform gaps between consecutive ranks.
    assert pols[0] > 0.4
    assert abs((pols[0] - pols[1]) - (pols[1] - pols[2])) > 1e-3


@requires_policy_maia
def test_policy_is_independent_of_ranking_alone(policy_maia):
    # Two different ratings can share a top-move ORDER yet differ in the actual
    # probabilities — impossible if `policy` were derived from rank.
    a = policy_maia.top_human_moves(START, 1100, n=5)
    b = policy_maia.top_human_moves(START, 2000, n=5)
    if [x["uci"] for x in a] == [x["uci"] for x in b]:     # same order (often true for the opening)
        assert [x["policy"] for x in a] != [x["policy"] for x in b]
