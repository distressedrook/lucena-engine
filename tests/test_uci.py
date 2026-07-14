"""Black-box integration tests for lucena_engine.uci (M2 contract).

Every engine-touching test uses `nodes=` + `threads=1` (NEVER movetime) so
results are reproducible across machines, per the contract's load-bearing
determinism rule. Node counts are kept low; these are fast, not "slow".

Requires a Stockfish 18 binary on $LUCENA_STOCKFISH (or PATH). Tests are
marked `engine` so they can be selected/deselected as a group.
"""

import shutil

import pytest

from lucena_engine import Engine, EngineError, Score

pytestmark = pytest.mark.engine

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
NODES = 100_000

# A clearly-not-startpos middlegame position for a second determinism anchor.
KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

# Terminal positions (engine emits `bestmove (none)`):
# Fool's mate — White to move but already checkmated.
CHECKMATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
# Classic stalemate — Black to move, no legal moves, not in check.
STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"


def _have_stockfish():
    import os

    return bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")


pytestmark = [pytestmark, pytest.mark.skipif(not _have_stockfish(), reason="no stockfish")]


@pytest.fixture
def engine():
    with Engine(threads=1) as e:
        yield e


# ---------------------- construction / version ----------------------


def test_version_assert_happy_path():
    # Default assert_version=True must succeed against a real Stockfish 18.
    with Engine(threads=1) as e:
        a = e.analyse(START, nodes=NODES)
        assert a.best.pv  # sanity: it produced a line


def test_missing_binary_raises_engine_error():
    with pytest.raises(EngineError) as exc:
        Engine(path="/nonexistent/stockfish-binary", assert_version=False)
    # Message must name the path and the env var (contract).
    msg = str(exc.value)
    assert "/nonexistent/stockfish-binary" in msg
    assert "LUCENA_STOCKFISH" in msg


# ---------------------- analyse: limit validation ----------------------


def test_analyse_requires_a_limit(engine):
    with pytest.raises(ValueError):
        engine.analyse(START)


def test_analyse_rejects_two_limits(engine):
    with pytest.raises(ValueError):
        engine.analyse(START, nodes=NODES, depth=8)
    with pytest.raises(ValueError):
        engine.analyse(START, nodes=NODES, movetime_ms=50)
    with pytest.raises(ValueError):
        engine.analyse(START, depth=8, movetime_ms=50)


def test_analyse_limit_validation_precedes_io(engine):
    # ValueError is raised before touching the engine, so the engine stays
    # usable afterward.
    with pytest.raises(ValueError):
        engine.analyse(START)
    a = engine.analyse(START, nodes=NODES)
    assert a.best.pv


# ---------------------- analyse: happy path ----------------------


def test_analyse_returns_analysis_shape(engine):
    a = engine.analyse(START, nodes=NODES)
    assert a.fen == START
    assert len(a.lines) == 1
    assert a.best is a.lines[0]
    assert a.best.rank == 1
    assert isinstance(a.best.score, Score)


def test_best_pv_is_nonempty_uci(engine):
    a = engine.analyse(START, nodes=NODES)
    mv = a.best.pv[0]
    assert isinstance(mv, str)
    # UCI long algebraic: 4 or 5 chars, files a-h, ranks 1-8.
    assert len(mv) in (4, 5)
    assert mv[0] in "abcdefgh" and mv[2] in "abcdefgh"
    assert mv[1] in "12345678" and mv[3] in "12345678"


def test_startpos_score_is_from_side_to_move(engine):
    # White to move at startpos: a small positive cp (white slightly better),
    # certainly not a mate. We assert sign/shape, not an exact value.
    a = engine.analyse(START, nodes=NODES)
    assert a.best.score.mate is None
    assert a.best.score.cp is not None
    assert -100 < a.best.score.cp < 200


# ---------------------- determinism ----------------------


def test_determinism_two_fresh_engines_identical():
    # nodes + threads=1 with fresh hash → identical bestmove/scores/pv.
    def run():
        with Engine(threads=1) as e:
            a = e.analyse(START, nodes=NODES)
            return a.best.score.cp, a.best.score.mate, list(a.best.pv)

    assert run() == run()


def test_determinism_across_new_game(engine):
    # new_game clears hash → repeated analysis of same FEN is identical.
    engine.new_game()
    a = engine.analyse(KIWIPETE, nodes=NODES)
    engine.new_game()
    b = engine.analyse(KIWIPETE, nodes=NODES)
    assert (a.best.score.cp, a.best.score.mate, list(a.best.pv)) == (
        b.best.score.cp,
        b.best.score.mate,
        list(b.best.pv),
    )


# ---------------------- multipv ----------------------


def test_multipv_returns_n_sorted_lines(engine):
    a = engine.analyse(START, nodes=NODES, multipv=3)
    assert len(a.lines) == 3
    assert [ln.rank for ln in a.lines] == [1, 2, 3]
    assert a.best is a.lines[0]
    # every pv non-empty
    assert all(ln.pv for ln in a.lines)


def test_multipv_lines_ranked_best_first(engine):
    # rank 1 must be at least as good (from side-to-move POV) as rank 2, etc.
    a = engine.analyse(START, nodes=NODES, multipv=3)
    ceils = [ln.score.to_ceiled_cp() for ln in a.lines]
    assert ceils == sorted(ceils, reverse=True)


# ---------------------- terminal positions ----------------------


def test_checkmate_position_raises(engine):
    with pytest.raises(EngineError):
        engine.analyse(CHECKMATE_FEN, nodes=NODES)


def test_stalemate_position_raises(engine):
    with pytest.raises(EngineError):
        engine.analyse(STALEMATE_FEN, nodes=NODES)


# ---------------------- lifecycle ----------------------


def test_close_is_idempotent():
    e = Engine(threads=1)
    e.close()
    e.close()  # must not raise
    e.close()


def test_context_manager_closes():
    with Engine(threads=1) as e:
        assert e.analyse(START, nodes=NODES).best.pv
    # exiting the context closed it; a further close is still safe.
    e.close()


# ---------------------- raw_eval ----------------------


def test_raw_eval_reaches_final_evaluation(engine):
    lines = engine.raw_eval(START)
    assert isinstance(lines, list)
    assert any("Final evaluation" in ln for ln in lines)


# ---------------------- CONSCIOUS GAPS (documented for the critic) ----------------------
#
# Not tested here — they need setup this black-box harness can't safely fake:
#   * version-mismatch EngineError (needs a non-SF18 binary reporting a
#     different `id name`).
#   * restart-once-then-propagate on mid-command process death (needs to kill
#     the child mid-`analyse`; can't be done without reaching into internals).
#   * `threads` default = max(2, cpu-2) — not observable through the contract.
