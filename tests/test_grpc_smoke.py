"""E2 smoke: start the gRPC server and exercise the core RPCs end-to-end.

Asserts the two guarantees that matter: SAN-only on the wire, and determinism
(nodes + threads=1 → identical results). Engine-backed RPCs need Stockfish.
"""

import os
import re
import shutil

import grpc
import pytest

from lucena_engine.server.serve import build_server
from lucena_engine._pb import engine_pb2 as pb
from lucena_engine._pb import engine_pb2_grpc as pbg

STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
_UCI_SHAPE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")

_have_sf = bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")
requires_engine = pytest.mark.skipif(not _have_sf, reason="no stockfish")


@pytest.fixture(scope="module")
def stubs():
    server, engines = build_server(port=50251, threads=1)
    server.start()
    channel = grpc.insecure_channel("localhost:50251")
    yield pbg.TruthStub(channel), pbg.BehaviourStub(channel)
    channel.close()
    server.stop(0)
    engines.close()


def test_validate_fen(stubs):
    truth, _ = stubs
    ok = truth.ValidateFen(pb.Position(fen=STARTPOS))
    assert ok.legal and ok.side_to_move == "white"
    bad = truth.ValidateFen(pb.Position(fen="not a fen"))
    assert not bad.legal and bad.error


def test_legal_moves_are_san(stubs):
    truth, _ = stubs
    r = truth.LegalMoves(pb.Position(fen=STARTPOS))
    assert len(r.san) == 20
    assert "e4" in r.san and "Nf3" in r.san
    for mv in r.san:
        assert not _UCI_SHAPE.match(mv), f"UCI leaked: {mv}"


def test_apply_is_san_in_fen_out(stubs):
    truth, _ = stubs
    after = truth.Apply(pb.ApplyReq(fen=STARTPOS, move="e4"))
    assert after.fen.split()[1] == "b"   # black to move after 1.e4


def test_parse_pgn(stubs):
    truth, _ = stubs
    r = truth.ParsePgn(pb.TextReq(text="1. e4 e5 2. Nf3 Nc6 *"))
    assert r.ok and len(r.plies) == 4
    assert r.plies[0].san == "e4" and r.plies[2].san == "Nf3"
    for p in r.plies:
        assert not _UCI_SHAPE.match(p.san)


def test_detect_fens(stubs):
    truth, _ = stubs
    r = truth.DetectFens(pb.TextReq(text=f"look at this: {STARTPOS} okay?"))
    assert any(f.startswith("rnbqkbnr") for f in r.fens)


def test_annotate_game_is_parked(stubs):
    truth, _ = stubs
    with pytest.raises(grpc.RpcError) as ei:
        truth.AnnotateGame(pb.TextReq(text="1. e4 e5 *"))
    assert ei.value.code() == grpc.StatusCode.UNIMPLEMENTED


@requires_engine
def test_get_info(stubs):
    truth, _ = stubs
    info = truth.GetInfo(pb.Empty())
    assert info.stockfish_major == 18
    assert info.engine_version.startswith("lucena-engine")


@requires_engine
def test_analyze_san_only_and_deterministic(stubs):
    truth, _ = stubs
    req = pb.AnalyzeReq(
        fen=STARTPOS,
        limit=pb.Limit(nodes=200_000, threads=1),
        fact_limit=pb.Limit(nodes=100_000, threads=1),
        multipv=2, top_facts=5,
    )
    a = truth.Analyze(req)
    assert a.side_to_move == "white"
    assert a.material.standing == "material is even"
    assert len(a.pieces) == 32
    assert len(a.lines) == 2
    for ln in a.lines:
        assert ln.pv_san, "empty pv"
        for mv in ln.pv_san:
            assert not _UCI_SHAPE.match(mv), f"UCI leaked in pv: {mv}"
    # determinism: same nodes-limited request twice -> identical best line
    b = truth.Analyze(req)
    assert list(a.lines[0].pv_san) == list(b.lines[0].pv_san)


@requires_engine
def test_evaluate_one_move(stubs):
    truth, _ = stubs
    r = truth.Evaluate(pb.EvaluateReq(
        fen=STARTPOS, moves=["e4"], limit=pb.Limit(nodes=200_000, threads=1)))
    assert r.san == "e4"
    assert getattr(r, "class") in ("ok", "only_move", "dubious", "mistake", "blunder", "brilliant")
    assert r.best.san and not _UCI_SHAPE.match(r.best.san)
    for mv in r.best.pv_san:
        assert not _UCI_SHAPE.match(mv)


@requires_engine
def test_explore_line(stubs):
    truth, _ = stubs
    r = truth.ExploreLine(pb.ExploreReq(
        fen=STARTPOS, moves=["e4", "e5", "Nf3"], analyze=True,
        limit=pb.Limit(nodes=200_000, threads=1)))
    assert list(r.line_san) == ["e4", "e5", "Nf3"]
    assert r.side_to_move == "black"
    assert r.best_san and not _UCI_SHAPE.match(r.best_san)
    assert r.terminal == ""


@requires_engine
def test_hints_returns_best_san(stubs):
    truth, _ = stubs
    r = truth.Hints(pb.HintsReq(fen=STARTPOS, limit=pb.Limit(nodes=200_000, threads=1)))
    assert r.best_san and not _UCI_SHAPE.match(r.best_san)
    # hints may be empty at the opening (no tactical handle) — that's valid.
    assert isinstance(list(r.hints), list)
