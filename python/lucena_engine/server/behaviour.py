"""Behaviour service — Maia human-move prediction. Optional; never returns an eval
to the player. Degrades to UNAVAILABLE when LUCENA_MAIA is not configured.
"""

from __future__ import annotations

import grpc

from ..board import Board
from ..evalmodel import win_pct_from_score
from ..poisoned_line_detector import find_poisoned_lines
from .._pb import engine_pb2 as pb
from .._pb import engine_pb2_grpc as pbg


class BehaviourServicer(pbg.BehaviourServicer):
    def __init__(self, engines, maia):
        self._engines = engines
        self._maia = maia    # MaiaHolder (lazy)

    def _get_maia(self, context):
        try:
            return self._maia.get()
        except Exception as e:  # noqa: BLE001 — no LUCENA_MAIA / wrapper missing
            context.abort(grpc.StatusCode.UNAVAILABLE, f"Maia unavailable: {e}")

    def TopHumanMoves(self, request, context):
        maia = self._get_maia(context)
        b = Board(request.fen)
        n = request.n or 5
        oppo = request.oppo_rating or None
        try:
            picks = maia.top_human_moves(request.fen, request.rating, n=n, oppo_rating=oppo)
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.UNAVAILABLE, f"Maia query failed: {e}")
        moves = []
        for m in picks:
            mm = pb.MaiaMove(san=b.san(m["uci"]), rank=m.get("rank", 0), policy=m.get("policy", 0.0))
            if "wdl" in m:
                mm.wdl.extend(m["wdl"])
            if "mate" in m:
                mm.eval.mate = m["mate"]
            elif "cp" in m:
                mm.eval.cp = m["cp"]
            moves.append(mm)
        return pb.MaiaResp(moves=moves)

    def CommonMistakes(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED,
                      "CommonMistakes not yet ported (see docs/grounding-engine-api.md)")

    def PoisonedLine(self, request, context):
        maia = self._get_maia(context)
        fen = request.fen
        board0 = Board(fen)
        mover = board0.side_to_move
        with self._engines.acquire() as engine:
            nres = find_poisoned_lines(fen, engine, maia, rating=request.rating)
            if not nres.get("has_poisoned_line") or not nres.get("temptations"):
                return pb.PoisonedLineResp(fen=fen, has_poisoned_line=False)
            top = nres["temptations"][0]
            seed_san = top["seeds"][0]
            seed_uci = board0.uci(seed_san)
            board = board0.apply(seed_uci)
            steps = [pb.PoisonedStep(san=seed_san, fen=board.fen)]
            for _ in range(10):
                if not board.legal_moves():
                    break
                if board.side_to_move != mover:            # defender: engine's best reply
                    engine.new_game()
                    a = engine.analyse(board.fen, multipv=1, nodes=200_000)
                    uci = a.best.pv[0]
                    decisive = win_pct_from_score(a.best.score) >= 90.0
                else:                                       # human greedy: Maia top-1
                    tops = maia.top_human_moves(board.fen, request.rating, n=1)
                    if not tops:
                        break
                    uci = tops[0]["uci"]
                    decisive = False
                san = board.san(uci)
                board = board.apply(uci)
                steps.append(pb.PoisonedStep(san=san, fen=board.fen))
                if decisive:
                    break
        return pb.PoisonedLineResp(fen=fen, has_poisoned_line=True, poisoned_line=steps,
                                   fatal=top.get("fatal", ""), idea=top.get("idea", ""))
