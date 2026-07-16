"""Truth service — Stockfish + board core, no Maia.

Each RPC re-composes a chess-truth response from engine primitives (the composition
that used to live in the backend's tools.py). Move inputs are SAN (Board converts
inward); every PV is converted UCI→SAN on the way out.
"""

from __future__ import annotations

import re

import grpc

from ..board import Board
from ..evalmodel import Glyph, Score, win_pct_from_score
from ..facts import build_fact_sheet
from ..hints import derive_hints
from ..brilliant import is_brilliant
from ..positional import analyze_positional
from .. import reads, detect, pgn as pgnmod
from .._fen import captured_piece
from .._pb import engine_pb2 as pb
from .._pb import engine_pb2_grpc as pbg
from . import _map as M

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
_CLASS = {
    Glyph.OK: "ok", Glyph.DUBIOUS: "dubious", Glyph.MISTAKE: "mistake",
    Glyph.BLUNDER: "blunder", Glyph.ONLY_MOVE: "only_move",
}
# "opening" rides along with every focus: it is CONTEXT for whatever was asked, not a tactical
# observation competing with the others, and it is the one fact a client cannot re-derive from the
# position. Filtering it out means a narrow focus silently loses it.
_THREAT_KINDS = {"threat", "hanging", "opening"}


def _resolve_uci(board: Board, move: str) -> str:
    return move if _UCI_RE.match(move) else board.uci(move)


class TruthServicer(pbg.TruthServicer):
    def __init__(self, engines, info):
        self._engines = engines      # EngineHolder (lock-guarded shared Stockfish)
        self._info = info            # dict: engine_version, stockfish_major, maia_available

    # -- board geometry / validation (no engine) --------------------------
    def ValidateFen(self, request, context):
        try:
            b = Board(request.fen)
            return pb.ValidateResp(legal=True, side_to_move=b.side_to_move)
        except Exception as e:  # noqa: BLE001 — surface as a structured field, not a crash
            return pb.ValidateResp(legal=False, error=str(e))

    def LegalMoves(self, request, context):
        b = self._board(request.fen, context)
        return pb.MovesResp(san=[b.san(u) for u in b.legal_moves()])

    def Apply(self, request, context):
        b = self._board(request.fen, context)
        try:
            after = b.apply(request.move)
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"illegal move: {e}")
        return pb.Position(fen=after.fen)

    def DetectFens(self, request, context):
        return pb.DetectFensResp(fens=detect.detect_fens(request.text))

    def ParsePgn(self, request, context):
        try:
            game = pgnmod.parse_pgn(request.text)
        except Exception as e:  # noqa: BLE001 — PgnError etc.
            return pb.ParsePgnResp(ok=False, error=str(e))
        plies = [
            pb.Ply(ply=p.ply, move_no=p.move_no, side=p.side, san=p.san,
                   fen_before=p.fen_before, fen_after=p.fen_after)
            for p in game.plies
        ]
        return pb.ParsePgnResp(ok=True, result=game.result, start_fen=game.start_fen,
                               headers=dict(game.headers), plies=plies)

    def GetInfo(self, request, context):
        return pb.InfoResp(
            engine_version=self._info["engine_version"],
            stockfish_major=self._info["stockfish_major"],
            maia_available=self._info["maia_available"],
        )

    def AnnotateGame(self, request, context):
        context.abort(grpc.StatusCode.UNIMPLEMENTED,
                      "AnnotateGame is a parked skeleton (see docs/grounding-engine-api.md)")

    # -- Stockfish-backed reads -------------------------------------------
    def Analyze(self, request, context):
        b = self._board(request.fen, context)
        lim = M.limit_kwargs(request.limit)
        fact_lim = M.limit_kwargs(request.fact_limit, default_movetime_ms=300)
        multipv = request.multipv or 2
        top_facts = request.top_facts or 5
        with self._engines.acquire() as engine:
            try:
                analysis = engine.analyse(request.fen, multipv=multipv, **lim)
            except Exception as e:  # noqa: BLE001 — terminal/no scored lines
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, f"cannot analyze: {e}")
            facts = build_fact_sheet(b, engine, top_n=top_facts, **fact_lim)

        if request.focus == pb.THREATS:
            facts = [f for f in facts if f.kind in _THREAT_KINDS]

        resp = pb.AnalyzeResp(
            fen=request.fen, side_to_move=b.side_to_move,
            pieces=M.pieces_msg(b),
            material=M.material_msg(reads.material(b)),
            eval=M.eval_msg(analysis.best.score),
            lines=[M.line_msg(request.fen, ln) for ln in analysis.lines],
            facts=[M.fact_msg(f) for f in facts],
        )
        if request.focus == pb.POSITIONAL:
            resp.positional.CopyFrom(M.positional_msg(analyze_positional(b)))
        return resp

    def Hints(self, request, context):
        b = self._board(request.fen, context)
        lim = M.limit_kwargs(request.limit)
        with self._engines.acquire() as engine:
            analysis = engine.analyse(request.fen, multipv=1, **lim)
        hints = derive_hints(b, analysis)
        best = reads.pv_san(request.fen, analysis.best.pv[:1])
        return pb.HintsResp(
            fen=request.fen,
            best_san=best[0] if best else "",
            hints=[pb.Hint(rung=h.rung, text=h.text, squares=list(h.squares),
                           provenance=h.provenance) for h in hints],
        )

    def ExploreLine(self, request, context):
        b = self._board(request.fen, context)
        moves = list(request.moves)
        if not (1 <= len(moves) <= 12):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "ExploreLine takes 1..12 moves")
        played = []
        try:
            for m in moves:
                uci = _resolve_uci(b, m)
                played.append(b.san(uci))
                b = b.apply(uci)
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"illegal line at move {len(played)+1}: {e}")
        end_fen = b.fen
        terminal = ""
        if not b.legal_moves():
            terminal = "checkmate" if b.in_check else "stalemate"
        resp = pb.ExploreResp(end_fen=end_fen, side_to_move=b.side_to_move,
                              line_san=played, terminal=terminal)
        if request.analyze and not terminal:
            lim = M.limit_kwargs(request.limit)
            with self._engines.acquire() as engine:
                analysis = engine.analyse(end_fen, multipv=1, **lim)
            score = analysis.best.score
            best = reads.pv_san(end_fen, analysis.best.pv[:1])
            resp.material.CopyFrom(M.material_msg(reads.material(b)))
            resp.eval.CopyFrom(M.eval_msg(score))
            resp.best_san = best[0] if best else ""
            resp.pv_san.extend(reads.pv_san(end_fen, analysis.best.pv))
            if score.is_mate:
                resp.mate_in = score.mate
        return resp

    def Evaluate(self, request, context):
        b = self._board(request.fen, context)
        moves = list(request.moves)
        if not moves:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Evaluate needs at least one move")
        lim = M.limit_kwargs(request.limit)
        if len(moves) == 1:
            return self._evaluate_one(b, request.fen, moves[0], lim, context)
        return self._evaluate_many(b, request.fen, moves, lim, context)

    # -- Evaluate helpers --------------------------------------------------
    def _evaluate_one(self, board, fen, raw, lim, context):
        try:
            uci = _resolve_uci(board, raw)
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"illegal move: {e}")
        with self._engines.acquire() as engine:
            before = engine.analyse(fen, multipv=2, **lim)
            best_from_mover = before.best.score
            second = before.lines[1].score if len(before.lines) > 1 else None
            best_wp = win_pct_from_score(best_from_mover)
            after_board = board.apply(uci)
            played_is_best = bool(before.best.pv) and uci == before.best.pv[0]
            refutation = []
            if not after_board.legal_moves():
                if after_board.in_check:
                    glyph, played_wp, played_cp = Glyph.OK, 100.0, 1000
                else:
                    glyph, _ = _classify(best_from_mover, Score(cp=0), second, played_is_best)
                    played_wp, played_cp = 50.0, 0
            else:
                after = engine.analyse(after_board.fen, multipv=1, **lim)
                played_result = after.best.score
                glyph, _ = _classify(best_from_mover, played_result, second, played_is_best)
                mover_view = played_result.negated()
                played_wp = win_pct_from_score(mover_view)
                played_cp = mover_view.to_ceiled_cp()
                refutation = reads.pv_san(after_board.fen, after.best.pv)
            facts = build_fact_sheet(board, engine, top_n=5, movetime_ms=300)
            best_pv_san = reads.pv_san(fen, before.best.pv)

        cls = _CLASS.get(glyph, "ok")
        if cls in ("ok", "only_move") and is_brilliant(
            board, uci, best_win=best_wp, played_win=played_wp,
            second_win=(win_pct_from_score(second) if second else None),
            played_is_best=played_is_best,
        ):
            cls = "brilliant"
        resp = pb.EvaluateResp(
            fen=fen, side_to_move=board.side_to_move,
            material=M.material_msg(reads.material(board)),
            san=board.san(uci), captured=captured_piece(fen, uci) or "",
            glyph=str(glyph.value),
            delta_win_pct=round(played_wp - best_wp, 1),
            eval=pb.Eval(cp=played_cp, win_pct=round(played_wp, 1)),
            best=pb.BestMove(san=(best_pv_san[0] if best_pv_san else ""),
                             pv_san=best_pv_san, eval=M.eval_msg(best_from_mover)),
            refutation_pv=refutation,
            facts=[M.fact_msg(f) for f in facts],
        )
        setattr(resp, "class", cls)   # 'class' is a Python keyword — set it by name
        return resp

    def _evaluate_many(self, board, fen, moves, lim, context):
        ranked = []
        for raw in moves[:4]:
            try:
                uci = _resolve_uci(board, raw)
                after_board = board.apply(uci)
            except Exception as e:  # noqa: BLE001
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"illegal move {raw}: {e}")
            with self._engines.acquire() as engine:
                before_best = engine.analyse(fen, multipv=1, **lim).best.score
                if after_board.legal_moves():
                    played = engine.analyse(after_board.fen, multipv=1, **lim).best.score.negated()
                else:
                    played = Score(mate=0) if after_board.in_check else Score(cp=0)
            dwp = round(win_pct_from_score(played) - win_pct_from_score(before_best), 1)
            ranked.append((dwp, pb.MoveRank(san=board.san(uci), eval=M.eval_msg(played),
                                            delta_win_pct=dwp)))
        ranked.sort(key=lambda r: r[0], reverse=True)
        verdict = " > ".join(r[1].san for r in ranked)
        return pb.EvaluateResp(fen=fen, side_to_move=board.side_to_move,
                               material=M.material_msg(reads.material(board)),
                               ranked=[r[1] for r in ranked], verdict=verdict)

    # -- shared -----------------------------------------------------------
    def _board(self, fen, context) -> Board:
        try:
            return Board(fen)
        except Exception as e:  # noqa: BLE001
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"illegal fen: {e}")


def _classify(best_from_mover, played_result, second, played_is_best):
    from ..evalmodel import classify
    return classify(best_from_mover, played_result,
                    second_best_from_mover=second, played_is_best=played_is_best)
