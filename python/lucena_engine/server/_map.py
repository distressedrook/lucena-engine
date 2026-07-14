"""Mappers between engine values and proto messages.

SAN is the only notation on the wire: PVs come back from Stockfish as UCI and are
converted to SAN via `reads.pv_san` at these boundaries; move *inputs* are SAN and
`Board.apply`/`Board.uci` convert them inward.
"""

from __future__ import annotations

from .. import reads
from ..evalmodel import Score, win_pct_from_score
from .._pb import engine_pb2 as pb


def limit_kwargs(limit_msg, *, default_movetime_ms: int = 1500) -> dict:
    """proto Limit → the `analyse` kwarg (exactly one of movetime_ms/nodes/depth).
    An unset Limit falls back to a production movetime. `threads` is a server-level
    engine setting, not per-call, so it is ignored here (see serve.py)."""
    kind = limit_msg.WhichOneof("kind") if limit_msg is not None else None
    if kind == "movetime_ms":
        return {"movetime_ms": limit_msg.movetime_ms}
    if kind == "nodes":
        return {"nodes": limit_msg.nodes}
    if kind == "depth":
        return {"depth": limit_msg.depth}
    return {"movetime_ms": default_movetime_ms}


def eval_msg(score: Score, glyph: str = "") -> pb.Eval:
    """evalmodel.Score → proto Eval (scored-side POV)."""
    return pb.Eval(cp=score.to_ceiled_cp(), win_pct=round(win_pct_from_score(score), 1), glyph=glyph)


def eval_from_block(block: dict, glyph: str = "") -> pb.Eval:
    return pb.Eval(cp=int(block["cp"]), win_pct=float(block["win_pct"]), glyph=glyph)


def material_msg(m: dict) -> pb.Material:
    return pb.Material(white=m["white"], black=m["black"], net=m["net"], standing=m["standing"])


def score_msg(score: Score) -> pb.Score:
    if score.is_mate:
        return pb.Score(mate=score.mate)
    return pb.Score(cp=score.to_ceiled_cp())


def pieces_msg(board) -> list:
    return [pb.PieceOn(square=p.square, piece=p.piece, color=p.color) for p in board.piece_list()]


def fact_msg(f) -> pb.Fact:
    """engine Fact → proto Fact. The engine owns the sheet, so salience/provenance
    are carried too (the backend may drop them)."""
    return pb.Fact(
        id=f.id, kind=f.kind, squares=list(f.squares), text=f.text,
        concept_id=f.concept_id, salience=f.salience, provenance=f.provenance,
    )


def line_msg(fen: str, ln) -> pb.Line:
    """engine Line → proto Line, PV converted UCI→SAN."""
    return pb.Line(rank=ln.rank, eval=eval_msg(ln.score), pv_san=reads.pv_san(fen, ln.pv))


def positional_msg(pos: dict) -> pb.Positional:
    terms = {}
    for name, t in pos.get("terms", {}).items():
        feats = t.get("features")
        # features are nested dicts; flatten to a compact list of "k=v" strings for the wire.
        flat = []
        if isinstance(feats, dict):
            flat = [f"{k}={v}" for k, v in feats.items()]
        terms[name] = pb.Term(cp=int(t.get("cp", 0)), standing=t.get("standing", ""), features=flat)
    return pb.Positional(
        phase=str(pos.get("phase", "")),
        leads=list(pos.get("leads", [])),
        terms=terms,
    )
