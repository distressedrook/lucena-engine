"""The coach's grounded briefing — deterministic natural-language analysis.

`build_analysis(board, engine)` assembles ONE readable, number-bearing briefing
from all the raw calculated material: the engine eval, the material balance, the
five-term positional read, and the tactical fact sheet. It is *a function of the
raw facts* — no LLM in its production, same board ⇒ same briefing.

It is the coach's INPUT, not its output: the coach reads it to understand the
position, then interprets it into player-facing coaching (translating numbers to
meaning). So — unlike a beat — the briefing may carry things the coach must never
say aloud: centipawns, win%. It may be surfaced in the UI or not; that is a
display choice, not part of what it is.

Every line is one grounded statement in plain English (all White's point of view,
so the coach never has to flip perspective). The tactical facts arrive already in
natural language from the fact sheet; the positional standings already read as
sentences; this module stitches them into a briefing and prepends the eval.
"""

from __future__ import annotations

from .evalmodel import win_pct_from_score
from .facts import build_fact_sheet
from .positional import analyze_positional


def _cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


def _eval_line(board, score) -> str:
    """Turn + verdict + the raw numbers (White POV), mate-aware."""
    stm = board.side_to_move
    turn = "White" if stm == "white" else "Black"
    if score.is_mate:
        if score.mate == 0:
            return f"{turn} to move and is checkmated."
        # `mate` is signed from the side-to-move POV: >0 means the mover delivers it
        mover = turn if score.mate > 0 else ("Black" if stm == "white" else "White")
        return f"{turn} to move; {mover} has a forced mate in {abs(score.mate)}."
    wp_stm = win_pct_from_score(score)
    white_wp = wp_stm if stm == "white" else 100 - wp_stm
    white_cp = score.to_ceiled_cp() if stm == "white" else -score.to_ceiled_cp()
    d = white_wp - 50
    if abs(d) <= 5:
        verdict = "the position is roughly equal"
    else:
        leader = "White" if d > 0 else "Black"
        aw = white_wp if d > 0 else 100 - white_wp
        band = ("is completely winning" if aw >= 90 else
                "is winning" if aw >= 75 else
                "is clearly better" if aw >= 60 else
                "is slightly better")
        verdict = f"{leader} {band}"
    return (f"{turn} to move; {verdict} "
            f"(eval {white_cp / 100:+.1f}, White {round(white_wp)}%).")


def assemble_analysis(board, best_score, pos, facts) -> list[str]:
    """Pure assembly of the briefing from already-computed pieces — the eval
    `best_score` (Score), the `analyze_positional` dict `pos`, and the fact-sheet
    list `facts`. No engine call, so the MCP layer can reuse facts it already
    built (for the board arrows) instead of recomputing the sheet."""
    lines = [_eval_line(board, best_score)]
    lines.append(_cap(pos["terms"]["material"]["standing"]) + ".")
    # the four strategic terms, leads first (material handled above)
    strategic = ["king_safety", "activity", "pawns", "center"]
    ordered = [t for t in pos["leads"] if t in strategic]
    ordered += [t for t in strategic if t not in ordered]
    for t in ordered:
        lines.append(_cap(pos["terms"][t]["standing"]) + ".")
    # The opening is CONTEXT, not a tactic. Joining every fact into one "Tactics:" line told the model
    # that "This is the Ruy Lopez." is a tactical observation about the position, alongside a hanging
    # queen — so it gets its own line, and the Tactics line keeps its original meaning.
    opening = [f for f in facts if f.kind == "opening"]
    tactics = [f for f in facts if f.kind != "opening"]
    if opening:
        lines.append("Opening: " + " ".join(f.text for f in opening))
    if tactics:
        lines.append("Tactics: " + "; ".join(f.text for f in tactics) + ".")
    return lines


def build_analysis(board, engine, *, nodes: int | None = None,
                   movetime_ms: int | None = None) -> list[str]:
    """The grounded briefing for `board` as a list of natural-language statements.

    Deterministic given the engine limit (pass `nodes=` for tests, `movetime_ms=`
    for production — the same split as `Engine.analyse` / `build_fact_sheet`).
    Order: eval, material, the four strategic terms (leads first), then the
    tactical facts joined into one line. Empty tail sections are omitted."""
    analysis = engine.analyse(board.fen, multipv=1, nodes=nodes, movetime_ms=movetime_ms)
    pos = analyze_positional(board)
    facts = build_fact_sheet(board, engine, nodes=nodes, movetime_ms=movetime_ms)
    return assemble_analysis(board, analysis.best.score, pos, facts)
