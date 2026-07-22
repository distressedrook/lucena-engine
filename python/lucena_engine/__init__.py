"""lucena-engine: the Stockfish/Maia wrapper package.

One package wrapping both engines behind clean session APIs, plus the eval
vocabulary their answers speak (Score / Analysis / Line, the lichess-verbatim
win% model, glyph classification) and the SF NNUE eval parser.

Slimmed 2026-07-23: the Rust board core, board-truth modules (positional,
detect, pgn, openings, reads, _fen) and the gRPC surface moved to the private
`lucena-core`; the tactical layer (detectors/facts/hints/line_tree/puzzle/
brilliant) moved to the private `lucena-tactics`; game-import assembly
(analysis, gamepass) moved to the private backend. What remains is exactly
the part that must exist as a public, license-clean boundary: GPL engines
(Stockfish, Maia) are spoken to ONLY as subprocesses over UCI, and nothing
copyleft is linked in (tests/test_gpl_hygiene.py enforces it).
"""

from .uci import Engine, EngineError, Line, Analysis
from .evalmodel import (
    win_pct,
    win_pct_from_score,
    classify,
    parse_score,
    Score,
    Glyph,
    MISTAKE,
)
from .nnue import PieceValue, parse_piece_values, piece_value_map

__all__ = [
    "Engine",
    "EngineError",
    "Line",
    "Analysis",
    "win_pct",
    "win_pct_from_score",
    "classify",
    "parse_score",
    "Score",
    "Glyph",
    "MISTAKE",
    "PieceValue",
    "parse_piece_values",
    "piece_value_map",
]
