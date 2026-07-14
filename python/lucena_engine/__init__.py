"""Lucena grounding engine: board core + Stockfish session, eval semantics, facts."""

from .board import Board, STARTPOS, PieceOn
from .uci import Engine, EngineError, Line, Analysis
from .evalmodel import (
    win_pct,
    win_pct_from_score,
    classify,
    parse_score,
    Score,
    Glyph,
)
from .nnue import PieceValue, parse_piece_values, piece_value_map
from .facts import Fact, build_fact_sheet
from .hints import Hint, derive_hints
from .brilliant import is_brilliant, is_material_sacrifice, classify_brilliant
from .pgn import parse_pgn, Game, Ply, PgnError
from .gamepass import run_pass
from .detectors import (
    detect_hanging,
    detect_defender_removed,
    detect_fork,
    detect_pin,
    detect_combination,
    detect_null_move_threat,
)

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
    "PieceValue",
    "parse_piece_values",
    "piece_value_map",
    "Fact",
    "build_fact_sheet",
    "Hint",
    "derive_hints",
    "is_brilliant",
    "is_material_sacrifice",
    "classify_brilliant",
    "detect_hanging",
    "detect_defender_removed",
    "detect_fork",
    "detect_pin",
    "detect_combination",
    "detect_null_move_threat",
    "parse_pgn",
    "Game",
    "Ply",
    "PgnError",
    "run_pass",
]
