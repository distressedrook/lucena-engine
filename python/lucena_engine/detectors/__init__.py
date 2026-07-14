"""Motif detectors — SEE/board-core (+ engine) salience hints (LLD §2.3).

Detectors are hints ranked by salience, never verdicts; the verdict is always
the eval delta / SEE. M3 ships the tactical trio:

- `detect_hanging`          — hanging-piece, both directions (SEE, board-only)
- `detect_defender_removed` — remove-the-guard / only-defender (board-only)
- `detect_null_move_threat` — "what if you pass?" (needs the engine)
"""

from .hanging import detect_hanging
from .defender import detect_defender_removed
from .fork import detect_fork
from .pin import detect_pin
from .combination import detect_combination
from .null_move import detect_null_move_threat

__all__ = [
    "detect_hanging",
    "detect_defender_removed",
    "detect_fork",
    "detect_pin",
    "detect_combination",
    "detect_null_move_threat",
]
