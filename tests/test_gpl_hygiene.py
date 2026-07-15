"""GPL-hygiene gate (enforces the dual-license invariant).

lucena-engine is dual-licensed: AGPL-3.0 to the public, proprietary use by the copyright holder. That
only holds while NOTHING copyleft is *linked* into the library — GPL engines (Stockfish, Maia) are used
only as subprocesses over UCI, and python-chess (GPL) may appear in tests/tooling but NEVER in the
shipped library. This test fails the build if `import chess` sneaks into `python/lucena_engine/`.
"""

import pathlib
import re

_LIB = pathlib.Path(__file__).resolve().parents[1] / "python" / "lucena_engine"
_IMPORT_CHESS = re.compile(r"^\s*(?:import\s+chess\b|from\s+chess\b)", re.MULTILINE)


def test_no_python_chess_in_library():
    offenders = []
    for py in _LIB.rglob("*.py"):
        if _IMPORT_CHESS.search(py.read_text(encoding="utf-8")):
            offenders.append(str(py.relative_to(_LIB.parent.parent)))
    assert not offenders, (
        "python-chess (GPL) must NOT be imported in the shipped library — it would bind the "
        f"dual-licensed engine to GPL. Offending files: {offenders}"
    )
