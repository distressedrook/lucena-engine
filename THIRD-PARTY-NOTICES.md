# Third-party notices — lucena-engine

lucena-engine (AGPL-3.0-or-later) is, since v0.2.0, a pure-Python wrapper
package: engine sessions over UCI plus the eval vocabulary their answers
speak. It links **nothing** third-party — no compiled extensions, no runtime
dependencies. The GPL chess engines are used **only as separate subprocesses
over a text protocol (UCI)** — an arm's-length boundary, not linking.

(Pre-0.2.0 releases bundled a Rust board core — cozy-chess, pyo3 — plus a
gRPC surface and Lichess opening data; those components moved to private
downstream packages in the 2026-07-23 slim and their notices moved with
them. The notices below describe what THIS package ships and invokes.)

## External programs invoked as subprocesses (arm's-length, over UCI)

These are **not linked**; lucena-engine launches them as separate processes
and exchanges UCI text. Distributing their binaries carries each program's
own obligations (ship the license text and provide or offer the
corresponding source).

| Program | Role | License | Source |
|---|---|---|---|
| Stockfish 18 | position evaluation / search | GPL-3.0-or-later | https://github.com/official-stockfish/Stockfish |
| Maia (maia3) | human-move prediction (own `.venv-maia`) | see upstream | https://github.com/CSSLab/maia3 |

> Maia's runtime environment pulls in `python-chess` (GPL-3.0) and `torch`.
> These live **only** in the separate `.venv-maia` subprocess and are never
> imported by lucena-engine.

## Development-only (not shipped, not imported by the library)

| Component | License | Note |
|---|---|---|
| python-chess | GPL-3.0 | tests/tooling only — never imported under `python/lucena_engine/` (CI-gated: `tests/test_gpl_hygiene.py`) |
| pytest | MIT | tests |
