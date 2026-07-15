# Third-party notices — lucena-engine

lucena-engine (AGPL-3.0-or-later) uses the following third-party components. Their licenses are
compatible with AGPL-3.0; permissive components are statically linked, and the GPL chess engines are
used **only as separate subprocesses over a text protocol (UCI)** — an arm's-length boundary, not
linking.

## Linked libraries (permissive — no copyleft)

| Component | Role | License |
|---|---|---|
| [cozy-chess](https://crates.io/crates/cozy-chess) 0.3 | bitboards + legal move generation (Rust) | MIT OR Apache-2.0 |
| [pyo3](https://crates.io/crates/pyo3) 0.23 | Rust ↔ Python bindings | Apache-2.0 OR MIT |
| [grpcio / grpcio-tools](https://pypi.org/project/grpcio/) | gRPC server + codegen (optional `grpc` extra) | Apache-2.0 |
| protobuf runtime (`google.protobuf`) | wire types for gRPC | BSD-3-Clause |

SAN encoding and the SEE (static exchange evaluation) in `rust/src/` are original to this project.

## External programs invoked as subprocesses (arm's-length, over UCI)

These are **not linked**; lucena-engine launches them as separate processes and exchanges UCI text.
Distributing their binaries carries each program's own obligations (ship the license text and provide
or offer the corresponding source).

| Program | Role | License | Source |
|---|---|---|---|
| Stockfish 18 | position evaluation / search | GPL-3.0-or-later | https://github.com/official-stockfish/Stockfish |
| Maia (maia3) | human-move prediction (own `.venv-maia`) | see upstream | https://github.com/CSSLab/maia3 |

> Maia's runtime environment pulls in `python-chess` (GPL-3.0) and `torch`. These live **only** in the
> separate `.venv-maia` subprocess and are never imported by lucena-engine.

## Development-only (not shipped, not imported by the library)

| Component | License | Note |
|---|---|---|
| python-chess | GPL-3.0 | tests/tooling only — never imported under `python/lucena_engine/` |
| pytest, maturin | MIT / Apache-2.0 | build & test |
