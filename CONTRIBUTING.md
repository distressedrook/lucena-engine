# Contributing to lucena-engine

Thanks for your interest. Two things before a contribution can be merged.

Contributions to the engine are accepted under the repository's AGPL-3.0-or-later license. By opening
a pull request, you confirm that you have the right to submit the work and agree that the contribution
may be distributed under that license. If your employer has rights to your work, get their sign-off
first.

## Keep the copyleft boundary clean

- **Never `import chess` (python-chess, GPL) under `python/lucena_engine/`.** Use the Rust board core.
  A CI grep enforces this.
- GPL engines (Stockfish, Maia) are used **only as subprocesses over UCI** — never linked or imported.
- New runtime dependencies must document their license and provenance. Keep GPL components behind the
  existing subprocess/UCI boundary unless the affected license and distribution obligations are
  explicitly documented.
