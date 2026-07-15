# Contributing to lucena-engine

Thanks for your interest. Two things before a contribution can be merged.

## Contributor License Agreement (required)

lucena-engine is **dual-licensed**: released to the public under AGPL-3.0-or-later, while its sole
copyright holder also uses it in proprietary software they own. For that arrangement to stay valid,
the project must retain unified copyright ownership.

By submitting a contribution (a pull request, patch, or any code/text), you agree to the following,
and you certify you have the right to do so:

1. You grant the copyright holder (Avismara) a perpetual, worldwide, royalty-free license to use,
   modify, sublicense, and **relicense** your contribution under any terms, including proprietary
   terms — in addition to AGPL-3.0.
2. Your contribution is your original work (or you have the right to submit it), and to your knowledge
   it does not knowingly infringe anyone's rights.

If your employer has rights to your work, get their sign-off first.

> Without this, a contribution would be AGPL-only and would bind the whole project to AGPL — breaking
> the dual-license model. Contributions submitted without agreeing to this cannot be merged.

## Keep the copyleft boundary clean

- **Never `import chess` (python-chess, GPL) under `python/lucena_engine/`.** Use the Rust board core.
  A CI grep enforces this.
- GPL engines (Stockfish, Maia) are used **only as subprocesses over UCI** — never linked or imported.
- New runtime dependencies must be permissive (MIT / Apache-2.0 / BSD). No new copyleft deps.
