#!/usr/bin/env python3
"""Policy-emitting `maia3-uci` wrapper — the production `LUCENA_MAIA` target.

Why this exists: `maia3-uci` computes each move's *real* policy probability
internally (`torch.softmax` over the legal-masked move logits, `maia3/uci.py`
`score_moves`), stores it as `item["policy"]`, then **drops it** — its `cmd_go`
only prints `score cp` / `wdl` / `pv` on the wire. `find_poisoned_lines` needs
that probability (it multiplies policies along a line; see
`docs/contracts/M-poisoned-line-detector.md` — "policy is a REAL probability,
not derived from rank"). Deriving it from rank or the value-head `cp`/`wdl` is
explicitly wrong.

This wrapper monkeypatches `Maia3UCIEngine.cmd_go` to append `policy <p>` to
every `info` line (before `pv`, so `pv <uci>` stays last), then hands off to
maia3's own `main`. The reported policy uses raw logits, so it is independent of
the `Temperature`/`TopP` sampling options — the model's true policy.

Runtime: runs in `.venv-maia` (has `torch` + `maia3`). Point `LUCENA_MAIA` at it:

    export LUCENA_MAIA="$PWD/.venv-maia/bin/python $PWD/tools/maia_policy_uci.py"

Parsed back by `lucena.engine.maia._parse_multipv`.
"""
from __future__ import annotations

import sys

from maia3 import uci as _uci


def _cmd_go_with_policy(self, line):
    """`Maia3UCIEngine.cmd_go`, verbatim, but emitting `policy <p>` per move."""
    self.ensure_model_loaded()
    move, top_moves = self.score_moves()
    for rank, item in enumerate(top_moves, start=1):
        win, draw, loss = item["wdl"]
        cp = _uci.cp_from_wdl(item["wdl"])
        policy = float(item.get("policy", 0.0))
        print(
            f"info depth 1 multipv {rank} score cp {cp} wdl {win} {draw} {loss} "
            f"policy {policy:.6f} pv {item['move'].uci()}",
            flush=True,
        )
    if "infinite" in line.split():
        self.pending_bestmove = move
        self.pending_search = True
        return
    self.print_bestmove(move)


_uci.Maia3UCIEngine.cmd_go = _cmd_go_with_policy


def main() -> None:
    # Mirror maia3.presets.main_5m: force the CPU-friendly 5M model at temperature
    # 0 (deterministic ordering). Caller args come last so they win (argparse
    # last-value-wins); in production LUCENA_MAIA passes none.
    _uci.main(["--temperature", "0", "--model", "maia3-5m", *sys.argv[1:]])


if __name__ == "__main__":
    main()
