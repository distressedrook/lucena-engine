"""Poisoned-line detection — find the human TRAPS in a position
(contract: docs/contracts/M-poisoned-line-detector.md).

A line is **poisoned** when a move a player at the student's rating would naturally play (Maia)
walks into a tactic (Stockfish), AND the real joint probability of a human walking that exact
path clears a noise floor — not merely "somewhere in Maia's top-k." Slice 1 (formerly `nuance.py`)
used rank alone as the human-plausibility signal; this slice replaces that with Maia's actual
policy probability, multiplied along the whole path (`poisoned`), because rank conflates a move
played 60% of the time with one played 0.8% of the time.

Two public functions on one primitive:
  - `find_poisoned_lines`      — the mover's own temptations (the defensive read; the live nudge).
  - `find_practical_tries` — my moves that shove the OPPONENT into the most-riddled position (the
                             offensive "Tal move"; built on `find_poisoned_lines`).

Determinism is load-bearing: a detected trap is a reproducible artifact, so search is **node-limited
on a single-threaded engine**, never `movetime` (wall-clock search yields phantom traps — proven on
the Tal set). The caller passes `Engine(threads=1)`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .reads import material as _material
from ._fen import norm_fen as _norm
from .board import Board
from .evalmodel import win_pct_from_score


# _norm is imported from ._fen (norm_fen) — the one home for position identity.


# Defaults (validated): the opponent's winning idea must be decisive (win_bar) AND essentially the
# ONLY winning move (spike) — "one winning idea", not "generally better". k seeds, plies deep.
_NODES = 200_000
_K = 6
_PLIES = 10
_WIN_BAR = 68.0
_SPIKE = 20.0
_NOISE_BAR = 1.0
_COMMON_BAR = 5.0
_MATERIAL_FLOOR = -1        # skip if the mover is already down > this many pawns at the root (static
                           # material, mover-relative). A poisoned line is only "special" from a
                           # materially sound position; if you START clearly material down, "play
                           # precisely or get punished" is just what being down means, not a trap.
_SHALLOW_KEEP_PROB = 40.0   # a 1-move (deep=False) temptation is kept only if it's THIS common — a
                           # rare one-mover is just a blunder with no nuance, but a one-move mistake
                           # 40%+ of players actually make is teachable by its prevalence alone.
                           # Multi-move (deep) lines are kept regardless of probability.


def _spike_at(engine, fen: str, nodes: int, win_bar: float, spike: float):
    """If the side to move at `fen` has ONE winning idea (a multi-PV spike), return (idea_uci,
    opp_win_pct, gap); else None. `idea` is the top move; the gap is best minus second-best win%."""
    engine.new_game()
    a = engine.analyse(fen, multipv=2, nodes=nodes)
    best = win_pct_from_score(a.best.score)
    second = win_pct_from_score(a.lines[1].score) if len(a.lines) > 1 else 0.0
    if best >= win_bar and (best - second) >= spike and a.best.pv:
        return a.best.pv[0], round(best, 1), round(best - second, 1)
    return None


def _best_reply(engine, fen: str, nodes: int) -> str:
    engine.new_game()
    return engine.analyse(fen, multipv=1, nodes=nodes).best.pv[0]


def _policy(move: dict) -> float:
    """The move's real Maia policy probability — fail loud, never default. A missing
    `"policy"` means the injected Maia isn't emitting real probabilities (needs the
    policy-emitting wrapper; see docs/contracts/M-poisoned-line-detector.md). Silently
    treating it as 0 would zero every `poisoned` product, so every poisoned line would
    vanish without a trace — the exact silent-death this guard exists to prevent."""
    p = move.get("policy")
    if p is None:
        raise ValueError(
            "maia move dict missing 'policy' — the injected Maia is not emitting real "
            "policy probabilities. Point LUCENA_MAIA at tools/maia_policy_uci.py (see "
            "docs/contracts/M-poisoned-line-detector.md). Refusing to default to 0."
        )
    return float(p)


def _tier(poisoned_pct: float, noise_bar: float, common_bar: float) -> str | None:
    if poisoned_pct < noise_bar:
        return None
    return "common" if poisoned_pct >= common_bar else "uncommon"


def _keep(deep: bool, poisoned_pct: float, noise_bar: float, common_bar: float,
          shallow_keep_prob: float) -> str | None:
    """The tier a temptation is kept at, or None to discard. Discards below the noise floor AND
    discards a shallow (one-move) temptation unless it's common enough (> shallow_keep_prob): a
    rare one-mover is just a blunder. Multi-move (deep) lines are kept at any non-noise tier."""
    tier = _tier(poisoned_pct, noise_bar, common_bar)
    if tier is None:
        return None
    if not deep and poisoned_pct <= shallow_keep_prob:
        return None
    return tier


def _rollout(root_fen: str, seed_uci: str, seed_policy: float, engine, maia, *, rating: int,
            nodes: int, plies: int, win_bar: float, spike: float) -> dict | None:
    """Walk the greedy human line from a seed: the mover plays Maia's top-1, the opponent plays
    Stockfish's best. Return the first tactic node (an opponent-to-move spike) as
    `{fatal_fen, fatal, idea, opp_win_pct, spike, deep, poisoned}` (poisoned is the unrounded
    fraction, seed_policy times every subsequent mover-turn's own policy along the way), or None
    if the line reaches no tactic (or ends)."""
    root = Board(root_fen)
    mover = root.side_to_move
    board = root.apply(seed_uci)
    fatal_fen, fatal_uci = root_fen, seed_uci     # the mover node most-recently played from
    poisoned = seed_policy
    mover_plies = 0                # how many mover moves since the seed (for the deep/immediate flag)
    for _ in range(plies):
        if not board.legal_moves():                      # the line ended (mate/stalemate) — no trap
            return None                                  # (guards a seed/reply that ends the game)
        if board.side_to_move != mover:                 # opponent to move — look for the tactic
            hit = _spike_at(engine, board.fen, nodes, win_bar, spike)
            if hit is not None:
                idea_uci, opp_win, gap = hit
                fatal_board = Board(fatal_fen)
                return {
                    "fatal_fen": fatal_fen, "fatal_uci": fatal_uci,
                    "fatal": fatal_board.san(fatal_uci),
                    "tactic_fen": board.fen,           # the POST-fatal node — dedup key, not fatal_fen
                    "idea_uci": idea_uci, "idea": board.san(idea_uci),
                    "opp_win_pct": opp_win, "spike": gap, "deep": mover_plies >= 1,
                    "poisoned": poisoned,
                }
            board = board.apply(_best_reply(engine, board.fen, nodes))  # no tactic — walk on
        else:                                            # the human (Maia) plays their natural move
            top = maia.top_human_moves(board.fen, rating, n=1)
            if not top:
                return None
            fatal_fen, fatal_uci = board.fen, top[0]["uci"]
            poisoned *= _policy(top[0])
            board = board.apply(fatal_uci)
            mover_plies += 1
    return None


def find_poisoned_lines(fen: str, engine, maia, *, rating: int, solution_uci: str | None = None,
                        nodes: int = _NODES, k: int = _K, plies: int = _PLIES,
                        win_bar: float = _WIN_BAR, spike: float = _SPIKE,
                        noise_bar: float = _NOISE_BAR, common_bar: float = _COMMON_BAR,
                        material_floor: int = _MATERIAL_FLOOR,
                        shallow_keep_prob: float = _SHALLOW_KEEP_PROB,
                        stop_on_first: bool = False, cache: dict | None = None) -> dict:
    """The mover's poisoned lines at `fen` — moves a player at `rating` would naturally play that
    walk into a tactic, weighted by the real joint probability a human actually walks that exact
    path. See the contract for the full shape. `engine` must be single-threaded (determinism).

    Two specialness gates (a poisoned line is only interesting from a fair position, and must have a
    real lesson): `material_floor` — skip the whole position if the mover is already down more than
    this many pawns at the root (static material, mover-relative; default -1 allows "down a pawn").
    `shallow_keep_prob` — a one-move (`deep=False`) temptation is kept only if it's more than this %
    likely; multi-move lines are kept at any non-noise tier. The mining tool calls this same function
    so the two never diverge.

    `stop_on_first`: return as soon as the first *kept* temptation is found (from the most
    human-likely seed onward, i.e. in the order `maia.top_human_moves` returns them) — enough for the
    live flag + a representative trap, at ~half the seeds' cost. `cache`: an optional caller-owned
    dict to memoize by `(norm_fen, rating, params…)` — production passes a persistent dict so a
    repeated evaluation is instant; tests pass None so fakes never cross-pollute."""
    empty = {"fen": fen, "has_poisoned_line": False, "temptations": []}
    if maia is None or engine is None:
        return empty
    key = None
    if cache is not None:
        key = (_norm(fen), rating, nodes, k, plies, win_bar, spike, noise_bar, common_bar,
              material_floor, shallow_keep_prob, stop_on_first, solution_uci)
        if key in cache:
            return cache[key]
    board = Board(fen)                                   # raises ValueError on a bad fen (per contract)
    if not board.legal_moves():
        return empty

    # Balanced-root gate: a poisoned line is only "special" from a materially sound position. If the
    # mover is already clearly down material, "play precisely or be punished" is just what being down
    # means, not a hidden trap. Static material, mover-relative -- same R.material() the miner uses.
    mat = _material(board)
    mover_net = mat["net"] if board.side_to_move == "white" else -mat["net"]
    if mover_net < material_floor:
        return empty

    # (norm(fatal_fen), idea_uci) -> merged temptation. Position identity, never SAN-string
    # equality — two unrelated positions can coincidentally render the same SAN.
    by_tactic: dict[tuple[str, str], dict] = {}
    for m in maia.top_human_moves(fen, rating, n=k):
        uci = m.get("uci")
        if not uci or uci == solution_uci:
            continue
        try:
            seed_san = board.san(uci)
        except Exception:
            continue                                     # a move Maia named that isn't legal here
        info = _rollout(fen, uci, _policy(m), engine, maia, rating=rating, nodes=nodes,
                        plies=plies, win_bar=win_bar, spike=spike)
        if info is None:
            continue
        # Dedup key is the TACTIC NODE (post-fatal position) + the idea move — that's the position
        # that's actually shared across merging seeds. The pre-fatal fatal_fen is NOT shared (each
        # seed generally reaches the tactic node via a different path/position beforehand).
        tkey = (_norm(info["tactic_fen"]), info["idea_uci"])
        if tkey not in by_tactic:
            # first-discovered seed for this tactic node "wins" opp_win_pct/spike/deep/avoids/drop
            # (computed once, below, from this first path's own fatal_fen) — later-merging seeds
            # only ever add to `seeds` and `poisoned`, per the contract's dedup precedent.
            engine.new_game()
            best = engine.analyse(info["fatal_fen"], multipv=1, nodes=nodes).best
            engine_best_uci = best.pv[0]
            engine_win_pct = win_pct_from_score(best.score)
            mover_win_after = 100.0 - info["opp_win_pct"]
            drop = engine_win_pct - mover_win_after
            avoids = engine_best_uci != info["fatal_uci"]
            by_tactic[tkey] = {
                "idea": info["idea"], "fatal": info["fatal"], "seeds": [],
                "opp_win_pct": info["opp_win_pct"], "spike": info["spike"], "deep": info["deep"],
                "avoids": avoids, "drop": round(drop, 1),
                "_poisoned_frac": 0.0,
            }
        entry = by_tactic[tkey]
        if seed_san not in entry["seeds"]:
            entry["seeds"].append(seed_san)
        entry["_poisoned_frac"] += info["poisoned"]
        if stop_on_first:                                # live path: one KEPT trap is enough. Stop on
            pp = round(entry["_poisoned_frac"] * 100, 2)  # a temptation that survives the gates, not
            if _keep(entry["deep"], pp, noise_bar, common_bar, shallow_keep_prob):  # merely one found
                break                                    # -- else a rare shallow hit would mask a
                                                         # real deep trap on a later seed.

    temptations = []
    for entry in by_tactic.values():
        poisoned_pct = round(entry.pop("_poisoned_frac") * 100, 2)
        tier = _keep(entry["deep"], poisoned_pct, noise_bar, common_bar, shallow_keep_prob)
        if tier is None:                                 # below noise floor, or a rare one-mover
            continue
        entry["poisoned"] = poisoned_pct
        entry["tier"] = tier
        temptations.append(entry)

    # deep traps first (the instructive kind), then the highest joint probability
    temptations.sort(key=lambda t: (not t["deep"], -t["poisoned"]))
    result = {"fen": fen, "has_poisoned_line": bool(temptations), "temptations": temptations}
    if key is not None:
        cache[key] = result
    return result


def evaluate_with_poisoned_lines(eval_fn, fen: str, engine, maia, *, rating: int,
                                 **detect_kwargs) -> dict:
    """Run a position eval and poisoned-line detection CONCURRENTLY, merged. `eval_fn` (the
    Stockfish position eval, on the caller's main engine) runs on the calling thread;
    `find_poisoned_lines` (Maia + a *separate* engine) runs on a worker thread. Both are
    subprocess-I/O-bound (GIL released), so they truly overlap → latency ≈ max(eval, detect), not
    the sum. Returns `{"eval": <eval_fn()>, "poisoned_line": <find_poisoned_lines()>}`."""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(find_poisoned_lines, fen, engine, maia, rating=rating, **detect_kwargs)
        ev = eval_fn()
        return {"eval": ev, "poisoned_line": fut.result()}


def find_practical_tries(fen: str, engine, maia, *, rating: int,
                         candidates: list[str] | None = None, floor_cp: int = -250,
                         nodes: int = _NODES, k: int = _K, plies: int = _PLIES,
                         win_bar: float = _WIN_BAR, spike: float = _SPIKE,
                         noise_bar: float = _NOISE_BAR, common_bar: float = _COMMON_BAR) -> dict:
    """The offensive read: which of MY moves shoves the OPPONENT into the most-riddled position.
    Play each candidate, run `find_poisoned_lines` for the opponent, rank by how many traps it
    creates. A move is kept only if it stays at/above `floor_cp` (a sacrifice into the band is
    fine, a blunder isn't) AND actually leaves the opponent a temptation. See the contract."""
    if maia is None or engine is None:
        return {"fen": fen, "tries": []}
    board = Board(fen)
    if candidates is None:
        candidates = [m["uci"] for m in maia.top_human_moves(fen, rating, n=k) if m.get("uci")]

    tries = []
    for uci in candidates:
        try:
            move_san = board.san(uci)
            after = board.apply(uci)
        except Exception:
            continue
        if not after.legal_moves():                      # my move ended the game — not a "try"
            continue
        # my eval after the move, from MY point of view (the opponent is to move, so negate)
        engine.new_game()
        opp_score = engine.analyse(after.fen, multipv=1, nodes=nodes).best.score
        my_cp = opp_score.negated().to_ceiled_cp()
        if my_cp < floor_cp:                             # I've thrown it away — not a practical try
            continue
        opp = find_poisoned_lines(after.fen, engine, maia, rating=rating, nodes=nodes, k=k,
                                  plies=plies, win_bar=win_bar, spike=spike,
                                  noise_bar=noise_bar, common_bar=common_bar)
        if not opp["temptations"]:                       # poses no problem — not a try
            continue
        top_spike = max(t["spike"] for t in opp["temptations"])
        tries.append({"move_san": move_san, "move_uci": uci, "my_cp": my_cp,
                      "trap_count": len(opp["temptations"]), "top_spike": top_spike,
                      "opponent": opp})

    tries.sort(key=lambda t: (-t["trap_count"], -t["top_spike"]))
    return {"fen": fen, "tries": tries}
