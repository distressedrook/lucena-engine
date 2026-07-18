"""Forcing-line TREE builder — the "only move" tree.

A *forcing line* is any position that takes the puzzle form: **exactly one** legal
move preserves the result — win OR draw — and every other move is *catastrophically
worse*, meaning it drops out of the result band. The band's floor is set by the best
move: a WINNING best must be kept winning (>= WIN_BAR, exactly today's behavior), a
HOLDING best kept at least drawn (>= HOLDABLE_FLOOR); a LOST best is no puzzle. Exactly
one move at/above that floor is the only-move. This subsumes both winning combinations
("only move to keep the win") and defensive holds ("only move to hold the draw"), and a
mate is just a maximally-winning only-move. The tree drills those only-move moments and
branches on **all** the opponent's real defenses (a node can have many children — play
through them all), continuing until the result is banked (many moves hold it) or mate.

Two-tier depth for latency vs trust: a cheap shallow pass over EVERY legal move
(`discover`) decides whether exactly one wins. When that verdict is *unambiguous*
(the only winner clears the bar by a margin and the runner-up is well under), we
trust it; only a *borderline* node pays the deeper `verify` pass. Serial by
design — the tree is a mostly-sequential main line, so thread/engine-pool
parallelism measured *slower* here (GIL-serialised engine reads + near-zero
parallel width); the real latency win is not re-searching when the shallow pass
already decided (see learnings.md #13). Deterministic: `nodes`-limited searches
and a fixed traversal order.
"""

from __future__ import annotations

from .board import Board
from .evalmodel import win_pct_from_score

# "Catastrophically worse" means DROPPING OUT OF THE RESULT BAND, not a fixed win% gap — win% is
# compressed near equality (a win->draw swing at +2 pawns is ~17 win%, but a draw->loss swing near 0
# is only ~8), so a fixed gap can't mean "the result changed" across the eval range. The band's floor
# is set by the best move: a WINNING best must be kept winning; a HOLDING best kept at least drawn.
WIN_BAR = 67.0          # win% ~ +2 pawns: a winning best move must keep the win (>= WIN_BAR). Exactly
                        # today's threshold, so winning combinations behave identically.
HOLDABLE_FLOOR = 40.0   # win% ~ -0.8 pawns: below this even the best move is losing (no puzzle). A best
                        # move in [HOLDABLE_FLOOR, WIN_BAR) is a HOLD — the floor to keep is the draw.
HOLD_SEPARATION = 3.0   # a HOLD only-move must clear the floor by this — a best that BARELY holds (win%
                        # right at the floor) is eval-noise, not a robust hold, and would flip
                        # drillable on jitter. WIN-band only-moves keep their exact threshold (no
                        # extra margin), so winning drills are unchanged.
VERIFY_MARGIN = 8.0     # trust the shallow pass without a deep re-search only when best clears the
                        # band floor by this AND the runner-up is under the floor by this
VERIFY_MULTIPV = 3      # the deep pass searches NARROW (top-N), not full width: a wide multipv
                        # over 40+ moves starves each line of depth so a mate-in-5 reads ~equal.
                        # 3 lines is enough to see the winner AND whether a 2nd move also wins.
DEFENSE_BAND = 12.0     # opponent replies within this win% of their best are real tries
MAX_DEFENSES = 5        # cap the opponent branching (recorded via `truncated` when it bites)
MAX_DEPTH = 6           # student moves deep (safety cap; the real terminator is "everything wins")
MATE_HORIZON = 6        # drill forced mates up to mate-in-N; longer is technique, not a tactic
MATE_MAX_REPLIES = 16   # safety cap on mate-mode branching (the real terminator is "multiple winning moves -> stop")


def _band_floor(best_win: float) -> float | None:
    """The result-band floor a move must clear to be 'as good as the best'. A WINNING best
    (>= WIN_BAR) sets the floor at WIN_BAR (keep the win); a HOLDING best ([HOLDABLE_FLOOR, WIN_BAR))
    sets it at HOLDABLE_FLOOR (keep at least the draw); a LOST best (< HOLDABLE_FLOOR) => None (no
    puzzle — the position doesn't hold even with best play)."""
    if best_win >= WIN_BAR:
        return WIN_BAR
    if best_win >= HOLDABLE_FLOOR:
        return HOLDABLE_FLOOR
    return None


def is_only_move(win_pcts) -> bool:
    """Whether these best-first move win%s form a drillable only-move — the puzzle form: exactly one
    move preserves the result and every other is catastrophically worse (drops out of the result
    band). The band floor comes from the best move (`_band_floor`); a HOLD-band only-move must ALSO
    clear the floor by `HOLD_SEPARATION` (a best hovering at the floor is eval-noise, not a robust
    hold — the WIN band is exempt, so winning drills are unchanged). `win_pcts` are the moves' win%
    (side-to-move POV), best-first. False means lost (best doesn't hold) or converted (2+ moves hold).
    This is the pure rule; `build_line_tree` and its callers observe it only through the tree."""
    if not win_pcts:
        return False
    best = win_pcts[0]
    floor = _band_floor(best)
    if floor is None:
        return False
    if floor == HOLDABLE_FLOOR and best < HOLDABLE_FLOOR + HOLD_SEPARATION:
        return False
    return sum(w >= floor for w in win_pcts) == 1


def _only_move(lines):
    """The single drillable only-move at this node as `(uci, score)`, or `None` — a thin wrapper
    that applies `is_only_move` to the lines' win%s. `lines` are multipv, best-first."""
    playable = [ln for ln in lines if ln.pv]
    if not playable:
        return None
    if not is_only_move([win_pct_from_score(ln.score) for ln in playable]):
        return None
    return playable[0].pv[0], playable[0].score


def _leaf_reason(lines) -> str:
    """Why a non-only-move node is a leaf: `lost` (best play doesn't even hold) vs `converted`
    (the result is secured — many moves keep it, nothing unique left to find)."""
    best_win = win_pct_from_score(lines[0].score) if lines and lines[0].pv else 0.0
    return "lost" if best_win < HOLDABLE_FLOOR else "converted"


def _piece_count(fen: str) -> int:
    """Pieces on the board — a move is a capture if fewer remain after it."""
    return sum(ch.isalpha() for ch in fen.split(" ", 1)[0])


def _fastest_mate_moves(lines):
    """Every move that forces mate in the FEWEST moves for the mover, and that
    mate distance. "Even multiple checkmating lines are all caught" — so we keep
    all moves achieving the shortest mate, not just one. `([], None)` if no
    forced mate. Line scores are from the mover's POV, `mate > 0` = mover mates."""
    mates = [(ln, ln.score.mate) for ln in lines
             if ln.pv and ln.score.is_mate and ln.score.mate > 0]
    if not mates:
        return [], None
    shortest = min(m for _, m in mates)
    return [ln.pv[0] for ln, m in mates if m == shortest], shortest


def build_line_tree(fen, engine, *, discover: dict, verify: dict,
                    max_depth: int = MAX_DEPTH) -> dict:
    """Build the forcing-line tree for the side to move in `fen`.

    `discover` / `verify` are engine limits (`{"nodes": N}` in tests,
    `{"movetime_ms": M}` in production) for the shallow and deep passes; `discover`
    should be the cheaper one. Deterministic for fixed limits."""
    board = Board(fen)
    root = _student_node(board, engine, 0, max_depth, discover, verify)
    return {"schema": 1, "fen": fen, "side_to_solve": board.side_to_move, "root": root}


def _analyse(engine, fen, multipv, limit):
    engine.new_game()
    return engine.analyse(fen, multipv=multipv, **limit)


def _done(board, reason, score=None) -> dict:
    node = {"kind": "done", "fen": board.fen, "reason": reason}
    if score is not None:
        node["win_pct"] = round(win_pct_from_score(score), 1)
    return node


def _student_node(board, engine, depth, max_depth, discover, verify) -> dict:
    """The student is to move: a FIND node iff exactly one move preserves the result (win OR draw)
    and every other move is catastrophically worse (see `_only_move`), else a leaf."""
    moves = board.legal_moves()
    if not moves:
        return _done(board, "mate" if board.in_check else "stalemate")

    # tier 1 — shallow, full width over every legal move.
    a = _analyse(engine, board.fen, len(moves), discover)
    mate_moves, mate_in = _fastest_mate_moves(a.lines)
    only = _only_move(a.lines)
    verified = False

    # A shallow pass MISSES deep tactics: a forced mate or a deep only-move several plies out reads
    # as ~equal at discover depth (a genuine mate-in-5 puzzle scored 0.0 at 45k nodes). So when the
    # shallow pass is inconclusive — no mate within horizon AND no clean only-move — RE-SEARCH at
    # verify depth before concluding. Without this, a real forcing line is wrongly dismissed.
    if not (mate_moves and mate_in <= MATE_HORIZON) and only is None:
        a = _analyse(engine, board.fen, min(VERIFY_MULTIPV, len(moves)), verify)
        mate_moves, mate_in = _fastest_mate_moves(a.lines)
        only = _only_move(a.lines)
        verified = True

    # A forced mate within the horizon is the point of the node — BUT it is still bounded by the depth
    # cap. Without this, a mate net past the tactic (win the queen, THEN mate) enumerated every mating
    # line against every defense, ignoring `max_depth`, and a single puzzle ballooned to ~66 solve
    # nodes / 23+ forced moves — unfinishable, so it never concluded and the reveal never fired. Past
    # the cap, a forced mate is a decisive result: end the drill on it (`done`), don't drill it out.
    if depth >= max_depth:
        return _done(board, "mate" if (mate_moves and mate_in <= MATE_HORIZON) else "depth")

    # "If there are multiple winning moves, STOP." No unique move to find means the tactic is over —
    # the position is converted (2+ comparable moves, so anything wins) or lost. Conclude here, BEFORE
    # the mate branch. Otherwise a position won on material with an INCIDENTAL forced mate (win the
    # queen, THEN a mate exists) drilled the entire mate net — one puzzle ballooned to ~66 solve nodes
    # / 23+ forced moves, unfinishable, so it never concluded and the poisoned reveal never fired. A
    # genuine mate/only-move puzzle keeps a UNIQUE move (`only` is not None), so the drill continues.
    if only is None:
        # lost (best doesn't hold) or converted (2+ comparable moves — result banked).
        return _done(board, _leaf_reason(a.lines), a.best.score)

    if mate_moves and mate_in <= MATE_HORIZON:
        options = [{
            "uci": u, "san": board.san(u),
            "then": _opponent_node(board.apply(u), engine, depth, max_depth,
                                   discover, verify, all_moves=True),
        } for u in mate_moves]
        return {"kind": "mate", "fen": board.fen, "mate_in": mate_in, "options": options}

    move, score = only
    best_win = win_pct_from_score(a.best.score)
    second_win = win_pct_from_score(a.lines[1].score) if len(a.lines) > 1 else 0.0
    floor = _band_floor(best_win)            # not None here (only is not None)
    unambiguous = (best_win >= floor + VERIFY_MARGIN and second_win <= floor - VERIFY_MARGIN)

    if not (unambiguous or verified):       # borderline near the floor — confirm the only-move at depth
        v = _analyse(engine, board.fen, min(VERIFY_MULTIPV, len(moves)), verify)
        only = _only_move(v.lines)
        if only is None:
            return _done(board, _leaf_reason(v.lines), v.best.score)
        move, score = only

    after = board.apply(move)
    return {
        "kind": "solve",
        "fen": board.fen,
        "expect_uci": move,
        "expect_san": board.san(move),
        "win_pct": round(win_pct_from_score(score), 1),
        "after": _opponent_node(after, engine, depth, max_depth, discover, verify),
    }


def _refutation_key(then: dict) -> str:
    """A signature for how a defense is refuted — the student's refuting move, or the
    line's finish. Two defenses with the same key are redundant to drill."""
    return (then.get("expect_uci")
            or (then["options"][0]["uci"] if then.get("options") else None)
            or ("done:" + str(then.get("reason"))))


def _opponent_node(board, engine, depth, max_depth, discover, verify,
                   *, all_moves: bool = False) -> dict:
    """The opponent is to move: gather candidate defenses, play through each, and keep
    only the ones with a **distinct refutation**. Two tries that force the *same*
    reply/finish are redundant to drill — the player would just find the same move
    twice — so only the first (best-first) of each refutation survives. Candidates:
    in material mode, the best reply + tempting human tries (a capture or a check —
    even a losing one like ...Bxh4 — plus quiet moves inside the win band); in mate
    mode (`all_moves`), every legal reply. A terminal position finished the line."""
    legal = board.legal_moves()
    if not legal:
        return _done(board, "mate" if board.in_check else "stalemate")

    if all_moves:   # mate mode: every reply loses to the mate — all are candidates
        candidates = list(legal)
        cap = MATE_MAX_REPLIES
    else:           # material mode: best reply + tempting tries (capture/check/in-band)
        # Rank at VERIFY (reliable) depth over enough moves to surface every
        # capture/check — at the shallow discover depth a forcing check dominates a
        # quiet capture, so the band alone drops real tries (e.g. ...Bxh4).
        n = min(len(legal), MAX_DEFENSES + 12)
        a = _analyse(engine, board.fen, n, verify)
        opp_best = win_pct_from_score(a.best.score)
        best_uci = a.best.pv[0] if a.best.pv else None
        before = _piece_count(board.fen)
        candidates = []
        for ln in a.lines:
            if not ln.pv:
                continue
            u = ln.pv[0]
            nxt = board.apply(u)
            forcing = nxt.in_check or _piece_count(nxt.fen) < before      # a check or a capture
            within_band = (opp_best - win_pct_from_score(ln.score)) <= DEFENSE_BAND
            if u == best_uci or forcing or within_band:
                candidates.append(u)
        cap = MAX_DEFENSES

    # Play each candidate and keep only DISTINCT refutations, so the drill never asks
    # the player to find the same reply twice. Cap the branching (never silently).
    seen, defenses, truncated = set(), [], False
    for u in candidates:
        then = _student_node(board.apply(u), engine, depth + 1, max_depth, discover, verify)
        ref = _refutation_key(then)
        if ref in seen:
            continue                       # same refutation as a kept defense — redundant
        if len(defenses) >= cap:
            truncated = True
            break
        seen.add(ref)
        defenses.append({"uci": u, "san": board.san(u), "then": then})

    node = {"kind": "reply", "fen": board.fen, "defenses": defenses}
    if truncated:   # never a silent cap
        node["truncated"] = True
    return node


def count_leaves(node: dict) -> int:
    """How many distinct forcing lines the tree drills (for logging / progress)."""
    if node.get("kind") == "solve":
        return count_leaves(node["after"])
    if node.get("kind") == "mate":
        return sum(count_leaves(o["then"]) for o in node["options"]) or 1
    if node.get("kind") == "reply":
        return sum(count_leaves(d["then"]) for d in node["defenses"]) or 1
    return 1


def only_move_nodes(tree: dict) -> list[dict]:
    """The student's only-move positions — `{fen, san}` per solve/mate node.
    Deduped by fen (a position can recur across branches), best-first order
    preserved. A structural helper over the tree; the app walks the full tree
    (every defense) and coaches each node live."""
    out: list[dict] = []

    def visit(node: dict) -> None:
        kind = node.get("kind")
        if kind == "solve":
            out.append({"fen": node["fen"], "san": node.get("expect_san")})
            if node.get("after"):
                visit(node["after"])
        elif kind == "mate":
            opts = node.get("options") or []
            out.append({"fen": node["fen"], "san": opts[0]["san"] if opts else None})
            for o in opts:
                visit(o["then"])
        elif kind == "reply":
            for d in node.get("defenses", []):
                visit(d["then"])

    visit(tree["root"])
    seen, uniq = set(), []
    for nd in out:
        if nd["fen"] not in seen:
            seen.add(nd["fen"])
            uniq.append(nd)
    return uniq
