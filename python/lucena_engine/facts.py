"""The fact sheet — deterministic salience extraction (LLD-analysis §2.3).

Every engine-tool response is built from a fact sheet, never a bare position.
This module owns the `Fact` record and `build_fact_sheet`, which runs the M3
detector trio, ranks the raw facts by salience, assigns stable `F#` ids, and
truncates to the top five.

Detectors are *hints ranked by salience, never verdicts* — the material
verdict is always SEE / the engine, never these facts. Salience weights are
hand-tuned in v1 (LLD OQ#8); the formulas live in each detector and are pinned
in `docs/contracts/M3-facts.md`.

`Fact` is defined here (not in the detectors package) so the detectors import
it without a cycle: `build_fact_sheet` imports the detector functions lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .evalmodel import _MISTAKE, win_pct_from_score


@dataclass(frozen=True)
class Fact:
    """One salient, deterministically-extracted observation about a position.

    `id` is assigned by `build_fact_sheet` (detectors leave it ""). `squares`
    drive board arrows (a cited fact = an arrow). `provenance` is for
    debuggability; `concept_id` routes to the mastery model.
    """

    kind: str                       # "hanging" | "threat" | "defender-removed" | "opening"
    squares: list[str]              # board squares this fact points at
    text: str                       # short human sentence (no id, no eval)
    provenance: str                 # e.g. "see", "nullmove", "static"
    salience: float                 # 0..1, higher = lead with this
    concept_id: str                 # mastery routing (memory/domain.json)
    id: str = ""                    # "F1".. assigned at assembly time

    def with_id(self, id: str) -> "Fact":
        return Fact(
            kind=self.kind,
            squares=self.squares,
            text=self.text,
            provenance=self.provenance,
            salience=self.salience,
            concept_id=self.concept_id,
            id=id,
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "squares": list(self.squares),
            "text": self.text,
            "provenance": self.provenance,
            "salience": round(self.salience, 3),
            "concept_id": self.concept_id,
        }


TOP_N = 5


def build_fact_sheet(
    board,
    engine=None,
    *,
    nodes: int | None = None,
    movetime_ms: int | None = None,
    top_n: int = TOP_N,
) -> list[Fact]:
    """Run the detectors over `board`, return the top `top_n` facts.

    Board-only detectors (hanging, defender-removed) always run. The null-move
    threat probe runs only when an `engine` is supplied; pass `nodes=` (tests,
    reproducible) or `movetime_ms=` (production) exactly as `Engine.analyse`
    expects — the same determinism split.

    When an engine is present, both hang directions are reconciled against the
    engine verdict (the §2.3 rule *a SEE fact that contradicts the engine loses*):
    `_reconcile_dangers` drops own-piece hangs the null-move probe does not
    confirm (a regained gambit pawn); `_reconcile_opportunities` drops enemy-piece
    hangs whose capture the engine scores as a blunder (a back-rank trap). Without
    an engine, neither runs — the board core cannot see past the single exchange,
    so the sheet is best-effort.

    Ranking: salience descending, with a deterministic tie-break so identical
    inputs always yield identical ids. Facts are assigned `F1..Fn` in rank
    order. Returns at most `top_n` facts (may be fewer, or empty).
    """
    # Lazy import breaks the facts <-> detectors cycle (detectors import Fact).
    from .detectors.hanging import detect_hanging
    from .detectors.defender import detect_defender_removed
    from .detectors.fork import detect_fork
    from .detectors.pin import detect_pin
    from .detectors.combination import detect_combination
    from .detectors.null_move import detect_null_move_threat

    raw: list[Fact] = []
    raw += detect_hanging(board)
    raw += detect_defender_removed(board)
    raw += detect_fork(board)
    raw += detect_pin(board)   # absolute pins — geometry, always true, no reconciliation
    if engine is not None:
        threats = detect_null_move_threat(
            board, engine, nodes=nodes, movetime_ms=movetime_ms
        )
        raw += threats
        raw += detect_combination(board, engine, nodes=nodes, movetime_ms=movetime_ms)
        raw = _reconcile_dangers(raw, threats)
        raw = _reconcile_opportunities(
            board, raw, engine, nodes=nodes, movetime_ms=movetime_ms
        )
        raw = _reconcile_forks(
            board, raw, engine, nodes=nodes, movetime_ms=movetime_ms
        )

    raw = _dedupe(raw)
    raw.sort(key=_rank_key)
    kept = raw[:top_n]

    # The opening name rides ALONGSIDE the ranking, never inside it. It is context ("this is the
    # Sicilian"), not a competing observation about the position, and the two do not trade off: given
    # a salience it would either evict a hanging queen from top_n, or be evicted by one and vanish
    # exactly when the position is quiet enough for the name to be the most useful thing to say.
    # `top_n` bounds the TACTICS, so it is applied before this.
    opening = _opening_fact(board)
    if opening is not None:
        kept = kept + [opening]
    return [f.with_id(f"F{i + 1}") for i, f in enumerate(kept)]


def _opening_fact(board) -> "Fact | None":
    """This position's opening, or None when it isn't a known one (any drill starting mid-game).

    No `squares`: an opening names the whole position, so there is nothing to point an arrow at — and
    a cited fact draws an arrow. Salience is set but unused for ordering (see above); it is kept
    non-zero so the field stays meaningful if the fact is ever ranked.
    """
    from . import openings
    name = openings.name_for(board.fen)
    if not name:
        return None
    return Fact(
        kind="opening",
        squares=[],
        text=f"This is the {name}.",
        provenance="openings-table",
        salience=0.3,
        concept_id="openings",
    )


# -- ranking -----------------------------------------------------------------

# Stable ordering when salience ties: a fixed kind priority, then squares, then
# text — so the same position always produces the same F-numbering.
_KIND_ORDER = {"combination": 0, "threat": 1, "fork": 2, "pin": 3, "hanging": 4,
               "defender-removed": 5, "opening": 6}


def _rank_key(f: Fact):
    return (-f.salience, _KIND_ORDER.get(f.kind, 99), tuple(f.squares), f.text)


def _reconcile_dangers(facts: list[Fact], threats: list[Fact]) -> list[Fact]:
    """Engine-verdict gate on own-piece "danger" hangs (LLD §2.3 hierarchy: *a
    SEE fact that contradicts the engine line loses*).

    `detect_hanging` flags an own piece as hanging whenever the opponent has a
    SEE-winning capture on it — but SEE sees only the one square, blind to a
    recapture or gambit compensation a move later. A pawn "hanging" to `dxc4`
    that the mover simply regains (eval unchanged) is not a real danger; leading
    with it mis-coaches the position.

    The engine already computes the truth: the null-move probe plays the mover's
    pass and reads the opponent's best reply. A danger hang survives ONLY if a
    threat fact confirms the opponent actually wins that piece (same target
    square). Unconfirmed danger hangs are dropped. Opportunity hangs (winning an
    enemy piece) and everything else are untouched.

    Limitation: the probe surfaces the opponent's single best reply, so if two of
    the mover's pieces hang at once only the graver one is confirmed — acceptable,
    since the mover can save only one and the engine leads with the worst.
    """
    confirmed = {t.squares[1] for t in threats if len(t.squares) >= 2}
    kept = []
    for f in facts:
        is_danger_hang = f.kind == "hanging" and f.provenance.startswith("nullsee:")
        if is_danger_hang and f.squares[0] not in confirmed:
            continue  # SEE says hanging, the engine disagrees — drop it
        kept.append(f)
    return kept


# A SEE/geometry "win" (an opportunity hang or a fork) that drops win% by at least this much versus
# the mover's best move is a FALSE opportunity — it nets material on the target square but the engine
# rates taking clearly inferior (a positional cost, a zwischenzug, a back-rank mate). Tied to the eval
# model's MISTAKE line, not a hand-picked number: if the engine would grade the capture a mistake (or
# worse), the coach must not surface it as "you can win X". Was _BLUNDER (15.0), which let mistake-
# level false wins through (e.g. Bxd5 in the corpus — wins a pawn, throws away +1.6). Locked by
# `test_opportunity_reconciliation_corpus`.
_OPP_DROP_THRESHOLD = _MISTAKE


def _reconcile_opportunities(
    board, facts, engine, *, nodes=None, movetime_ms=None
) -> list[Fact]:
    """Engine-verdict gate on opportunity hangs (the §2.3 per-candidate
    counterfactual; same hierarchy rule as `_reconcile_dangers`).

    An opportunity hang says "you can win X (SEE > 0)". SEE nets material on the
    target square but is blind to everything off it — so the capture can still be
    a blunder (a back-rank mate, a zwischenzug, opening your own king). We play
    the capture and read the engine's eval of the result from the mover's POV; if
    taking drops win% by a blunder's worth vs the mover's best move, the SEE fact
    contradicts the engine and is dropped.

    Cost: one analysis of the real position, then one per *suspect* capture — the
    prune skips any capture that already **is** the engine's best move (obviously
    sound), so clean tactics cost nothing extra. All local Stockfish, no Claude
    tokens. (The real-position analysis duplicates one the null-move probe made
    internally; sharing it is a future tidy-up, not a correctness issue.)
    """
    opp = [
        f for f in facts
        if f.kind == "hanging" and f.provenance.startswith("see:")
    ]
    if not opp:
        return facts

    real = engine.analyse(board.fen, nodes=nodes, movetime_ms=movetime_ms, multipv=1)
    best_move = real.best.pv[0] if real.best.pv else None
    best_win = win_pct_from_score(real.best.score)

    dropped: set[str] = set()
    for f in opp:
        uci = f.provenance.split(":", 1)[1]
        if uci == best_move:
            continue  # the engine's own choice — sound, skip the counterfactual
        after = board.apply(uci)
        if not after.legal_moves():
            after_win = 100.0 if after.in_check else 50.0  # capture mates / stalemates
        else:
            after_score = engine.analyse(
                after.fen, nodes=nodes, movetime_ms=movetime_ms, multipv=1
            ).best.score
            after_win = win_pct_from_score(after_score.negated())
        if best_win - after_win >= _OPP_DROP_THRESHOLD:
            dropped.add(f.provenance)  # SEE says win, the engine says blunder — drop

    if not dropped:
        return facts
    return [f for f in facts if f.provenance not in dropped]


def _reconcile_forks(
    board, facts, engine, *, nodes=None, movetime_ms=None
) -> list[Fact]:
    """Engine-verdict gate on fork facts (same §2.3 rule and per-candidate
    counterfactual as `_reconcile_opportunities`).

    `detect_fork` is board-geometry only: it sees that a move attacks two loose
    targets, but not whether the whole shot is *sound* — the forking piece may
    itself be lost to a resource off those squares, or the position simply has
    something clearly better, in which case leading with the fork mis-coaches.
    We play each forking move and read the engine's eval of the result from the
    mover's POV; a fork that drops win% by a blunder's worth versus the mover's
    best move contradicts the engine and is dropped. A fork that already **is**
    the engine's best move is kept without a counterfactual (it is the point of
    the position). All local Stockfish, no Claude tokens."""
    forks = [f for f in facts if f.kind == "fork"]
    if not forks:
        return facts

    real = engine.analyse(board.fen, nodes=nodes, movetime_ms=movetime_ms, multipv=1)
    best_move = real.best.pv[0] if real.best.pv else None
    best_win = win_pct_from_score(real.best.score)

    dropped: set[str] = set()
    for f in forks:
        uci = f.provenance.split(":", 1)[1]
        if uci == best_move:
            continue  # the engine's own choice — sound, skip the counterfactual
        after = board.apply(uci)
        if not after.legal_moves():
            after_win = 100.0 if after.in_check else 50.0
        else:
            after_score = engine.analyse(
                after.fen, nodes=nodes, movetime_ms=movetime_ms, multipv=1
            ).best.score
            after_win = win_pct_from_score(after_score.negated())
        if best_win - after_win >= _OPP_DROP_THRESHOLD:
            dropped.add(f.provenance)  # geometry says fork, the engine says worse — drop

    if not dropped:
        return facts
    return [f for f in facts if f.provenance not in dropped]


def _dedupe(facts: list[Fact]) -> list[Fact]:
    """Collapse facts of the same kind pointing at the same squares, keeping
    the highest-salience one. Two detectors can surface the same motif (e.g. a
    static own-piece hang and the null-move threat on it); we keep the stronger
    statement rather than double-count it in the top five."""
    best: dict[tuple, Fact] = {}
    for f in facts:
        key = (f.kind, tuple(f.squares))
        cur = best.get(key)
        if cur is None or f.salience > cur.salience:
            best[key] = f
    return list(best.values())
