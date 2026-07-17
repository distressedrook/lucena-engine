"""Black-box tests for the M3 fact sheet + tactical detectors.

Written from the M3 contract (`docs/contracts/M3-facts.md`) ONLY — the
implementation under `python/lucena/engine/facts.py` and
`python/lucena/engine/detectors/` was never read. Every expected value below is
derived from the contract's documented semantics/formulae and cross-checked
against an independent board reference (python-chess + Stockfish, in scratch —
never imported here, GPL hygiene).

Engine-touching tests use `nodes=` + `threads=1` (NEVER movetime), per the
load-bearing determinism rule, and are marked `engine` + skipped without a
Stockfish binary. The `Fact` tests and the board-only detector /
`build_fact_sheet(engine=None)` tests need no engine and are never skipped.

Salience formulae used to derive expectations (from the contract):
  hanging:  s = 0.60 + min(0.35, see_cp/2000); danger -> min(0.98, s + 0.05)
  defender-removed: s = 0.50 + min(0.30, E_value_cp/1000)
  threat (material): s = min(0.97, 0.55 + min(0.40, material_cp/2000))
  threat (positional): s = min(0.97, 0.50 + min(0.30, threat_swing/100))
  PIECE_CP: P=100 N=300 B=300 R=500 Q=900 K=0
"""

import shutil

import pytest

from lucena_engine import Board
from lucena_engine import Engine
from lucena_engine.facts import Fact, build_fact_sheet
from lucena_engine.detectors import (
    detect_defender_removed,
    detect_hanging,
    detect_null_move_threat,
)

NODES = 200_000


def _have_stockfish():
    import os

    return bool(os.environ.get("LUCENA_STOCKFISH")) or shutil.which("stockfish")


requires_engine = pytest.mark.skipif(not _have_stockfish(), reason="no stockfish")


@pytest.fixture
def engine():
    with Engine(threads=1) as e:
        yield e


def as_dict(f):
    """Contract-shaped view of a Fact for order-independent comparison."""
    return f.to_dict()


def by_key(facts):
    """Sort facts by (kind, squares) so tests never assume detector order."""
    return sorted(facts, key=lambda f: (f.kind, tuple(f.squares)))


# =====================================================================
# Fact — dataclass semantics
# =====================================================================


def make_fact(**over):
    base = dict(
        kind="hanging",
        squares=["e5", "e1"],
        text="Rxe5 wins the knight on e5",
        provenance="see:e1e5",
        salience=0.75,
        concept_id="hanging-pieces",
    )
    base.update(over)
    return Fact(**base)


def test_fact_field_values():
    f = Fact(
        kind="hanging",
        squares=["e5", "e1"],
        text="Rxe5 wins the knight on e5",
        provenance="see:e1e5",
        salience=0.75,
        concept_id="hanging-pieces",
    )
    assert f.kind == "hanging"
    assert f.squares == ["e5", "e1"]
    assert f.text == "Rxe5 wins the knight on e5"
    assert f.provenance == "see:e1e5"
    assert f.salience == 0.75
    assert f.concept_id == "hanging-pieces"


def test_fact_default_id_is_empty_string():
    assert make_fact().id == ""


def test_with_id_returns_copy_with_id_set():
    f = make_fact()
    g = f.with_id("F3")
    assert g.id == "F3"
    # all other fields identical
    assert (g.kind, g.squares, g.text, g.provenance, g.salience, g.concept_id) == (
        f.kind,
        f.squares,
        f.text,
        f.provenance,
        f.salience,
        f.concept_id,
    )


def test_with_id_does_not_mutate_original():
    f = make_fact()
    _ = f.with_id("F7")
    assert f.id == ""  # frozen: original untouched


def test_with_id_is_reassignable():
    f = make_fact().with_id("F1")
    assert f.with_id("F9").id == "F9"
    assert f.id == "F1"  # previous copy unchanged


def test_to_dict_keys():
    d = make_fact().with_id("F2").to_dict()
    assert set(d.keys()) == {
        "id",
        "kind",
        "squares",
        "text",
        "provenance",
        "salience",
        "concept_id",
    }


def test_to_dict_values():
    d = make_fact().with_id("F2").to_dict()
    assert d["id"] == "F2"
    assert d["kind"] == "hanging"
    assert d["squares"] == ["e5", "e1"]
    assert d["text"] == "Rxe5 wins the knight on e5"
    assert d["provenance"] == "see:e1e5"
    assert d["salience"] == 0.75
    assert d["concept_id"] == "hanging-pieces"


def test_to_dict_emits_empty_id_as_is():
    # No id stripping: a detector-shaped Fact (id="") reports "".
    assert make_fact().to_dict()["id"] == ""


def test_to_dict_squares_is_fresh_list():
    f = make_fact()
    d = f.to_dict()
    d["squares"].append("h8")
    assert f.squares == ["e5", "e1"]  # original untouched


def test_to_dict_rounds_salience_to_3dp():
    f = make_fact(salience=0.123456)
    assert f.to_dict()["salience"] == 0.123


def test_fact_equality_all_fields():
    a = make_fact()
    b = make_fact()
    assert a == b
    assert a != make_fact(salience=0.80)
    assert a != make_fact(squares=["e5", "e2"])
    assert a != make_fact(kind="threat")
    # id participates in equality
    assert a != a.with_id("F1")


def test_fact_is_frozen_and_not_hashable():
    # Contract (reconciled): the dataclass is frozen=True (no rebinding) but a
    # Fact is NOT hashable — its `squares` list makes hash() raise. Facts are
    # never put in sets/dict keys; equality is by value.
    a = make_fact()
    with pytest.raises(Exception):  # FrozenInstanceError
        a.kind = "threat"
    with pytest.raises(TypeError):
        hash(a)


# =====================================================================
# build_fact_sheet — ranking / ids / truncation / dedupe / engine gating
# =====================================================================

# Engine=None deterministic multi-kind sheet: the defender-removed worked
# position also has a static hanging opportunity (exd5 wins the d5 pawn, SEE
# 100 -> salience 0.65). Ranked: hanging 0.65 (> defrem 0.60).
DR_WORKED = "r1bqk2r/ppp2ppp/2n5/1B1pp3/1b2P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1"

# Two undefended enemy knights, each capturable by a rook (SEE 300 each) ->
# two opportunity facts tied at 0.75; tie-break orders ('c5','c1') < ('f5','f1').
TWO_KNIGHTS = "6k1/8/8/2n2n2/8/8/8/2R2RK1 w - - 0 1"

# White knight e4 hangs to the g6 bishop. With an engine this yields BOTH a
# hanging danger (kind hanging, squares [e4,g6], 0.80) and a null-move threat
# (kind threat, squares [g6,e4], 0.70) — different kinds, so no dedupe.
KNIGHT_HANGS = "6k1/5ppp/6b1/8/4N3/8/5PPP/6K1 w - - 0 1"


def test_sheet_engine_none_ranks_by_salience_and_assigns_ids():
    # DR_WORKED is a Ruy-Lopez shape: Bb5 pins Nc6 to the e8 king (a real pin,
    # 0.70) that now leads, ahead of the hanging (0.65) and defender-removed (0.60).
    sheet = build_fact_sheet(Board(DR_WORKED), engine=None, top_n=5)
    assert [f.id for f in sheet] == ["F1", "F2", "F3"]
    assert [f.salience for f in sheet] == [0.70, 0.65, 0.60]
    assert sheet[0].kind == "pin"
    assert sheet[0].squares == ["c6", "b5"]
    assert sheet[1].kind == "hanging"
    assert sheet[1].squares == ["d5", "e4"]
    assert sheet[2].kind == "defender-removed"
    assert sheet[2].squares == ["c6", "e5"]


def test_sheet_engine_none_never_yields_threat():
    sheet = build_fact_sheet(Board(KNIGHT_HANGS), engine=None, top_n=5)
    assert all(f.kind != "threat" for f in sheet)
    # ... but still surfaces the board-only hanging danger.
    assert any(f.kind == "hanging" and f.squares == ["e4", "g6"] for f in sheet)


def test_sheet_every_returned_fact_has_nonempty_id():
    sheet = build_fact_sheet(Board(DR_WORKED), engine=None, top_n=5)
    assert sheet
    assert all(f.id for f in sheet)


def test_sheet_ids_are_sequential_in_rank_order():
    sheet = build_fact_sheet(Board(DR_WORKED), engine=None, top_n=5)
    assert [f.id for f in sheet] == [f"F{i}" for i in range(1, len(sheet) + 1)]


def test_sheet_tie_break_by_squares_within_kind():
    sheet = build_fact_sheet(Board(TWO_KNIGHTS), engine=None, top_n=5)
    assert [f.salience for f in sheet] == [0.75, 0.75]
    assert [f.squares for f in sheet] == [["c5", "c1"], ["f5", "f1"]]
    assert [f.id for f in sheet] == ["F1", "F2"]


# At equal salience the tie-break's kind_priority (threat=0, hanging=1,
# defrem=2) decides. This board-only position yields a hanging DANGER (own a1
# bishop, Rxa1+, SEE 300 -> 0.80) and a defender-removed (d6 is e5-knight's only
# defender, E=knight 300 -> 0.80), both at 0.80 -> hanging must precede defrem.
KIND_TIE = "r5k1/8/3p4/2P1n3/5P2/8/8/B5K1 w - - 0 1"


def test_sheet_tie_break_hanging_before_defender_removed_at_equal_salience():
    sheet = build_fact_sheet(Board(KIND_TIE), engine=None, top_n=5)
    at_080 = [(f.kind, f.squares) for f in sheet if f.salience == 0.80]
    assert ("hanging", ["a1", "a8"]) in at_080
    assert ("defender-removed", ["d6", "e5"]) in at_080
    kinds = [f.kind for f in sheet]
    assert kinds.index("hanging") < kinds.index("defender-removed")


def test_sheet_top_n_truncates_keeping_highest():
    sheet = build_fact_sheet(Board(DR_WORKED), engine=None, top_n=1)
    assert len(sheet) == 1
    assert sheet[0].id == "F1"
    assert sheet[0].kind == "pin"        # the highest-salience fact (0.70) survives
    assert sheet[0].salience == 0.70


def test_sheet_top_n_zero_returns_empty():
    assert build_fact_sheet(Board(DR_WORKED), engine=None, top_n=0) == []


def test_sheet_top_n_negative_returns_empty():
    assert build_fact_sheet(Board(DR_WORKED), engine=None, top_n=-3) == []


def test_sheet_quiet_position_is_empty():
    sheet = build_fact_sheet(Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), engine=None)
    assert sheet == []


def test_sheet_unique_kind_squares_after_dedupe():
    sheet = build_fact_sheet(Board(DR_WORKED), engine=None, top_n=5)
    keys = [(f.kind, tuple(f.squares)) for f in sheet]
    assert len(keys) == len(set(keys))


def test_sheet_default_top_n_is_5():
    # Three facts exist here (pin, hanging, defender-removed), all <= 5.
    sheet = build_fact_sheet(Board(DR_WORKED), engine=None)
    assert len(sheet) == 3


@pytest.mark.engine
@requires_engine
def test_sheet_with_engine_adds_threat_without_deduping_hanging(engine):
    engine.new_game()
    sheet = build_fact_sheet(Board(KNIGHT_HANGS), engine=engine, nodes=NODES, top_n=5)
    kinds = {f.kind for f in sheet}
    assert "threat" in kinds
    assert "hanging" in kinds
    # hanging danger (0.80) outranks the threat (0.70); different kinds coexist.
    hanging = next(f for f in sheet if f.kind == "hanging")
    threat = next(f for f in sheet if f.kind == "threat")
    assert hanging.squares == ["e4", "g6"]
    assert threat.squares == ["g6", "e4"]
    assert hanging.salience == 0.80
    assert threat.salience == 0.70
    assert sheet.index(hanging) < sheet.index(threat)
    assert all(f.id for f in sheet)


# A regained gambit pawn: SEE flags c4 as hanging (dxc4), but the engine sees
# White regains it, so the null-move probe confirms no threat. (Complements
# KNIGHT_HANGS above, where the danger IS confirmed and kept.)
GAMBIT_PAWN = "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 1"


@pytest.mark.engine
@requires_engine
def test_sheet_reconciles_away_unconfirmed_danger_hang(engine):
    # Board-only, SEE flags c4 as a danger hang...
    board_only = detect_hanging(Board(GAMBIT_PAWN))
    assert any(
        f.provenance.startswith("nullsee:") and f.squares[0] == "c4"
        for f in board_only
    ), "precondition: SEE should flag c4 as a danger hang"
    # ...but with the engine, the null-move probe finds no threat on c4 (White
    # regains the pawn), so the contradicted SEE fact is dropped.
    engine.new_game()
    sheet = build_fact_sheet(Board(GAMBIT_PAWN), engine=engine, nodes=NODES, top_n=5)
    assert all(
        not (f.kind == "hanging" and f.squares[0] == "c4") for f in sheet
    ), f"unconfirmed c4 danger hang should be reconciled away; got {[f.text for f in sheet]}"


def test_sheet_engine_none_keeps_unconfirmed_danger_hang():
    # Without an engine there is no reconciliation; the SEE danger hang survives
    # (best-effort — the board core cannot see the regain).
    sheet = build_fact_sheet(Board(GAMBIT_PAWN), engine=None, top_n=5)
    assert any(f.kind == "hanging" and f.squares[0] == "c4" for f in sheet)


# A back-rank trap: SEE says Rxd5 wins the bishop, but Rd1 is the sole first-rank
# guard, so after Rxd5, Re1# (material is otherwise even). The engine's best move
# is not the capture, and the capture drops win% by a blunder's worth.
OPP_TRAP = "4r1k1/5ppp/8/3b4/7B/8/5PPP/3R2K1 w - - 0 1"
# A sound opportunity: Rxe5 wins the undefended knight and IS the engine's best
# move — the gate's prune skips it, so it survives.
KNIGHT_HANGS_OPP = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"


@pytest.mark.engine
@requires_engine
def test_sheet_reconciles_away_trap_opportunity(engine):
    # Board-only, SEE flags Rxd5 as an opportunity...
    board_only = detect_hanging(Board(OPP_TRAP))
    assert any(
        f.provenance == "see:d1d5" and f.squares[0] == "d5" for f in board_only
    ), "precondition: SEE should flag Rxd5 as winning the bishop"
    # ...but with the engine, playing Rxd5 walks into Re1#, so the fact is dropped.
    engine.new_game()
    sheet = build_fact_sheet(Board(OPP_TRAP), engine=engine, nodes=NODES, top_n=5)
    assert all(
        f.provenance != "see:d1d5" for f in sheet
    ), f"trap opportunity should be reconciled away; got {[f.text for f in sheet]}"


def test_sheet_engine_none_keeps_trap_opportunity():
    # Without an engine there is no counterfactual; the SEE opportunity survives.
    sheet = build_fact_sheet(Board(OPP_TRAP), engine=None, top_n=5)
    assert any(f.provenance == "see:d1d5" for f in sheet)


@pytest.mark.engine
@requires_engine
def test_sheet_keeps_sound_opportunity(engine):
    # A genuinely winning capture (Rxe5, the engine's best move) is NOT dropped by
    # the opportunity gate — the prune skips the counterfactual for a best move.
    engine.new_game()
    sheet = build_fact_sheet(Board(KNIGHT_HANGS_OPP), engine=engine, nodes=NODES, top_n=5)
    assert any(f.provenance == "see:e1e5" and f.squares[0] == "e5" for f in sheet)


@pytest.mark.engine
@requires_engine
def test_sheet_engine_without_limit_raises_value_error(engine):
    # Passing an engine but neither nodes nor movetime_ms surfaces the
    # Engine.analyse "exactly one limit" ValueError.
    with pytest.raises(ValueError):
        build_fact_sheet(Board(KNIGHT_HANGS), engine=engine, top_n=5)


# =====================================================================
# detect_hanging
# =====================================================================


def test_hanging_opportunity_worked_example():
    facts = detect_hanging(Board("6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"))
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == "hanging"
    assert f.squares == ["e5", "e1"]
    assert f.text == "White can play Rxe5, winning the knight on e5"
    assert f.provenance == "see:e1e5"
    assert f.salience == 0.75
    assert f.concept_id == "hanging-pieces"
    assert f.id == ""


def test_hanging_danger_worked_example():
    facts = detect_hanging(Board("6k1/5ppp/6b1/8/4N3/8/5PPP/6K1 w - - 0 1"))
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == "hanging"
    assert f.squares == ["e4", "g6"]
    assert f.text == "Black threatens Bxe4, winning White's knight on e4"
    assert f.provenance == "nullsee:g6e4"
    assert f.salience == 0.80
    assert f.concept_id == "hanging-pieces"
    assert f.id == ""


def test_hanging_defended_piece_does_not_fire():
    # d5 knight defended by the d8 queen; Rxd5 has SEE < 0 -> no fact on d5.
    facts = detect_hanging(Board("3q1rk1/5ppp/8/3n4/8/8/5PPP/3R1BK1 w - - 0 1"))
    assert all(f.squares[0] != "d5" for f in facts)


def test_hanging_empty_position_returns_empty():
    assert detect_hanging(Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")) == []


def test_hanging_all_detector_facts_have_empty_id():
    facts = detect_hanging(Board("6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"))
    assert facts
    assert all(f.id == "" for f in facts)


# --- opportunity salience by victim value (undefended -> SEE == piece value) ---


@pytest.mark.parametrize(
    "fen,squares,name,prov,sal",
    [
        ("6k1/5ppp/3q4/1N6/8/8/5PPP/6K1 w - - 0 1", ["d6", "b5"], "queen", "see:b5d6", 0.95),
        ("6k1/5ppp/3r4/1N6/8/8/5PPP/6K1 w - - 0 1", ["d6", "b5"], "rook", "see:b5d6", 0.85),
        ("6k1/5ppp/3p4/1N6/8/8/5PPP/6K1 w - - 0 1", ["d6", "b5"], "pawn", "see:b5d6", 0.65),
    ],
)
def test_hanging_opportunity_salience_by_value(fen, squares, name, prov, sal):
    facts = detect_hanging(Board(fen))
    assert len(facts) == 1
    f = facts[0]
    assert f.squares == squares
    assert f.text == f"White can play Nxd6, winning the {name} on d6"
    assert f.provenance == prov
    assert f.salience == sal


# --- danger salience by victim value (own undefended piece) ---


def test_hanging_danger_queen_salience_capped():
    # queen SEE 900 danger: 0.95 + 0.05 = 1.00 -> capped at 0.98.
    facts = detect_hanging(Board("6k1/5ppp/8/5n2/3Q4/8/5PPP/6K1 w - - 0 1"))
    assert len(facts) == 1
    f = facts[0]
    assert f.squares == ["d4", "f5"]
    assert f.text == "Black threatens Nxd4, winning White's queen on d4"
    assert f.provenance == "nullsee:f5d4"
    assert f.salience == 0.98


def test_hanging_danger_pawn_salience():
    # pawn SEE 100 danger: 0.65 + 0.05 = 0.70.
    facts = detect_hanging(Board("6k1/5ppp/8/3P4/4b3/8/5PPP/6K1 w - - 0 1"))
    assert len(facts) == 1
    f = facts[0]
    assert f.squares == ["d5", "e4"]
    assert f.text == "Black threatens Bxd5, winning White's pawn on d5"
    assert f.provenance == "nullsee:e4d5"
    assert f.salience == 0.70


def test_hanging_en_passant_victim_square():
    # exd6 e.p. captures the d5 pawn; victim_square is d5 (where the pawn is),
    # NOT the d6 destination. SAN of the move is "exd6".
    facts = detect_hanging(Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"))
    assert len(facts) == 1
    f = facts[0]
    assert f.squares == ["d5", "e5"]
    assert f.text == "White can play exd6, winning the pawn on d5"
    assert f.provenance == "see:e5d6"
    assert f.salience == 0.65


def test_hanging_in_check_yields_opportunities_only():
    # White is in check from the f3 knight; gxf3 both wins the knight and
    # resolves the check (opportunity). The a1 bishop hangs to the a8 rook,
    # but danger detection is skipped entirely while in check.
    fen = "r3k3/8/8/8/8/5n2/6P1/B3K3 w - - 0 1"
    facts = detect_hanging(Board(fen))
    assert len(facts) == 1
    f = facts[0]
    assert f.provenance.startswith("see:")  # opportunity, not "nullsee:"
    assert f.squares == ["f3", "g2"]
    assert f.text == "White can play gxf3, winning the knight on f3"
    assert f.salience == 0.75
    # control: with the checker moved away the danger *does* fire.
    control = detect_hanging(Board("r3k3/8/8/8/6n1/8/6P1/B3K3 w - - 0 1"))
    assert any(
        g.provenance == "nullsee:a8a1" and g.squares == ["a1", "a8"] for g in control
    )


def test_hanging_enemy_king_never_victim():
    # No fact ever names the enemy king square as its victim (squares[0]).
    fen = "6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1"
    facts = detect_hanging(Board(fen))
    assert all(f.squares[0] != "g8" for f in facts)


def test_hanging_king_as_attacker():
    # The contract permits a king to be the *attacker* winning a hanging piece.
    # Kxe5 wins the undefended e5 knight (SEE 300 -> opportunity salience 0.75);
    # attacker origin d4 (K has PIECE_CP 0 in the LVA tie-break path).
    facts = detect_hanging(Board("7k/8/8/4n3/3K4/8/8/8 w - - 0 1"))
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == "hanging"
    assert f.squares == ["e5", "d4"]
    assert f.text == "White can play Kxe5, winning the knight on e5"
    assert f.provenance == "see:d4e5"
    assert f.salience == 0.75


def test_hanging_does_not_mutate_board():
    b = Board("6k1/5ppp/8/4n3/8/8/5PPP/4R1K1 w - - 0 1")
    fen_before = b.fen
    detect_hanging(b)
    assert b.fen == fen_before


# =====================================================================
# detect_defender_removed
# =====================================================================


def test_defender_removed_worked_example():
    fen = "r1bqk2r/ppp2ppp/2n5/1B1pp3/1b2P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1"
    facts = detect_defender_removed(Board(fen))
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == "defender-removed"
    assert f.squares == ["c6", "e5"]
    assert f.text == "the knight on c6 is the only defender of the pawn on e5"
    assert f.provenance == "static:c6->e5"
    assert f.salience == 0.60
    assert f.concept_id == "removing-the-defender"
    assert f.id == ""


def test_defender_removed_knight_victim_salience():
    # E = d5 knight (300), lone defender D = c6 pawn, D attacked by Bb5.
    fen = "6k1/8/2p5/1B1n4/4P3/8/8/6K1 w - - 0 1"
    facts = detect_defender_removed(Board(fen))
    assert len(facts) == 1
    f = facts[0]
    assert f.squares == ["c6", "d5"]
    assert f.text == "the pawn on c6 is the only defender of the knight on d5"
    assert f.provenance == "static:c6->d5"
    assert f.salience == 0.80


def test_defender_removed_queen_victim_salience_capped():
    # E = queen (900): 0.50 + min(0.30, 0.90) = 0.80 — the +0.30 cap holds, the
    # queen does not push salience above the knight-victim value.
    # Bg2 attacks the d5 queen; its lone defender Nf6 is attacked by the e5 pawn.
    fen = "6k1/8/5n2/3qP3/8/8/6B1/6K1 w - - 0 1"
    facts = detect_defender_removed(Board(fen))
    assert len(facts) == 1
    f = facts[0]
    assert f.squares == ["f6", "d5"]
    assert f.text == "the knight on f6 is the only defender of the queen on d5"
    assert f.salience == 0.80


def test_defender_removed_requires_exactly_one_defender():
    # Adding an f6 pawn gives e5 a second defender -> the motif does not fire.
    fen = "r1bqk2r/ppp2p1p/2n2p2/1B1pp3/1b2P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1"
    facts = detect_defender_removed(Board(fen))
    assert all(f.squares != ["c6", "e5"] for f in facts)


def test_defender_removed_requires_guard_attacked_by_stm():
    # Bishop on c4 (not b5) no longer attacks the c6 guard -> no fact on e5.
    fen = "r1bqk2r/ppp2ppp/2n5/3pp3/1bB1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1"
    facts = detect_defender_removed(Board(fen))
    assert all(f.squares != ["c6", "e5"] for f in facts)


def test_defender_removed_all_facts_have_empty_id():
    fen = "r1bqk2r/ppp2ppp/2n5/1B1pp3/1b2P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1"
    facts = detect_defender_removed(Board(fen))
    assert facts
    assert all(f.id == "" for f in facts)


def test_defender_removed_squares0_is_defender():
    # squares[0] is the defender D, squares[1] is the victim E (contract).
    fen = "r1bqk2r/ppp2ppp/2n5/1B1pp3/1b2P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1"
    f = detect_defender_removed(Board(fen))[0]
    assert f.provenance == f"static:{f.squares[0]}->{f.squares[1]}"


def test_defender_removed_does_not_mutate_board():
    fen = "r1bqk2r/ppp2ppp/2n5/1B1pp3/1b2P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1"
    b = Board(fen)
    before = b.fen
    detect_defender_removed(b)
    assert b.fen == before


# =====================================================================
# detect_null_move_threat  (engine)
# =====================================================================

MATERIAL_THREAT = "6k1/5ppp/6b1/8/4N3/8/5PPP/6K1 w - - 0 1"  # Ne4 hangs to Bg6
POSITIONAL_THREAT = "4r1k1/R4ppp/8/8/8/8/5PPP/6K1 w - - 0 1"  # ...Re1# after a pass
QUIET_NO_THREAT = "4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1"  # symmetric, swing ~ 0
IN_CHECK = "6k1/5ppp/8/8/8/8/5qPP/6K1 w - - 0 1"  # Qf2+ -> null move illegal


@pytest.mark.engine
@requires_engine
def test_threat_in_check_returns_empty(engine):
    engine.new_game()
    assert detect_null_move_threat(Board(IN_CHECK), engine, nodes=NODES) == []


# White is up a queen; if White passes, Black (Kh8) has no legal move — passing
# would stalemate the opponent. The probe must return [] without analysing the
# terminal passed position (regression: it used to raise EngineError).
PASS_TO_STALEMATE = "7k/5K2/8/8/8/8/8/1Q6 w - - 0 1"


@pytest.mark.engine
@requires_engine
def test_threat_returns_empty_when_pass_leaves_no_reply(engine):
    engine.new_game()
    assert detect_null_move_threat(Board(PASS_TO_STALEMATE), engine, nodes=NODES) == []


@pytest.mark.engine
@requires_engine
def test_threat_material_capture(engine):
    engine.new_game()
    facts = detect_null_move_threat(Board(MATERIAL_THREAT), engine, nodes=NODES)
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == "threat"
    assert f.concept_id == "tactical-signals"
    assert f.squares == ["g6", "e4"]  # [M_from, M_to]
    assert f.provenance == "nullmove:g6e4"
    assert f.text == "after a pass, Black plays Bxe4, winning White's knight on e4"
    # material_cp = 300 -> 0.55 + min(0.40, 300/2000) = 0.70 (node-independent).
    assert f.salience == 0.70
    assert f.id == ""


@pytest.mark.engine
@requires_engine
def test_threat_positional_branch(engine):
    engine.new_game()
    facts = detect_null_move_threat(Board(POSITIONAL_THREAT), engine, nodes=NODES)
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == "threat"
    assert f.squares == ["e8", "e1"]
    assert f.provenance == "nullmove:e8e1"
    # a mate threat is now flagged as a mate (loudest severity tier).
    assert f.text == "the opponent threatens mate: Re1#"
    assert f.salience == 0.98
    assert f.id == ""


@pytest.mark.engine
@requires_engine
def test_threat_quiet_position_no_threat(engine):
    engine.new_game()
    # Symmetric position: threat_swing ~ 0 (< 10) -> no fact.
    assert detect_null_move_threat(Board(QUIET_NO_THREAT), engine, nodes=NODES) == []


@pytest.mark.engine
@requires_engine
def test_threat_at_most_one_fact(engine):
    engine.new_game()
    facts = detect_null_move_threat(Board(MATERIAL_THREAT), engine, nodes=NODES)
    assert len(facts) <= 1


@pytest.mark.engine
@requires_engine
def test_threat_deterministic_across_fresh_engines():
    # nodes + threads=1 from two fresh engines -> identical fact.
    def run():
        with Engine(threads=1) as e:
            facts = detect_null_move_threat(Board(MATERIAL_THREAT), e, nodes=NODES)
            return [as_dict(f) for f in facts]

    assert run() == run()


@pytest.mark.engine
@requires_engine
def test_threat_requires_a_limit(engine):
    # Neither nodes nor movetime_ms -> ValueError from Engine.analyse.
    with pytest.raises(ValueError):
        detect_null_move_threat(Board(MATERIAL_THREAT), engine)
