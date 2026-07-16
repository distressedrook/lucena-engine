"""The opening-name table, and the opening fact.

There was no test here, which is precisely how the table came to be looked up at the wrong path and
stay broken silently: `_table()` swallowed the OSError, every lookup returned None, and None is
indistinguishable from "not a known opening". The feature was simply absent and nothing failed.

So the first test asserts the table LOADS — not merely that lookups don't raise.
"""

from lucena_engine import openings
from lucena_engine.board import Board
from lucena_engine.facts import build_fact_sheet

# The START POSITION IS NOT IN THE TABLE: a name needs at least one move, so the book begins at ply 1.
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
KINGS_PAWN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"   # 1.e4
SICILIAN = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2"    # 1.e4 c5
RUY_LOPEZ = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
# Genuinely off-book: an endgame no opening line reaches.
NOT_AN_OPENING = "8/8/8/4k3/8/4K3/8/7R w - - 0 1"


def test_the_table_actually_loads():
    """The regression that hid: a wrong path + a swallowed OSError = an empty table forever."""
    assert openings._PATH.exists(), f"the shipped table is missing at {openings._PATH}"
    assert len(openings._table()) > 1000, "the table loaded but is implausibly small"


def test_the_table_is_inside_the_package():
    """It must resolve relative to the module, not its parent: an installed wheel has no
    `site-packages/data/`, so a path one level up exists in no install."""
    import lucena_engine
    from pathlib import Path
    pkg = Path(lucena_engine.__file__).resolve().parent
    assert openings._PATH.is_relative_to(pkg), \
        f"{openings._PATH} is outside the package ({pkg}) — it will not survive an install"


def test_known_openings_are_named():
    assert openings.name_for(KINGS_PAWN)
    assert "Sicilian" in (openings.name_for(SICILIAN) or "")


def test_an_unknown_position_is_not_named():
    """A drill starting mid-game has no opening — None, not a guess."""
    assert openings.name_for(NOT_AN_OPENING) is None


def test_the_start_position_has_no_name():
    """The book begins at ply 1: an unplayed board is not an opening."""
    assert openings.name_for(START) is None


def test_lookup_ignores_move_counters():
    """Position identity is placement/side/castling/ep — the same position reached by a different
    move order (different clocks) is the same opening."""
    same_position_later_clocks = KINGS_PAWN.replace(" 0 1", " 4 9")
    assert openings.name_for(same_position_later_clocks) == openings.name_for(KINGS_PAWN)


# -- the fact ----------------------------------------------------------------

def _facts(fen, **kw):
    return build_fact_sheet(Board(fen), None, **kw)


def _openings_in(facts):
    return [f for f in facts if f.kind == "opening"]


def test_the_fact_sheet_emits_the_opening():
    facts = _facts(SICILIAN)
    got = _openings_in(facts)
    assert len(got) == 1
    assert "Sicilian" in got[0].text
    assert got[0].id, "the opening fact was not assigned an F-id"


def test_no_opening_fact_when_the_position_is_not_an_opening():
    assert _openings_in(_facts(NOT_AN_OPENING)) == []


def test_the_opening_points_at_no_squares():
    """A cited fact draws an arrow; an opening names the whole position, so there is nothing to
    point at."""
    assert _openings_in(_facts(RUY_LOPEZ))[0].squares == []


def test_the_opening_does_not_consume_a_tactical_slot():
    """`top_n` bounds the TACTICS. The opening rides alongside the ranking, so it can neither evict a
    real tactic nor be evicted by one — it must not trade off against them."""
    tactical_only = build_fact_sheet(Board(RUY_LOPEZ), None, top_n=1)
    assert len(_openings_in(tactical_only)) == 1
    # Everything else still fits its own budget.
    assert len([f for f in tactical_only if f.kind != "opening"]) <= 1


def test_ids_stay_contiguous_with_the_opening_appended():
    facts = _facts(RUY_LOPEZ)
    assert [f.id for f in facts] == [f"F{i + 1}" for i in range(len(facts))]


# -- the en-passant convention ---------------------------------------------------
#
# The table keys on norm_fen, which KEEPS the ep field, and it stores ep unconditionally on a double
# push (778/3733 rows carry one; `1.e4` is keyed with `e3`). That only works because our board core
# uses the same convention. python-chess uses the LEGAL-ONLY convention, against which every double
# push resolves to None.
#
# These MUST go through Board.apply, not FEN literals. The other tests in this file use hardcoded
# FENs and would keep passing if the board core switched conventions tomorrow — while every opening
# starting with a double push silently vanished. A miss is indistinguishable from "not an opening",
# which is exactly how the broken table path shipped unnoticed.

def _line(ucis):
    """Replay ucis from the start THROUGH THE BOARD CORE; return the fen after each."""
    b, out = Board(START), []
    for u in ucis:
        b = b.apply(u)
        out.append(b.fen)
    return out


def test_a_double_push_is_named_through_the_board_core():
    """1.e4 via Board.apply must resolve. This is the ep-convention pin."""
    (after_e4,) = _line(["e2e4"])
    assert openings.name_for(after_e4) == "King's Pawn Game", (
        f"the board core's ep convention no longer matches the table: {after_e4}"
    )


def test_the_board_core_writes_ep_on_any_double_push():
    """The convention itself, stated: ep is set even when no capture is possible."""
    (after_e4,) = _line(["e2e4"])
    assert after_e4.split()[3] == "e3", f"expected ep=e3, got {after_e4.split()[3]!r}"


def test_every_first_move_double_push_is_named():
    for uci, expect in [("e2e4", "King's Pawn Game"), ("d2d4", "Queen's Pawn Game")]:
        (fen,) = _line([uci])
        assert openings.name_for(fen) == expect, f"{uci} -> {openings.name_for(fen)!r}"


# -- book_name: the fold ---------------------------------------------------------

RUY = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
NIMZO = ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4"]


def _names(ucis):
    """The folded name after each ply."""
    fens = _line(ucis)
    return [openings.book_name(fens[:i + 1]) for i in range(len(fens))]


def test_book_name_follows_a_line_into_its_opening():
    assert _names(RUY)[4] == "Ruy Lopez"
    assert _names(RUY)[5] == "Ruy Lopez: Morphy Defense"
    assert _names(NIMZO)[-1] == "Nimzo-Indian Defense"


def test_unnamed_plies_are_sticky():
    """Ruy Ba4/Nf6 are not in the table; the line is still the Morphy Defense."""
    n = _names(RUY)
    assert openings.name_for(_line(RUY)[6]) is None      # Ba4 really is unnamed
    assert n[6] == "Ruy Lopez: Morphy Defense"           # ...but the fold holds
    assert n[7] == "Ruy Lopez: Morphy Defense"


def test_the_name_never_coarsens():
    """THE regression that killed `deepest()`. The table re-attaches coarser names deeper in a line;
    naive last-wins walks backwards and the coach appears to forget what it just said."""
    najdorf = ["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"]
    names = _names(najdorf)
    assert openings.name_for(_line(najdorf)[4]) == "Sicilian Defense"   # the raw table DOES coarsen
    assert names[3] == "Sicilian Defense: Modern Variations"
    assert names[4] == "Sicilian Defense: Modern Variations", \
        "the fold coarsened back to 'Sicilian Defense' — deepest()'s bug"


def test_a_line_that_never_enters_the_book_has_no_name():
    assert openings.book_name([NOT_AN_OPENING]) is None
    assert openings.book_name([]) is None


def test_change_detection_is_stateless():
    """The cadence rule: fire when book_name(hist) != book_name(hist[:-1]). No stored state."""
    fens = _line(RUY)
    fired = [i for i in range(len(fens))
             if openings.book_name(fens[:i + 1]) != openings.book_name(fens[:i])]
    # e4(0) e5(1) Nf3(2) Nc6(3) Bb5(4) a6(5) Ba4(6) Nf6(7)
    assert 1 not in fired, "1...e5 repeats 'King's Pawn Game' — must not fire"
    assert 6 not in fired and 7 not in fired, "unnamed plies must not fire"
    assert 4 in fired and 5 in fired, "Bb5 -> Ruy Lopez and a6 -> Morphy must fire"


# -- plies_since_named -----------------------------------------------------------

def test_plies_since_named_is_zero_in_book():
    assert openings.plies_since_named(_line(["e2e4"])) == 0


def test_plies_since_named_counts_the_gap():
    fens = _line(RUY)
    assert openings.plies_since_named(fens[:7]) == 1   # Ba4
    assert openings.plies_since_named(fens[:8]) == 2   # Nf6


def test_plies_since_named_is_none_off_book_entirely():
    """A pasted midgame FEN never entered the book — 'you have left it' is meaningless there and
    must never be announced."""
    assert openings.plies_since_named([NOT_AN_OPENING]) is None
    assert openings.plies_since_named([]) is None


def test_a_real_line_never_reaches_the_off_book_threshold():
    """THRESHOLD=4 must not false-fire on mainline theory. The Ruy's biggest internal gap is 2."""
    for line in (RUY, NIMZO):
        fens = _line(line)
        worst = max(openings.plies_since_named(fens[:i + 1]) or 0 for i in range(len(fens)))
        assert worst < 4, f"a mainline hit psn={worst}; THRESHOLD=4 would false-announce"


# -- the ep-stripped fallback ----------------------------------------------------
#
# Our board core (and the table) record ep on ANY double push. Everyone else — Lichess, chess.com,
# python-chess — records it only when a capture is actually available. A FEN pasted from outside
# therefore misses 21% of the book, silently.

# 1.e4 as the rest of the world writes it: no ep square, because no capture is available.
KINGS_PAWN_LEGAL_EP = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"


def test_a_foreign_fen_with_legal_only_ep_still_resolves():
    """The paste path: a FEN copied from Lichess after 1.e4 carries ep '-', not 'e3'."""
    assert KINGS_PAWN_LEGAL_EP.split()[3] == "-"                      # the foreign convention
    assert openings.name_for(KINGS_PAWN_LEGAL_EP) == "King's Pawn Game"


def test_our_own_fens_still_hit_the_exact_index():
    """The fallback must not shadow an exact hit."""
    (ours,) = _line(["e2e4"])
    assert ours.split()[3] == "e3"                                    # our convention
    assert openings.name_for(ours) == "King's Pawn Game"


def test_the_ambiguous_stripped_key_is_declined_not_guessed():
    """Exactly one placement appears in the table under two names — and the ep field is what tells
    them apart, so stripping it would be a coin flip:

        ep='e3' -> Van Geet Opening: Nowokunski Gambit        (White just played e2-e4)
        ep='-'  -> King's Gambit Accepted: Mason-Keres Gambit (White just played Nc3)

    Same pieces, different move orders. Both exact spellings must keep resolving to their OWN name,
    and the stripped key must be absent from the fallback so it can never be guessed.

    (In this position the ep capture f4xe3 is genuinely legal, so the foreign legal-only convention
    writes 'e3' here too and hits the exact index — the exclusion costs nothing real.)
    """
    placement = "rnbqkbnr/pppp1ppp/8/8/4Pp2/2N5/PPPP2PP/R1BQKBNR b KQkq"
    assert placement not in openings._ep_stripped_table(), \
        "the ambiguous key is guessable through the fallback"
    assert openings.name_for(f"{placement} e3 0 3") == "Van Geet Opening: Nowokunski Gambit"
    assert openings.name_for(f"{placement} - 0 3") == "King's Gambit Accepted: Mason-Keres Gambit"


def test_the_fallback_never_invents_a_name():
    """An off-book position must stay off-book through the fallback too."""
    assert openings.name_for(NOT_AN_OPENING) is None
    assert openings.name_for("8/8/8/8/8/8/8/K6k w - - 0 1") is None


def test_names_with_commas_before_a_colon_parse_sanely():
    """45 table names put a comma before any colon. The ancestor test splits on both, so check the
    fold treats such a name as one node rather than mis-nesting it."""
    commas = [n for n in openings._table().values() if "," in n.split(":")[0]]
    assert commas, "expected some comma-first names in the table"
    for name in commas[:20]:
        assert openings._is_ancestor(name, name), f"{name!r} is not its own ancestor"
        # A comma-first name must not be read as an ancestor of an unrelated line.
        assert not openings._is_ancestor(name, "Ruy Lopez") or name.startswith("Ruy Lopez")
