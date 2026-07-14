"""Black-box tests for `lucena_engine.positional.analyze_positional`.

Written from the contract ONLY (docs/contracts/M-positional.md). Nothing under
`python/lucena/engine/positional.py` or `_pesto.py` was opened, grepped, or run
to discover an expected value. Every expectation below is derived from the
contract's documented semantics, cited public standards, and independent board
reasoning — with python-chess (`import chess`, permitted under tests/) as the
board oracle, and `lucena.mcp.response.material` used ONLY where the contract
explicitly declares material.standing to equal it (never the positional impl).

Coverage map (contract sections):
  * Invariants 1-7 (determinism, shape, start symmetry, colour-mirror
    antisymmetry, no-eval-numbers in standings, the `leads` law, no-crash on
    sparse boards).
  * The sign rule per term + side-to-move independence of cp.
  * Standings swap under the colour-mirror.
  * The four anchored examples.
  * The per-term `features` schemas.
"""

import re

import chess
import pytest

from lucena_engine.board import Board
from lucena_engine.positional import analyze_positional
from lucena_engine import reads as R


# --------------------------------------------------------------------------
# Positions (verified independently with python-chess in scratch)
# --------------------------------------------------------------------------

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Anchored examples (contract "Anchored examples" table).
PASSER_EP = "8/6k1/4P3/8/8/8/5K2/8 w - - 0 1"          # White e6 passer, K+P endgame
IQP = "r1bq1rk1/pp2bppp/2n2n2/3p4/3P4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 11"  # black d5 isolani
KING_DANGER = "r4rk1/ppp2p2/3p3p/4p3/4P1nq/2NP4/PPP2PP1/R3QRK1 w - - 0 15"  # Qh4+Ng4 hit g1

# Sign-rule fixtures.
WHITE_EXTRA_N = "4k3/8/8/8/8/8/8/N3K3 w - - 0 1"        # White up a clean knight
CENTER_WHITE = "4k3/8/8/8/3PP3/8/8/4K3 w - - 0 1"       # only White holds d4/e4
ACTIVITY = "1n2k3/8/8/3N4/8/8/8/4K3 w - - 0 1"          # White N on d5 vs Black N on b8

# Non-standard / degenerate but LEGALLY CONSTRUCTIBLE boards.
KINGS_ONLY = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"            # self-mirror -> everything balanced
DOUBLED = "4k3/8/8/8/8/3P4/3P4/4K3 w - - 0 1"          # White doubled+isolated d-pawns
SPARSE = "4k3/8/8/3n4/8/4B3/8/4K3 w - - 0 1"           # a knight vs a bishop, kings

MIRROR_CASES = [PASSER_EP, IQP, KING_DANGER, WHITE_EXTRA_N, CENTER_WHITE, ACTIVITY]
LEADS_LAW_CASES = [START, PASSER_EP, IQP, KING_DANGER, WHITE_EXTRA_N,
                   CENTER_WHITE, ACTIVITY, KINGS_ONLY, DOUBLED, SPARSE]

TERM_NAMES = {"material", "king_safety", "activity", "pawns", "center"}
FEATURE_TERMS = TERM_NAMES - {"material"}  # material features are optional

_SQUARE_RE = re.compile(r"^[a-h][1-8]$")


# --------------------------------------------------------------------------
# Oracles / helpers (contract-derived, python-chess as board oracle)
# --------------------------------------------------------------------------

def _mirror_fen(fen):
    """Colour-swapped, vertically-flipped twin (contract invariant 4)."""
    return chess.Board(fen).mirror().fen()


def _flip_stm(fen):
    """Same placement, opposite side-to-move (must stay legal both ways)."""
    parts = fen.split(" ")
    parts[1] = "b" if parts[1] == "w" else "w"
    return " ".join(parts)


def _expected_phase(fen):
    """phase = min(24, minors + 2*rooks + 4*queens)/24, both colours, 2dp."""
    b = chess.Board(fen)
    minors = sum(len(b.pieces(p, c))
                 for p in (chess.KNIGHT, chess.BISHOP) for c in (True, False))
    rooks = sum(len(b.pieces(chess.ROOK, c)) for c in (True, False))
    queens = sum(len(b.pieces(chess.QUEEN, c)) for c in (True, False))
    return round(min(24, minors + 2 * rooks + 4 * queens) / 24, 2)


def _is_int(x):
    # bool is a subclass of int; the contract says cp/counts are ints, not bools.
    return isinstance(x, int) and not isinstance(x, bool)


def _assert_term_shape(name, term):
    assert set(term) >= {"cp", "standing"}, name
    assert _is_int(term["cp"]), (name, term["cp"])
    assert isinstance(term["standing"], str) and term["standing"].strip(), name
    if name in FEATURE_TERMS:
        assert "features" in term and isinstance(term["features"], dict), name


def _assert_shape(result):
    assert set(result) == {"phase", "terms", "leads"}
    assert isinstance(result["phase"], float)
    assert set(result["terms"]) == TERM_NAMES
    for name, term in result["terms"].items():
        _assert_term_shape(name, term)
    assert isinstance(result["leads"], list)


def _assert_no_eval_numbers(standing):
    """Contract invariant 5: standings never quote an engine number (cp / win% /
    `%` / signed eval like +7). Square/file coords and small counts ARE allowed."""
    assert "%" not in standing, standing
    assert "cp" not in standing.lower(), standing
    assert not re.search(r"[+-]\d", standing), standing


def _assert_standings_have_no_eval_numbers(result):
    for name, term in result["terms"].items():
        _assert_no_eval_numbers(term["standing"])


def _check_leads_law(result):
    """Contract invariant 6 / the `leads` definition, tie-tolerant."""
    terms, leads = result["terms"], result["leads"]
    assert isinstance(leads, list)
    assert len(leads) <= 2
    assert len(set(leads)) == len(leads)               # no duplicates
    assert all(name in terms for name in leads)        # valid term keys
    assert all(abs(terms[name]["cp"]) >= 25 for name in leads)  # each salient
    absvals = [abs(terms[name]["cp"]) for name in leads]
    assert absvals == sorted(absvals, reverse=True)    # abs-desc order
    salient = [n for n in terms if abs(terms[n]["cp"]) >= 25]
    assert len(leads) == min(2, len(salient))          # count is right
    if leads:                                          # excluded salient <= included
        smallest_in = min(abs(terms[n]["cp"]) for n in leads)
        for n in salient:
            if n not in leads:
                assert abs(terms[n]["cp"]) <= smallest_in


# ==========================================================================
# Invariant 1 — determinism
# ==========================================================================

@pytest.mark.parametrize("fen", LEADS_LAW_CASES)
def test_determinism_byte_identical(fen):
    assert analyze_positional(Board(fen)) == analyze_positional(Board(fen))


# ==========================================================================
# Invariant 2 — shape
# ==========================================================================

@pytest.mark.parametrize("fen", LEADS_LAW_CASES)
def test_shape_is_well_formed(fen):
    _assert_shape(analyze_positional(Board(fen)))


# ==========================================================================
# Invariant 3 — start position symmetry
# ==========================================================================

def test_start_position_is_perfectly_balanced():
    r = analyze_positional(Board(START))
    assert r["phase"] == 1.0
    assert r["leads"] == []
    for name, term in r["terms"].items():
        assert term["cp"] == 0, name
    assert r["terms"]["material"]["standing"] == "material is even"


# ==========================================================================
# Invariant 4 — colour-mirror antisymmetry
# ==========================================================================

@pytest.mark.parametrize("fen", MIRROR_CASES)
def test_colour_mirror_antisymmetry(fen):
    a = analyze_positional(Board(fen))
    b = analyze_positional(Board(_mirror_fen(fen)))
    assert b["phase"] == a["phase"]                    # phase unchanged
    for name in TERM_NAMES:
        assert b["terms"][name]["cp"] == -a["terms"][name]["cp"], name
    # abs(cp) is invariant under negation, so leads (names + order) match.
    assert b["leads"] == a["leads"]


def test_self_mirror_kings_only_is_all_zero():
    # e1 <-> e8 makes KINGS_ONLY its own colour-mirror, so every term must be 0.
    r = analyze_positional(Board(KINGS_ONLY))
    for name, term in r["terms"].items():
        assert term["cp"] == 0, name
    assert r["leads"] == []


def test_material_standing_swaps_under_mirror():
    # Standings swap White<->Black under the colour-mirror (invariant 4 note).
    a = analyze_positional(Board(PASSER_EP))["terms"]["material"]["standing"]
    b = analyze_positional(Board(_mirror_fen(PASSER_EP)))["terms"]["material"]["standing"]
    assert a == "White is up a pawn"
    assert b == "Black is up a pawn"


def test_king_danger_standing_swaps_under_mirror():
    a = analyze_positional(Board(KING_DANGER))["terms"]["king_safety"]["standing"]
    b = analyze_positional(Board(_mirror_fen(KING_DANGER)))["terms"]["king_safety"]["standing"]
    assert "White's king is in danger" in a
    assert "Black's king is in danger" in b


# ==========================================================================
# Side-to-move independence of cp (sign rule is absolute / White-POV)
# ==========================================================================

@pytest.mark.parametrize("fen", [WHITE_EXTRA_N, ACTIVITY])
def test_cp_is_independent_of_side_to_move(fen):
    # Flipping ONLY whose move it is leaves every term's cp unchanged; the sign
    # rule is White-POV and stm-independent. (Only the activity *standing* may
    # nudge toward the mover's worst piece — cp does not move.)
    a = analyze_positional(Board(fen))
    b = analyze_positional(Board(_flip_stm(fen)))
    assert a["phase"] == b["phase"]
    for name in TERM_NAMES:
        assert a["terms"][name]["cp"] == b["terms"][name]["cp"], name


# ==========================================================================
# Invariant 5 — no engine numbers in standings
# ==========================================================================

@pytest.mark.parametrize("fen", LEADS_LAW_CASES)
def test_standings_have_no_eval_numbers(fen):
    _assert_standings_have_no_eval_numbers(analyze_positional(Board(fen)))


# ==========================================================================
# Invariant 6 — the `leads` law
# ==========================================================================

@pytest.mark.parametrize("fen", LEADS_LAW_CASES)
def test_leads_law_holds(fen):
    _check_leads_law(analyze_positional(Board(fen)))


# ==========================================================================
# Invariant 7 — no crash on non-standard boards
# ==========================================================================

@pytest.mark.parametrize("fen", [KINGS_ONLY, DOUBLED, SPARSE,
                                 "4k3/8/8/8/8/8/8/N3K3 w - - 0 1",
                                 "8/6k1/4P3/8/8/8/5K2/8 w - - 0 1"])
def test_no_crash_on_nonstandard_boards(fen):
    _assert_shape(analyze_positional(Board(fen)))


def test_phase_never_exceeds_start_and_matches_formula():
    for fen in LEADS_LAW_CASES:
        r = analyze_positional(Board(fen))
        assert r["phase"] == _expected_phase(fen), fen
        assert 0.0 <= r["phase"] <= 1.0


# ==========================================================================
# Sign rule per term (positive favours White, White-POV, stm-independent)
# ==========================================================================

def test_material_sign_and_even():
    assert analyze_positional(Board(WHITE_EXTRA_N))["terms"]["material"]["cp"] > 0
    assert analyze_positional(Board(_mirror_fen(WHITE_EXTRA_N)))["terms"]["material"]["cp"] < 0
    assert analyze_positional(Board(START))["terms"]["material"]["cp"] == 0


def test_material_cp_zero_only_for_symmetric_positions():
    # material cp is TAPERED PESTO, not the naive count. cp == 0 is guaranteed
    # only for a fully symmetric position (start / a self-mirror) — NOT merely a
    # count-even one.
    assert analyze_positional(Board(START))["terms"]["material"]["cp"] == 0
    assert analyze_positional(Board(KINGS_ONLY))["terms"]["material"]["cp"] == 0


def test_material_count_even_but_pesto_unequal_bishop_vs_knight():
    # SPARSE is knight-vs-bishop: piece-count EVEN (standing "material is even")
    # yet PeSTO values the bishop higher, so a small nonzero cp is CORRECT here.
    # Per the clarified contract we do NOT assert cp == 0 from a count-even board.
    r = analyze_positional(Board(SPARSE))
    assert r["terms"]["material"]["standing"] == "material is even"


def test_material_standing_equals_response_material():
    # Contract: material.standing IS lucena.mcp.response.material(board)["standing"].
    for fen in [START, WHITE_EXTRA_N, PASSER_EP, IQP, SPARSE]:
        got = analyze_positional(Board(fen))["terms"]["material"]["standing"]
        assert got == R.material(Board(fen))["standing"], fen


def test_king_safety_sign_against_the_attacked_side():
    # Anchored: an attack on White's king => cp < 0 (signed against White).
    assert analyze_positional(Board(KING_DANGER))["terms"]["king_safety"]["cp"] < 0
    # Its mirror attacks Black's king => cp > 0.
    assert analyze_positional(Board(_mirror_fen(KING_DANGER)))["terms"]["king_safety"]["cp"] > 0


def test_pawns_sign_passer_favours_owner():
    # A White passed pawn scores FOR White => pawns cp > 0.
    assert analyze_positional(Board(PASSER_EP))["terms"]["pawns"]["cp"] > 0
    # Mirror: Black owns the passer => pawns cp < 0.
    assert analyze_positional(Board(_mirror_fen(PASSER_EP)))["terms"]["pawns"]["cp"] < 0


def test_center_sign_favours_the_controlling_side():
    # Only White occupies/controls the centre => center cp > 0.
    assert analyze_positional(Board(CENTER_WHITE))["terms"]["center"]["cp"] > 0
    assert analyze_positional(Board(_mirror_fen(CENTER_WHITE)))["terms"]["center"]["cp"] < 0


def test_activity_sign_favours_the_more_active_side():
    # White knight centralised on d5, Black knight buried on b8 => White the more
    # active (PeSTO placement + mobility) => activity cp > 0.
    assert analyze_positional(Board(ACTIVITY))["terms"]["activity"]["cp"] > 0
    assert analyze_positional(Board(_mirror_fen(ACTIVITY)))["terms"]["activity"]["cp"] < 0


# ==========================================================================
# Anchored examples (contract "Anchored examples" table)
# ==========================================================================

def test_anchor_start_position():
    r = analyze_positional(Board(START))
    assert r["phase"] == 1.0
    assert r["leads"] == []
    assert all(t["cp"] == 0 for t in r["terms"].values())
    assert r["terms"]["material"]["standing"] == "material is even"


def test_anchor_passer_endgame():
    r = analyze_positional(Board(PASSER_EP))
    assert r["phase"] == 0.0
    assert r["terms"]["material"]["standing"] == "White is up a pawn"
    assert r["terms"]["pawns"]["features"]["white"]["passed"] == ["e6"]
    # standing flags a passed pawn (its square may appear) and carries no engine
    # number (invariant 5).
    standing = r["terms"]["pawns"]["standing"]
    assert "passed" in standing.lower()
    _assert_no_eval_numbers(standing)
    assert "material" in r["leads"] and "pawns" in r["leads"]


def test_anchor_iqp_isolated_d5():
    r = analyze_positional(Board(IQP))
    assert "d5" in r["terms"]["pawns"]["features"]["black"]["isolated"]


def test_anchor_king_danger():
    r = analyze_positional(Board(KING_DANGER))
    ks = r["terms"]["king_safety"]
    assert "White's king is in danger" in ks["standing"]
    assert ks["cp"] < 0
    assert "king_safety" in r["leads"]
    assert "h" in ks["features"]["white"]["open_files_nearby"]
    # Qh4 and Ng4 bear on White's king zone -> non-empty attacker list.
    assert ks["features"]["white"]["zone_attackers"]


# ==========================================================================
# Per-term features schemas
# ==========================================================================

@pytest.mark.parametrize("fen", [KINGS_ONLY, SPARSE, WHITE_EXTRA_N, KING_DANGER])
def test_king_safety_features_always_has_both_colours(fen):
    # Both kings always exist (kingless is unconstructible), so both keys present.
    feats = analyze_positional(Board(fen))["terms"]["king_safety"]["features"]
    assert "white" in feats and "black" in feats


def test_king_safety_features_schema():
    r = analyze_positional(Board(KING_DANGER))
    feats = r["terms"]["king_safety"]["features"]
    for color in ("white", "black"):     # both kings exist in this position
        assert color in feats
        f = feats[color]
        assert set(f) >= {"king", "zone_attackers", "attack_units",
                          "shield_pawns", "open_files_nearby"}
        assert isinstance(f["zone_attackers"], list)
        assert all(isinstance(s, str) for s in f["zone_attackers"])
        assert _is_int(f["attack_units"])
        assert _is_int(f["shield_pawns"]) and 0 <= f["shield_pawns"] <= 3
        assert isinstance(f["open_files_nearby"], list)
        assert all(isinstance(s, str) and s in "abcdefgh"
                   for s in f["open_files_nearby"])


def test_pawns_features_schema():
    r = analyze_positional(Board(IQP))
    feats = r["terms"]["pawns"]["features"]
    for color in ("white", "black"):
        f = feats[color]
        assert set(f) >= {"doubled_files", "isolated", "passed", "islands"}
        assert all(_SQUARE_RE.match(sq) for sq in f["isolated"])
        assert all(_SQUARE_RE.match(sq) for sq in f["passed"])
        assert all(isinstance(x, str) and x in "abcdefgh" for x in f["doubled_files"])
        assert _is_int(f["islands"]) and f["islands"] >= 1   # both sides have pawns


def test_pawns_islands_zero_with_no_pawns():
    # Contract: islands is "an int >= 1 (or 0 with no pawns)".
    r = analyze_positional(Board(WHITE_EXTRA_N))   # no pawns for either side
    feats = r["terms"]["pawns"]["features"]
    for color in ("white", "black"):
        assert feats[color]["islands"] == 0
        assert feats[color]["isolated"] == []
        assert feats[color]["passed"] == []
        assert feats[color]["doubled_files"] == []


def test_pawns_doubled_files_reported():
    r = analyze_positional(Board(DOUBLED))          # White pawns doubled on d-file
    assert "d" in r["terms"]["pawns"]["features"]["white"]["doubled_files"]


def test_center_features_schema():
    r = analyze_positional(Board(START))
    feats = r["terms"]["center"]["features"]
    for sq in ("d4", "e4", "d5", "e5"):
        assert sq in feats
        f = feats[sq]
        assert set(f) >= {"white_attackers", "black_attackers"}
        assert _is_int(f["white_attackers"])
        assert _is_int(f["black_attackers"])


def test_activity_worst_piece_schema_when_present():
    r = analyze_positional(Board(ACTIVITY))
    feats = r["terms"]["activity"]["features"]
    assert isinstance(feats, dict)
    for key in ("worst_piece_white", "worst_piece_black"):
        if key in feats:                            # optional per contract
            wp = feats[key]
            assert set(wp) >= {"square", "piece"}
            assert _SQUARE_RE.match(wp["square"])
            # kings and pawns are out of activity's scope, so a worst piece is a
            # minor or major only.
            assert wp["piece"] in "NBRQ"


# ==========================================================================
# Board-layer behaviour: a kingless FEN is unconstructible (contract §API /
# invariant 7 — kingless is out of scope, not a degrade-gracefully case)
# ==========================================================================

def test_kingless_side_is_rejected_at_board_construction():
    # The contract now states a kingless FEN is rejected at Board construction,
    # so analyze_positional never receives one. This documents that Board-layer
    # guarantee (it is not a claim about the module itself).
    with pytest.raises(Exception):
        Board("8/8/8/8/8/8/8/4K3 w - - 0 1")
