from lucena_engine import Board

SCHOLARS = "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3"
MIDGAME = "r2q1rk1/pp3ppp/2p2n2/4n3/8/2NB1Q2/PPP2PPP/R1B2RK1 w - - 0 14"


def test_piece_list_startpos():
    pieces = Board().piece_list()
    assert len(pieces) == 32
    assert any(p.square == "e1" and p.piece == "K" and p.color == "white" for p in pieces)
    assert any(p.square == "d8" and p.piece == "Q" and p.color == "black" for p in pieces)


def test_attackers_validated_positions():
    # Spike-validated: Bc4 and Qh5 both attack f7 in the Scholar's-mate setup.
    b = Board(SCHOLARS)
    assert sorted(b.attackers("f7", "white")) == ["c4", "h5"]
    assert b.attackers("e5", "white") == ["h5"]
    # Nothing attacks e5 in the midgame position (validated in the spike).
    assert Board(MIDGAME).attackers("e5", "white") == []
    assert Board(MIDGAME).attackers("e5", "black") == []


def test_defenders():
    # Black knight e5 in MIDGAME is undefended.
    assert Board(MIDGAME).defenders("e5") == []


def test_legal_move_count_startpos():
    assert len(Board().legal_moves()) == 20


def test_san_castling_and_roundtrip():
    # After 1.e4 e5 2.Nf3 Nc6 3.Bc4 Bc5, O-O is legal for White.
    b = Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]:
        b = b.apply(san)
    assert b.uci("O-O") == "e1g1"
    assert b.san("e1g1") == "O-O"
    # Full round-trip over every legal move in a few positions.
    for fen in [Board().fen, SCHOLARS, MIDGAME, b.fen]:
        pos = Board(fen)
        for uci in pos.legal_moves():
            assert pos.uci(pos.san(uci)) == uci


def test_san_disambiguation_file():
    # Rooks a1 + f1, both can reach d1 → Rad1 / Rfd1.
    fen = "4k3/8/8/8/8/8/8/R4RK1 w - - 0 1"
    b = Board(fen)
    assert b.san("a1d1") == "Rad1"
    assert b.san("f1d1") == "Rfd1"
    assert b.uci("Rad1") == "a1d1"


def test_san_promotion_and_mate():
    # Fool's mate: 1.f3 e5 2.g4 Qh4#
    b = Board().apply("f3").apply("e5").apply("g4")
    assert b.san("d8h4") == "Qh4#"
    # Promotion with check (king on h8 so the pawn itself gives no check).
    fen = "7k/4P3/8/8/8/8/8/4K3 w - - 0 1"
    assert Board(fen).san("e7e8q") == "e8=Q+"
    assert Board(fen).uci("e8=Q+") == "e7e8q"


def test_null_move():
    b = Board(MIDGAME)
    assert b.side_to_move == "white"
    assert b.null_move().side_to_move == "black"


def test_bad_inputs_raise():
    import pytest

    with pytest.raises(ValueError):
        Board("not a fen")
    with pytest.raises(ValueError):
        Board().san("e2e5")  # illegal
    with pytest.raises(ValueError):
        Board().uci("Qh5")  # no legal match
