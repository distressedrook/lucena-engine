//! Static exchange evaluation over cozy-chess bitboards.
//!
//! Swap algorithm with least-valuable-attacker ordering and x-ray/battery
//! re-inclusion (slider attacks are recomputed against the shrinking
//! occupancy, so a rook behind a rook joins the exchange automatically).
//! Promotions are valued as the moving pawn (documented simplification:
//! good enough for fact salience; not a search heuristic).

use cozy_chess::{Board, Color, Piece, Square, BitBoard};

pub const VALUES: [i32; 6] = [100, 300, 300, 500, 900, 20_000]; // P N B R Q K

#[inline]
pub fn value(piece: Piece) -> i32 {
    VALUES[piece as usize]
}

/// All pieces (of both colors) attacking `sq`, given occupancy `occ`.
/// Sliders are computed against `occ`, which is what makes x-rays work
/// as the exchange peels pieces off.
pub fn attackers_to(board: &Board, sq: Square, occ: BitBoard) -> BitBoard {
    let bishops = board.pieces(Piece::Bishop) | board.pieces(Piece::Queen);
    let rooks = board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    let mut atk = BitBoard::EMPTY;
    // A white pawn standing on `sq` attacks exactly the squares from which
    // black pawns attack `sq` (mirror trick), and vice versa.
    atk |= cozy_chess::get_pawn_attacks(sq, Color::White)
        & board.pieces(Piece::Pawn) & board.colors(Color::Black);
    atk |= cozy_chess::get_pawn_attacks(sq, Color::Black)
        & board.pieces(Piece::Pawn) & board.colors(Color::White);
    atk |= cozy_chess::get_knight_moves(sq) & board.pieces(Piece::Knight);
    atk |= cozy_chess::get_king_moves(sq) & board.pieces(Piece::King);
    atk |= cozy_chess::get_bishop_moves(sq, occ) & bishops;
    atk |= cozy_chess::get_rook_moves(sq, occ) & rooks;
    atk & occ
}

fn least_valuable(board: &Board, set: BitBoard) -> Option<(Square, Piece)> {
    let mut best: Option<(Square, Piece)> = None;
    for sq in set {
        let p = board.piece_on(sq).expect("attacker square must be occupied");
        match best {
            Some((_, bp)) if value(p) >= value(bp) => {}
            _ => best = Some((sq, p)),
        }
    }
    best
}

/// SEE for moving the piece on `from` to `to`. Positive = the exchange wins
/// material for the side to move (centipawns). Handles en passant, x-rays,
/// and the king-can't-be-recaptured-into-check guard.
pub fn see(board: &Board, from: Square, to: Square) -> i32 {
    let mover = board
        .piece_on(from)
        .expect("SEE: no piece on from-square");
    let us = board.color_on(from).expect("SEE: from-square has no color");

    let mut occ = board.occupied();
    let mut gain = [0i32; 32];
    let mut d = 0usize;

    // Initial capture value; en passant captures a pawn not on `to`.
    let captured_value = match board.piece_on(to) {
        Some(p) => value(p),
        None => {
            if mover == Piece::Pawn && Some(to) == board.en_passant().map(|f| {
                // en_passant() gives the file; the captured pawn's "arrival"
                // square for the capturing side:
                let rank = if us == Color::White { cozy_chess::Rank::Sixth } else { cozy_chess::Rank::Third };
                Square::new(f, rank)
            }) {
                // remove the actual captured pawn from occupancy
                let cap_rank = if us == Color::White { cozy_chess::Rank::Fifth } else { cozy_chess::Rank::Fourth };
                let cap_sq = Square::new(to.file(), cap_rank);
                occ ^= cap_sq.bitboard();
                value(Piece::Pawn)
            } else {
                0
            }
        }
    };

    gain[0] = captured_value;
    let mut attacker_value = value(mover);
    occ ^= from.bitboard();
    let mut side = !us;

    loop {
        let atts = attackers_to(board, to, occ) & board.colors(side);
        let Some((sq, piece)) = least_valuable(board, atts) else { break };
        // A king may only take if the opponent has no remaining attacker
        // (it could not legally step into a defended square).
        if piece == Piece::King {
            let opp = attackers_to(board, to, occ ^ sq.bitboard()) & board.colors(!side);
            if !(opp & !sq.bitboard()).is_empty() {
                break;
            }
        }
        d += 1;
        gain[d] = attacker_value - gain[d - 1];
        // Pruning: if neither continuing nor stopping can help, stop.
        if std::cmp::max(-gain[d - 1], gain[d]) < 0 {
            break;
        }
        attacker_value = value(piece);
        occ ^= sq.bitboard();
        side = !side;
    }

    while d > 0 {
        gain[d - 1] = -std::cmp::max(-gain[d - 1], gain[d]);
        d -= 1;
    }
    gain[0]
}
