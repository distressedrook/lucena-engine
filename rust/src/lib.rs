//! lucena-board: the product's board core (MIT-licensed foundations only).
//! cozy-chess supplies bitboards/legality; SAN and SEE are ours.

mod see;

use cozy_chess::{Board, Color, Move, Piece, Square};
use cozy_chess::util::{display_uci_move, parse_uci_move};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::str::FromStr;

fn parse_board(fen: &str) -> PyResult<Board> {
    Board::from_str(fen).map_err(|e| PyValueError::new_err(format!("bad FEN: {e:?}")))
}

fn parse_square(s: &str) -> PyResult<Square> {
    Square::from_str(s).map_err(|_| PyValueError::new_err(format!("bad square: {s}")))
}

fn parse_color(s: &str) -> PyResult<Color> {
    match s {
        "white" | "w" => Ok(Color::White),
        "black" | "b" => Ok(Color::Black),
        _ => Err(PyValueError::new_err(format!("bad color: {s} (use white|black)"))),
    }
}

fn legal_moves_vec(board: &Board) -> Vec<Move> {
    let mut out = Vec::with_capacity(64);
    board.generate_moves(|mvs| {
        out.extend(mvs);
        false
    });
    out
}

fn piece_char(p: Piece) -> char {
    match p {
        Piece::Pawn => 'P',
        Piece::Knight => 'N',
        Piece::Bishop => 'B',
        Piece::Rook => 'R',
        Piece::Queen => 'Q',
        Piece::King => 'K',
    }
}

fn file_char(f: cozy_chess::File) -> char {
    (b'a' + f as u8) as char
}

fn rank_char(r: cozy_chess::Rank) -> char {
    (b'1' + r as u8) as char
}

fn is_castling(board: &Board, mv: Move) -> bool {
    // cozy-chess encodes castling as king-captures-own-rook.
    board.piece_on(mv.from) == Some(Piece::King)
        && board.color_on(mv.to) == board.color_on(mv.from)
}

/// SAN for a legal cozy-chess move (native castling representation), without
/// the check/mate suffix.
fn san_body(board: &Board, mv: Move) -> String {
    if is_castling(board, mv) {
        return if mv.to.file() > mv.from.file() { "O-O".into() } else { "O-O-O".into() };
    }
    let piece = board.piece_on(mv.from).expect("legal move has a mover");
    let to_s = mv.to.to_string();
    let is_capture = board.piece_on(mv.to).is_some()
        || (piece == Piece::Pawn && mv.from.file() != mv.to.file());

    if piece == Piece::Pawn {
        let mut s = String::new();
        if is_capture {
            s.push(file_char(mv.from.file()));
            s.push('x');
        }
        s.push_str(&to_s);
        if let Some(promo) = mv.promotion {
            s.push('=');
            s.push(piece_char(promo));
        }
        return s;
    }

    // Disambiguation among same-piece moves to the same square.
    let mut same_file = false;
    let mut same_rank = false;
    let mut ambiguous = false;
    for other in legal_moves_vec(board) {
        if other.to == mv.to
            && other.from != mv.from
            && board.piece_on(other.from) == Some(piece)
            && !is_castling(board, other)
        {
            ambiguous = true;
            if other.from.file() == mv.from.file() {
                same_file = true;
            }
            if other.from.rank() == mv.from.rank() {
                same_rank = true;
            }
        }
    }
    let mut s = String::new();
    s.push(piece_char(piece));
    if ambiguous {
        if !same_file {
            s.push(file_char(mv.from.file()));
        } else if !same_rank {
            s.push(rank_char(mv.from.rank()));
        } else {
            s.push(file_char(mv.from.file()));
            s.push(rank_char(mv.from.rank()));
        }
    }
    if is_capture {
        s.push('x');
    }
    s.push_str(&to_s);
    s
}

fn san_full(board: &Board, mv: Move) -> String {
    let mut s = san_body(board, mv);
    let mut after = board.clone();
    after.play_unchecked(mv);
    if !after.checkers().is_empty() {
        s.push(if legal_moves_vec(&after).is_empty() { '#' } else { '+' });
    }
    s
}

fn normalize_san(s: &str) -> String {
    s.trim()
        .trim_end_matches(['+', '#', '!', '?'])
        .replace('0', "O") // 0-0 → O-O
        .replace("e.p.", "")
        .trim()
        .to_string()
}

// ---------------- Python surface ----------------

/// [(square, piece, color)] for every piece, e.g. ("e5", "N", "black").
#[pyfunction]
fn piece_list(fen: &str) -> PyResult<Vec<(String, String, String)>> {
    let board = parse_board(fen)?;
    let mut out = Vec::new();
    for sq in board.occupied() {
        let p = board.piece_on(sq).unwrap();
        let c = board.color_on(sq).unwrap();
        out.push((
            sq.to_string(),
            piece_char(p).to_string(),
            if c == Color::White { "white".into() } else { "black".to_string() },
        ));
    }
    Ok(out)
}

/// Legal moves in standard UCI ("e1g1" for castling).
#[pyfunction]
fn legal_moves(fen: &str) -> PyResult<Vec<String>> {
    let board = parse_board(fen)?;
    Ok(legal_moves_vec(&board)
        .into_iter()
        .map(|m| display_uci_move(&board, m).to_string())
        .collect())
}

/// Squares of `color` pieces attacking `sq` (full occupancy).
#[pyfunction]
fn attackers(fen: &str, sq: &str, color: &str) -> PyResult<Vec<String>> {
    let board = parse_board(fen)?;
    let target = parse_square(sq)?;
    let c = parse_color(color)?;
    let atts = see::attackers_to(&board, target, board.occupied()) & board.colors(c);
    Ok(atts.into_iter().map(|s| s.to_string()).collect())
}

/// New FEN after playing a standard-UCI move.
#[pyfunction]
fn apply_uci(fen: &str, uci: &str) -> PyResult<String> {
    let mut board = parse_board(fen)?;
    let mv = parse_uci_move(&board, uci)
        .map_err(|e| PyValueError::new_err(format!("bad move {uci}: {e:?}")))?;
    board
        .try_play(mv)
        .map_err(|e| PyValueError::new_err(format!("illegal move {uci}: {e:?}")))?;
    Ok(format!("{board}"))
}

#[pyfunction]
fn uci_to_san(fen: &str, uci: &str) -> PyResult<String> {
    let board = parse_board(fen)?;
    let mv = parse_uci_move(&board, uci)
        .map_err(|e| PyValueError::new_err(format!("bad move {uci}: {e:?}")))?;
    if !board.is_legal(mv) {
        return Err(PyValueError::new_err(format!("illegal move: {uci}")));
    }
    Ok(san_full(&board, mv))
}

#[pyfunction]
fn san_to_uci(fen: &str, san: &str) -> PyResult<String> {
    let board = parse_board(fen)?;
    let want = normalize_san(san);
    for mv in legal_moves_vec(&board) {
        if normalize_san(&san_body(&board, mv)) == want {
            return Ok(display_uci_move(&board, mv).to_string());
        }
    }
    Err(PyValueError::new_err(format!("no legal move matches SAN '{san}'")))
}

/// Static exchange evaluation of a standard-UCI move, in centipawns
/// (positive = good for the mover).
#[pyfunction]
fn see_move(fen: &str, uci: &str) -> PyResult<i32> {
    let board = parse_board(fen)?;
    let mv = parse_uci_move(&board, uci)
        .map_err(|e| PyValueError::new_err(format!("bad move {uci}: {e:?}")))?;
    if !board.is_legal(mv) {
        return Err(PyValueError::new_err(format!("illegal move: {uci}")));
    }
    if is_castling(&board, mv) {
        return Ok(0);
    }
    Ok(see::see(&board, mv.from, mv.to))
}

/// FEN with the side to move flipped (null move), or error if in check.
/// Used by the null-move threat probe.
#[pyfunction]
fn null_move_fen(fen: &str) -> PyResult<String> {
    let board = parse_board(fen)?;
    match board.null_move() {
        Some(b) => Ok(format!("{b}")),
        None => Err(PyValueError::new_err("null move illegal (in check)")),
    }
}

#[pyfunction]
fn is_check(fen: &str) -> PyResult<bool> {
    Ok(!parse_board(fen)?.checkers().is_empty())
}

#[pyfunction]
fn side_to_move(fen: &str) -> PyResult<String> {
    Ok(match parse_board(fen)?.side_to_move() {
        Color::White => "white".into(),
        Color::Black => "black".to_string(),
    })
}

#[pymodule]
fn lucena_board(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(piece_list, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(attackers, m)?)?;
    m.add_function(wrap_pyfunction!(apply_uci, m)?)?;
    m.add_function(wrap_pyfunction!(uci_to_san, m)?)?;
    m.add_function(wrap_pyfunction!(san_to_uci, m)?)?;
    m.add_function(wrap_pyfunction!(see_move, m)?)?;
    m.add_function(wrap_pyfunction!(null_move_fen, m)?)?;
    m.add_function(wrap_pyfunction!(is_check, m)?)?;
    m.add_function(wrap_pyfunction!(side_to_move, m)?)?;
    Ok(())
}
