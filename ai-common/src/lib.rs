use poulet_chess::{Board, Color, Piece, PieceType};

pub fn encode_board(board: &Board) -> Vec<f32> {
    let mut neurons = Vec::with_capacity(768);

    let piece_iter = vec![Color::White, Color::Black].into_iter().flat_map(|c| {
        vec![
            PieceType::Pawn,
            PieceType::Bishop,
            PieceType::Knight,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ]
        .into_iter()
        .map(move |t| Piece::new(c, t))
    });

    for piece in piece_iter {
        for y in 0..8 {
            for x in 0..8 {
                neurons.push(if board.get_square(x as u8, y as u8) == Some(piece) {
                    1.0
                } else {
                    0.0
                });
            }
        }
    }

    neurons
}

pub fn decode_move(i: usize) -> (u8, u8, u8, u8) {
    let src = i / 64;
    let dst = i % 64;

    (
        (src % 8) as u8,
        (src / 8) as u8,
        (dst % 8) as u8,
        (dst / 8) as u8,
    )
}
