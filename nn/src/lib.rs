use poulet_chess::{Board, Color, Piece, PieceType};

pub fn encode_board(board: &Board, neurons: &mut Vec<f32>) {
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

    for (i, piece) in piece_iter.enumerate() {
        for x in 0..8 {
            for y in 0..8 {
                let neuron_idx = i * 64 + (y * 8 + x);
                neurons[neuron_idx] = if board.get_square(x as u8, y as u8) == Some(piece) {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }
}
