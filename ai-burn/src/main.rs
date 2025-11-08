use burn::{Tensor, backend::Wgpu, tensor::TensorData};
use poulet_chess::{Board, Color, Game, Piece, PieceType};

use crate::model::predict_move;

mod model;

fn encode_board(board: &Board) -> Vec<f32> {
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

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = model::ModelConfig::new().init::<MyBackend>(&device);

    let mut neurons = Vec::with_capacity(768);
    let mut game = Game::default();
    neurons.extend(encode_board(&game.board));

    let data = TensorData::new(neurons, [1, 768]);
    let tensor: Tensor<MyBackend, 2> = Tensor::from_floats(data, &device);

    let logits = model.forward(tensor);

    println!("{model:#?}");

    let res = predict_move(vec![&mut game], logits, &device);
    println!("{res:?}");
}
