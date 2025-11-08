use crate::model::predict_move;

use poulet_ai_common::encode_board;
use poulet_chess::Game;

use burn::{Tensor, backend::Wgpu, tensor::TensorData};

mod model;

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
