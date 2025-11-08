use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
    tensor::activation::softmax,
};
use poulet_chess::Game;

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "512")]
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(768, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            linear3: LinearConfig::new(self.hidden_size, 4096).init(device),
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(state);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear3.forward(x);
        x
    }
}

fn idx_to_move(i: usize) -> (u8, u8, u8, u8) {
    let src = i / 64;
    let dst = i % 64;

    (
        (src % 8) as u8,
        (src / 8) as u8,
        (dst % 8) as u8,
        (dst / 8) as u8,
    )
}

pub fn predict_move<B: Backend>(
    games: Vec<&mut Game>,
    logits: Tensor<B, 2>,
    device: &Device<B>,
) -> Vec<(f32, u8, u8, u8, u8)> {
    let batch_size = logits.shape()[0];
    let mut mask: Vec<f32> = Vec::with_capacity(batch_size * 4096);

    for game in games {
        for i in 0..4096 {
            let (src_x, src_y, dst_x, dst_y) = idx_to_move(i);
            mask.push(
                if game
                    .board
                    .get_square(src_x, src_y)
                    .map(|p| game.can_move(p.color))
                    .unwrap_or(false)
                    && game.safe_move(src_x as u8, src_y as u8, dst_x as u8, dst_y as u8)
                {
                    0.0
                } else {
                    -f32::INFINITY
                },
            );
        }
    }

    let mask_tensor_data = TensorData::new(mask, [batch_size, 4096]);
    let mask_tensor: Tensor<B, 2> = Tensor::from_floats(mask_tensor_data, device);

    let x = logits.add(mask_tensor);
    let x = softmax(x, 1);
    let (x, indices) = x.sort_descending_with_indices(1);

    let scores_col: Tensor<B, 2, Float> = x.slice([0..batch_size, 0..1]);
    let scores_col: Tensor<B, 1, Float> = scores_col.reshape([batch_size]);
    let scores_col = scores_col.to_data();
    let scores_col: Vec<f32> = scores_col.into_vec().unwrap();

    let indices_col: Tensor<B, 2, Int> = indices.slice([0..batch_size, 0..1]);
    let indices_col: Tensor<B, 1, Int> = indices_col.reshape([batch_size]);
    let indices_col = indices_col.to_data();
    let indices_col: Vec<i32> = indices_col.into_vec().unwrap();

    scores_col
        .iter()
        .zip(indices_col)
        .map(|(score, idx)| {
            let (src_x, src_y, dst_x, dst_y) = idx_to_move(idx as usize);
            (*score, src_x, src_y, dst_x, dst_y)
        })
        .collect()
}
