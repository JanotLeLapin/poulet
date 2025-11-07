use std::sync::Arc;

use poulet_nn::{State, encode_board};
use rand::distr::Distribution;

struct LayerParams {
    input_size: u32,
    output_size: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = State::new().await;

    let rng = rand::rng();
    let distr = rand::distr::Uniform::new(-0.0884, 0.0884)?;

    let weights_data: Vec<f32> = distr.sample_iter(rng).take(768 * 512).collect();
    let biases_data: Vec<f32> = (0..512).map(|_| 0.0 as f32).collect();

    let hidden = poulet_nn::DenseLayer::new(
        Arc::new(state.device),
        Arc::new(state.queue),
        Arc::new(state.pipeline),
        &weights_data,
        &biases_data,
        768,
        512,
    );

    let mut input_data: Vec<f32> = (0..768).map(|_| 0.0 as f32).collect();
    encode_board(&poulet_chess::Game::default().board, &mut input_data);

    let out = hidden.forward(&input_data).await.unwrap();
    println!("{out:?}");

    Ok(())
}
