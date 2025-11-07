use poulet_nn::{State, encode_board};
use rand::distr::Distribution;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = State::new().await;

    let rng = rand::rng();
    let distr = rand::distr::Uniform::new(-0.0884, 0.0884)?;

    let weights_data_a: Vec<f32> = distr.sample_iter(rng.clone()).take(768 * 512).collect();
    let biases_data_a: Vec<f32> = (0..512).map(|_| 0.0 as f32).collect();

    let weights_data_b: Vec<f32> = distr.sample_iter(rng).take(512 * 512).collect();
    let biases_data_b: Vec<f32> = (0..512).map(|_| 0.0 as f32).collect();

    let hidden_a = poulet_nn::DenseLayer::new(
        state.device.clone(),
        state.queue.clone(),
        state.pipeline.clone(),
        &weights_data_a,
        &biases_data_a,
        768,
        512,
    );

    let hidden_b = poulet_nn::DenseLayer::new(
        state.device.clone(),
        state.queue.clone(),
        state.pipeline.clone(),
        &weights_data_b,
        &biases_data_b,
        512,
        512,
    );

    let mut input_data: Vec<f32> = (0..768).map(|_| 0.0 as f32).collect();
    encode_board(&poulet_chess::Game::default().board, &mut input_data);

    let out = hidden_a.forward(&input_data).await.unwrap();
    println!("{out:?}");

    let out = hidden_b.forward(&out).await.unwrap();
    println!("{out:?}");

    Ok(())
}
