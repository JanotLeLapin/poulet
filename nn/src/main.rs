use poulet_nn::{State, encode_board};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = State::new().await;

    let player = poulet_nn::Player::new(&state)?;

    let mut input_data: Vec<f32> = (0..768).map(|_| 0.0 as f32).collect();
    encode_board(&poulet_chess::Game::default().board, &mut input_data);

    let res = player.forward(&input_data).await?;
    println!("{res:?}");

    Ok(())
}
