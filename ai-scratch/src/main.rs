use poulet_ai_common::encode_board;
use poulet_ai_scratch::{Player, State, predict_move};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = State::new().await;

    let player = Player::new(&state)?;

    let mut game = poulet_chess::Game::default();

    let input_data = encode_board(&game.board);

    let logits = player.forward(&input_data).await?;

    let next_move = predict_move(&mut game, logits);

    println!("{next_move:?}");

    Ok(())
}
