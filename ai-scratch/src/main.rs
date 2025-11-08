use poulet_ai_common::encode_board;
use poulet_ai_scratch::{Player, State};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = State::new().await;

    let player = Player::new(&state)?;

    let input_data = encode_board(&poulet_chess::Game::default().board);

    let res = player.forward(&input_data).await?;
    println!("{res:?}");

    Ok(())
}
