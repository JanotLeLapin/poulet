pub fn game_loop(net_a: &mut poulet::ai::Network, net_b: &mut poulet::ai::Network) -> (f64, f64) {
    let mut scores = (0.0, 0.0);
    let mut game = poulet::chess::Game::default();
    while let Ok(Some(m)) = poulet::next_move(
        if game.turn == poulet::chess::Color::White {
            net_a
        } else {
            net_b
        },
        &mut game,
    ) {
        println!("{:?}", m);
        game.do_move(m.src.0, m.src.1, m.dst.0, m.dst.1);
    }

    if game.is_check(game.turn) {
        println!("checkmate!");
    } else {
        println!("stalemate!");
    }

    scores
}

fn main() {
    let mut a = poulet::new_chess_network().unwrap();
    let mut b = poulet::new_chess_network().unwrap();

    game_loop(&mut a, &mut b);
}
