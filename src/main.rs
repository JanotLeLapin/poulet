static HEAT_MAP: [[u8; 8]; 8] = [
    [0, 0, 1, 2, 2, 1, 0, 0],
    [0, 1, 2, 3, 3, 2, 1, 0],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [2, 3, 4, 5, 5, 4, 3, 2],
    [2, 3, 4, 5, 5, 4, 3, 2],
    [1, 2, 3, 4, 4, 3, 2, 1],
    [0, 1, 2, 3, 3, 2, 1, 0],
    [0, 0, 1, 2, 2, 1, 0, 0],
];

fn game_loop(net_a: &mut poulet::ai::Network, net_b: &mut poulet::ai::Network) -> (f64, f64) {
    let networks = [net_a, net_b];
    let mut scores = [0.0, 0.0];
    let mut game = poulet::chess::Game::default();
    let mut turn = 0;
    loop {
        let net = &mut *networks[turn];
        let m = match poulet::next_move(net, &mut game) {
            Ok(Some(v)) => v,
            _ => break,
        };

        if game.moves.len() <= 40 {
            scores[turn] += (HEAT_MAP[m.dst.0 as usize][m.dst.1 as usize] as f64)
                / (if Some(poulet::chess::PieceType::Knight)
                    == game
                        .board
                        .get_square(m.src.0, m.src.1)
                        .map(|square| square.piece_type)
                {
                    18.0
                } else {
                    24.0
                });
        }

        if let Some(dst) = game.board.get_square(m.dst.0, m.dst.1) {
            scores[turn] += dst.piece_type.value() as f64;
        }

        println!("{:?}, scores: {:?}", m, scores);
        game.do_move(m.src.0, m.src.1, m.dst.0, m.dst.1);
        turn = 1 - turn;
    }

    if game.is_check(game.turn) {
        println!("checkmate!");
        scores[turn] -= 1000.0;
        scores[1 - turn] += 1000.0;
    } else {
        println!("stalemate!");
    }

    (scores[0], scores[1])
}

fn main() {
    let mut a = poulet::new_chess_network().unwrap();
    let mut b = poulet::new_chess_network().unwrap();

    let scores = game_loop(&mut a, &mut b);

    println!("final scores: {:?}", scores);
}
