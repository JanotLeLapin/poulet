use std::sync::{Arc, Mutex};

use rand::seq::SliceRandom;
use rayon::prelude::*;

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
            scores[1 - turn] -= dst.piece_type.value() as f64;
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

fn make_matches(network_count: usize, max_matches: usize) -> Vec<(usize, usize)> {
    let mut rng = rand::rng();
    let mut matches = Vec::with_capacity(max_matches);
    let mut match_count = vec![0; network_count];

    let mut all_pairs: Vec<_> = (0..network_count)
        .flat_map(|i| (0..network_count).map(move |j| (i, j)))
        .collect();

    all_pairs.shuffle(&mut rng);

    for (i, j) in all_pairs {
        if match_count[i] < max_matches && match_count[j] < max_matches && i != j {
            matches.push((i, j));
            match_count[i] += 1;
            match_count[j] += 1;
        }
    }

    matches
}

fn run_matches(
    networks: &mut [Arc<Mutex<poulet::ai::Network>>],
    matches: &[(usize, usize)],
) -> Vec<f64> {
    let results: Vec<Mutex<f64>> = (0..networks.len()).map(|_| Mutex::new(0.0)).collect();
    matches.par_iter().for_each(|&(i, j)| {
        if i == j || i >= networks.len() || j >= networks.len() {
            return;
        }

        let (first, second) = if i < j { (i, j) } else { (j, i) };
        let mut net_a = networks[first].lock().unwrap();
        let mut net_b = networks[second].lock().unwrap();
        let (score_a, score_b) = game_loop(&mut *net_a, &mut *net_b);
        *results[i].lock().unwrap() += score_a;
        *results[j].lock().unwrap() += score_b;
    });

    results
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect()
}

fn get_elites(scores: &Vec<f64>, n: usize) -> Vec<usize> {
    let mut indexed: Vec<_> = scores.iter().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    indexed.into_iter().take(n).map(|(i, _)| i).collect()
}

fn main() {
    let mut networks: Vec<_> = (0..64)
        .into_iter()
        .map(|_| Arc::new(Mutex::new(poulet::new_chess_network().unwrap())))
        .collect();
    let matches = make_matches(networks.len(), 4);

    let start = std::time::Instant::now();
    let scores = run_matches(&mut networks, &matches);
    let duration = start.elapsed();

    let elite = get_elites(&scores, 8);
    println!("{:?}", elite);

    println!("{:?}, took {:?}", scores, duration);
}
