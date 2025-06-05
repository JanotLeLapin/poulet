use std::sync::{Arc, Mutex};

use clap::Parser;

use rand::{Rng, seq::SliceRandom};
use rand_distr::Distribution;
use rayon::prelude::*;

#[derive(Parser, Debug)]
struct Args {
    /// generation to start from
    #[arg(short, long)]
    start_gen: usize,

    /// generation to stop
    #[arg(short, long)]
    end_gen: usize,
}

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

fn game_loop(net_a: &poulet::ai::Network, net_b: &poulet::ai::Network) -> (f64, f64) {
    let buffer = net_a.get_buffer();
    let (mut buf_in, mut buf_out) = buffer;

    let networks = [net_a, net_b];
    let mut scores = [0.0, 0.0];
    let mut game = poulet::chess::Game::default();
    let mut turn = 0;
    loop {
        let net = &*networks[turn];
        let m = match poulet::next_move(net, &mut game, (&mut buf_in, &mut buf_out)) {
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

fn run_matches(networks: &[poulet::ai::Network], matches: &[(usize, usize)]) -> Vec<f64> {
    let results: Vec<Mutex<f64>> = (0..networks.len()).map(|_| Mutex::new(0.0)).collect();
    matches.par_iter().for_each(|&(i, j)| {
        if i == j || i >= networks.len() || j >= networks.len() {
            return;
        }

        let (first, second) = if i < j { (i, j) } else { (j, i) };
        let net_a = &networks[first];
        let net_b = &networks[second];
        let (score_a, score_b) = game_loop(net_a, net_b);
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

fn make_generation(
    elite: Option<Vec<poulet::ai::Network>>,
    count: usize,
) -> Vec<poulet::ai::Network> {
    if let Some(elite) = elite {
        let elite_count = elite.len();

        let mut res = Vec::with_capacity(count);
        let num_offspring = count - elite_count;

        let mut rng = rand::rng();
        let dist = rand::distr::Uniform::new(0, elite.len()).unwrap();

        for _ in 0..num_offspring {
            let parent_a = dist.sample(&mut rng);
            let parent_b = loop {
                let candidate = dist.sample(&mut rng);
                if candidate != parent_a {
                    break candidate;
                }
            };

            res.push(elite[parent_a].offspring(&elite[parent_b]).unwrap());
        }

        res.extend(elite);

        res
    } else {
        vec![0; count]
            .iter()
            .map(|_| poulet::new_chess_network().unwrap())
            .collect()
    }
}

fn main() {
    let args = Args::parse();

    let mut elite = match args.start_gen {
        0 => None,
        n => Some(
            (0..8)
                .map(|i| format!("models/gen-{}-net-{}.model", n, i))
                .map(|filename| poulet::ai::Network::load(&filename).unwrap())
                .collect(),
        ),
    };

    for i in args.start_gen..args.end_gen {
        let generation = i + 1;

        println!("making generation {}", generation);
        let networks: Vec<_> = make_generation(elite, 256).into_iter().collect();
        println!("making matches");
        let matches = make_matches(networks.len(), 16);

        let start = std::time::Instant::now();
        let scores = run_matches(&networks, &matches);
        let duration = start.elapsed();
        println!("{:?}, took {:?}", scores, duration);

        let elite_indices = get_elites(&scores, 8);
        let current_elite: Vec<_> = networks
            .into_iter()
            .enumerate()
            .filter_map(|(i, net)| {
                if elite_indices.contains(&i) {
                    Some(net)
                } else {
                    None
                }
            })
            .collect();

        if generation % 5 == 0 {
            println!("saving elite");
            for (i, net) in current_elite.iter().enumerate() {
                net.save(&format!("models/gen-{}-net-{}.model", generation, i))
                    .unwrap();
            }
        }

        elite = Some(current_elite);
    }
}
