use poulet_ai_common::encode_board;
use poulet_chess::Game;
use rand_distr::Distribution;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use crate::model::{Player, State, predict_move};

use futures::executor::block_on;

pub const GAME_PER_PLAYER: usize = 8;
pub const POOL_COUNT: usize = 2;

pub const PLAYER_PER_GEN: usize = POOL_COUNT * (GAME_PER_PLAYER + 1);

pub struct Match {
    pub player_indices: [usize; 2],
    pub white_score: f32,
    pub black_score: f32,
}

pub struct Pool {
    pub matches: Vec<Match>,
}

pub struct Generation {
    pub players: Vec<Player>,
    pub pools: Vec<Pool>,
}

impl Match {
    pub fn new(white: usize, black: usize) -> Self {
        Self {
            player_indices: [white, black],
            white_score: 0.0,
            black_score: 0.0,
        }
    }

    pub fn match_loop(&mut self, players: &Vec<Player>) {
        let players = self.player_indices.map(|i| players.get(i).unwrap());

        let mut game = Game::default();

        let mut current = 0;
        let mut next;
        let mut is_checkmate = false;
        let mut scores: [Vec<f32>; 2] = [vec![], vec![]];

        loop {
            next = (current + 1) % 2;

            let neurons = encode_board(&game.board);
            let logits = block_on(async { players[current].forward(&neurons).await }).unwrap();
            let res = match predict_move(&mut game, logits, 1.0) {
                Ok(res) => res,
                Err(_) => break,
            };
            let (_, src_file, src_rank, dst_file, dst_rank) = res;

            if !game.safe_move(src_file, src_rank, dst_file, dst_rank) {
                break;
            }

            scores[current].push(0.0); // TODO: evaluated score here
            game.do_move(src_file, src_rank, dst_file, dst_rank);

            if game.is_checkmate(next.try_into().unwrap()) {
                is_checkmate = true;
                break;
            }

            if game.until_stalemate >= 60 {
                break;
            }

            current = next;
        }

        let mut scores: [f32; 2] = scores.map(|vec| {
            let len = vec.len() as f32;
            vec.into_iter().map(|v| v / len).sum()
        });

        if is_checkmate {
            scores[current] += 10.0;
            scores[next] -= 10.0;
        }

        println!("{:?} done, scores: {:?}", self.player_indices, scores);

        if is_checkmate {
            println!("fen: {:?}", game.board.fen());
        }

        self.white_score = scores[0];
        self.black_score = scores[1];
    }
}

impl Pool {
    pub fn init(player_indices: &[usize]) -> Self {
        let indice_couples: Vec<(usize, usize)> = (0..9)
            .flat_map(|i| (0..9).map(move |j| (i, j)))
            .filter(|(i, j)| i < j)
            .collect();

        let matches = indice_couples
            .iter()
            .map(|(a, b)| Match::new(player_indices[*a], player_indices[*b]))
            .collect();

        Self { matches }
    }

    pub fn play(&mut self, players: &Vec<Player>) {
        for m in self.matches.iter_mut() {
            m.match_loop(players);
        }
    }
}

impl Generation {
    pub fn new(player_indices: Vec<usize>, players: Vec<Player>) -> Self {
        let pools = (0..POOL_COUNT)
            .map(|i| {
                Pool::init(
                    &player_indices[(i * (GAME_PER_PLAYER + 1))..((i + 1) * (GAME_PER_PLAYER + 1))],
                )
            })
            .collect();

        Self { players, pools }
    }

    pub fn init(state: &State) -> Self {
        let (player_indices, players) = (0..PLAYER_PER_GEN)
            .map(|_| Player::init(state).unwrap())
            .enumerate()
            .unzip();

        Self::new(player_indices, players)
    }

    pub fn populate(state: &State, elite: Vec<Player>) -> Self {
        let mut rng = rand::rng();
        let distr = rand::distr::Uniform::new(0, elite.len()).unwrap();

        let (player_indices, players) = (0..PLAYER_PER_GEN)
            .map(|_| {
                elite
                    .get(distr.sample(&mut rng))
                    .unwrap()
                    .crossover_and_mutate(
                        state,
                        elite.get(distr.sample(&mut rng)).unwrap(),
                        0.5,
                        0.1,
                    )
            })
            .enumerate()
            .unzip();

        Self::new(player_indices, players)
    }

    pub fn play(&mut self) {
        self.pools
            .par_iter_mut()
            .for_each(|p| p.play(&self.players));
    }

    pub fn get_elite(&self, count: usize) -> Vec<usize> {
        let mut scores: Vec<f32> = (0..PLAYER_PER_GEN).map(|_| 0.0).collect();
        for p in self.pools.iter() {
            for m in p.matches.iter() {
                let [w, b] = m.player_indices;
                scores[w] += m.white_score;
                scores[b] += m.black_score;
            }
        }

        let mut enumerated: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        enumerated.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        enumerated.iter().map(|(i, _)| *i).take(count).collect()
    }
}
