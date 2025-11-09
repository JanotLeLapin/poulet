use std::collections::HashSet;

use poulet_ai_common::encode_board;
use poulet_chess::{Board, Game};
use rand_distr::Distribution;

use crate::model::{BATCH_SIZE, Player, State, predict_move};

pub struct Match {
    player_indices: [usize; 2],
    scores: [f32; 2],
    game: Game,
}

pub struct Generation {
    matches: Vec<Match>,
    players: Vec<Player>,
    player_map: std::collections::HashMap<usize, std::collections::HashSet<usize>>,
}

impl Match {
    pub fn new(white: usize, black: usize) -> Self {
        Self {
            player_indices: [white, black],
            scores: [0.0; 2],
            game: Game::default(),
        }
    }
}

impl Generation {
    pub fn new(players: Vec<Player>) -> Self {
        let player_map = players
            .iter()
            .enumerate()
            .map(|(i, _)| (i, std::collections::HashSet::new()))
            .collect();
        Self {
            matches: Vec::new(),
            players,
            player_map,
        }
    }

    pub fn init(state: &State, pop_size: usize) -> Self {
        Self::new(
            (0..pop_size)
                .map(|_| Player::init(state).unwrap())
                .collect(),
        )
    }

    pub fn populate(state: &State, elite: Vec<Player>, pop_size: usize) -> Self {
        let mut rng = rand::rng();
        let distr = rand::distr::Uniform::new(0, elite.len()).unwrap();

        let players = (0..pop_size)
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
            .collect();

        Self::new(players)
    }

    pub fn generate_matches(&mut self, match_per_player: usize) {
        let mut rng = rand::rng();
        let distr = rand::distr::Uniform::new(0, self.players.len()).unwrap();

        for i in 0..self.players.len() {
            for _ in 0..match_per_player.div_ceil(2) {
                let (a, b) = loop {
                    let a = distr.sample(&mut rng);
                    let b = distr.sample(&mut rng);

                    if a != b && a != i && b != i {
                        break (a, b);
                    }
                };
                self.player_map
                    .get_mut(&i)
                    .unwrap()
                    .insert(self.matches.len());
                self.player_map
                    .get_mut(&i)
                    .unwrap()
                    .insert(self.matches.len() + 1);
                self.player_map
                    .get_mut(&a)
                    .unwrap()
                    .insert(self.matches.len());
                self.player_map
                    .get_mut(&b)
                    .unwrap()
                    .insert(self.matches.len() + 1);
                self.matches.extend([Match::new(a, i), Match::new(i, b)]);
            }
        }
    }

    pub async fn play(&mut self) {
        loop {
            let mut should_break = true;
            for (i, p) in self.players.iter().enumerate() {
                let match_indices = self.player_map.get_mut(&i).unwrap();
                let (match_indices, boards): (Vec<_>, Vec<_>) = match_indices
                    .iter()
                    .filter_map(|j| {
                        let m = self.matches.get(*j).unwrap();
                        let is_white = m.player_indices[0] == i;
                        let is_turn_white = m.game.turn == poulet_chess::Color::White;
                        if is_white == is_turn_white {
                            let mut board;
                            if is_white {
                                board = m.game.board.clone();
                            } else {
                                board = Board::new();
                                m.game.board.flip(&mut board);
                            }

                            Some((j, encode_board(&board)))
                        } else {
                            None
                        }
                    })
                    .take(BATCH_SIZE)
                    .unzip();

                if match_indices.len() == 0 {
                    continue;
                }

                should_break = false;
                let output = p.forward(&boards).await.unwrap();

                let mut rem = vec![];

                for (l, mi) in output.into_iter().zip(match_indices) {
                    let m: &mut Match = self.matches.get_mut(mi).unwrap();
                    let (should_unflip, next) = match m.game.turn {
                        poulet_chess::Color::White => (false, poulet_chess::Color::Black),
                        poulet_chess::Color::Black => (true, poulet_chess::Color::White),
                    };
                    let (_, src_file, src_rank, dst_file, dst_rank) =
                        match predict_move(&mut m.game, should_unflip, l, 1.0) {
                            Ok(prediction) => prediction,
                            Err(_) => {
                                rem.push(mi);
                                continue;
                            }
                        };

                    if !m.game.safe_move(src_file, src_rank, dst_file, dst_rank) {
                        rem.push(mi);
                        continue;
                    }

                    m.game.do_move(src_file, src_rank, dst_file, dst_rank);

                    if m.game.is_checkmate(next) {
                        m.scores[usize::from(m.game.turn)] = 10.0;
                        m.scores[usize::from(next)] = -10.0;
                        rem.push(mi);
                        println!("checkmate! {}", m.game.board.fen());
                        continue;
                    }

                    if m.game.until_stalemate >= 60 {
                        rem.push(mi);
                        continue;
                    }
                }

                for mi in rem {
                    println!("removing match {mi}");
                    let m: &Match = self.matches.get(mi).unwrap();
                    for i in m.player_indices {
                        self.player_map.get_mut(&i).unwrap().remove(&mi);
                    }
                }
            }

            if should_break {
                break;
            }
        }
    }

    pub fn get_elite(self, count: usize) -> Vec<Player> {
        let mut scores: Vec<f32> = (0..self.players.len()).map(|_| 0.0).collect();
        for m in self.matches.iter() {
            let [w, b] = m.player_indices;
            scores[w] += m.scores[0];
            scores[b] += m.scores[1];
        }

        let mut enumerated: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
        enumerated.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let indices: HashSet<_> = enumerated.iter().map(|(i, _)| *i).take(count).collect();

        self.players
            .into_iter()
            .enumerate()
            .filter_map(|(i, p)| indices.contains(&i).then_some(p))
            .collect()
    }
}
