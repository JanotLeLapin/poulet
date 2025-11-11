use std::collections::HashSet;

use poulet_ai_common::encode_board;
use poulet_chess::{Board, Game};
use rand::{Rng, seq::SliceRandom};
use rand_distr::Distribution;
use rayon::prelude::*;

use crate::model::{BATCH_SIZE, Player, State, predict_move};

pub struct Match {
    player_indices: [usize; 2],
    scores: [f32; 2],
    game: Game,
}

struct MatchUpdate {
    new_game: Option<Game>,
    score_updates: [f32; 2],
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

    pub fn populate(
        state: &State,
        elite: Vec<Player>,
        pop_size: usize,
        mut_dev: f32,
        burst_mut_rate: f32,
        burst_mut_dev: f32,
        elitism_count: usize,
    ) -> Self {
        let mut rng = rand::rng();
        let distr = rand::distr::Uniform::new(0, elite.len()).unwrap();

        let corrected_elitism_count = elitism_count.min(elite.len());

        let mut players = Vec::with_capacity(pop_size);

        for _ in 0..(pop_size - corrected_elitism_count) {
            let (a, b) = loop {
                let a = distr.sample(&mut rng);
                let b = distr.sample(&mut rng);

                if a != b {
                    break (a, b);
                }
            };

            let dev = if rng.random::<f32>() < burst_mut_rate {
                burst_mut_dev
            } else {
                mut_dev
            };

            let player =
                elite
                    .get(a)
                    .unwrap()
                    .crossover_and_mutate(state, elite.get(b).unwrap(), 0.5, dev);

            players.push(player);
        }

        let mut truncated_elite = elite.into_iter().take(corrected_elitism_count).collect();
        players.append(&mut truncated_elite);

        players.reverse();

        Self::new(players)
    }

    pub fn create_match(&mut self, white: usize, black: usize) {
        if white == black
            || self.player_map.get(&white).is_none()
            || self.player_map.get(&black).is_none()
        {
            return;
        }

        let w = self.player_map.get_mut(&white).unwrap();
        w.insert(self.matches.len());

        let b = self.player_map.get_mut(&black).unwrap();
        b.insert(self.matches.len());

        let m = Match::new(white, black);
        self.matches.push(m);
    }

    pub fn generate_matches(&mut self, match_per_player: usize) {
        let match_per_player = match_per_player.div_ceil(2);

        let mut rng = rand::rng();
        let mut order: Vec<_> = (0..self.players.len()).collect();
        order.shuffle(&mut rng);

        for offset in 1..=match_per_player {
            for i in 0..self.players.len() {
                let a = order[i];
                let b = order[(i + offset) % self.players.len()];
                self.create_match(a, b);
                self.create_match(b, a);
            }
        }
    }

    pub async fn play(&mut self) {
        loop {
            let player_map = &self.player_map;
            let matches = &self.matches;

            let player_futures: Vec<_> = self
                .players
                .par_iter()
                .enumerate()
                .map(|(i, p)| async move {
                    let match_indices = player_map.get(&i).unwrap();
                    let (match_indices, boards): (Vec<_>, Vec<_>) = match_indices
                        .iter()
                        .filter_map(|j| {
                            let m = matches.get(*j).unwrap();
                            let is_white = m.player_indices[0] == i;
                            let is_turn_white = m.game.turn == poulet_chess::Color::White;
                            if is_white == is_turn_white {
                                let mut board;
                                if is_white {
                                    board = m.game.board.clone();
                                } else {
                                    board = Board::new();
                                    m.game.board.flip_to(&mut board);
                                }

                                Some((*j, encode_board(&board)))
                            } else {
                                None
                            }
                        })
                        .take(BATCH_SIZE)
                        .unzip();

                    if match_indices.len() == 0 {
                        return None;
                    }

                    let output = p.forward(&boards).await.ok()?;
                    Some((match_indices, output))
                })
                .collect();

            let results: Vec<_> = futures::future::join_all(player_futures)
                .await
                .into_iter()
                .filter_map(|v| v)
                .flat_map(|(match_indices, output)| output.into_iter().zip(match_indices))
                .collect();

            if results.is_empty() {
                break;
            }

            let new_states: Vec<_> = results
                .par_iter()
                .map(|(l, mi)| {
                    let m = self.matches.get(*mi).unwrap();
                    let mut tmp_game = m.game.clone();
                    let should_unflip = tmp_game.turn == poulet_chess::Color::Black;
                    let prediction = predict_move(&mut tmp_game, should_unflip, l.clone(), 1.0);
                    if prediction
                        .map(|(_, src_file, src_rank, dst_file, dst_rank)| {
                            !tmp_game.safe_move(src_file, src_rank, dst_file, dst_rank)
                        })
                        .unwrap_or(true)
                    {
                        if tmp_game.is_checkmate(m.game.turn) {
                            println!("{mi}: checkmate! {}", tmp_game.board.fen());
                            let score_update = if m.game.turn == poulet_chess::Color::White {
                                [-10.0, 10.0]
                            } else {
                                [10.0, -10.0]
                            };

                            return (
                                mi,
                                MatchUpdate {
                                    new_game: None,
                                    score_updates: score_update,
                                },
                            );
                        } else {
                            println!("{mi}: stalemate");
                            return (
                                mi,
                                MatchUpdate {
                                    new_game: None,
                                    score_updates: [0.0; 2],
                                },
                            );
                        }
                    }

                    let (_, src_file, src_rank, dst_file, dst_rank) = prediction.unwrap();

                    tmp_game.do_move(src_file, src_rank, dst_file, dst_rank);

                    if tmp_game.until_stalemate >= 60 {
                        println!("{mi}: draw");
                        return (
                            mi,
                            MatchUpdate {
                                new_game: None,
                                score_updates: [0.0; 2],
                            },
                        );
                    }

                    (
                        mi,
                        MatchUpdate {
                            new_game: Some(tmp_game),
                            score_updates: [0.0; 2],
                        },
                    )
                })
                .collect();

            let mut rem = vec![];
            for (mi, update) in new_states {
                let m: &mut Match = self.matches.get_mut(*mi).unwrap();
                m.scores[0] += update.score_updates[0];
                m.scores[1] += update.score_updates[1];
                match update.new_game {
                    Some(new_game) => m.game = new_game,
                    None => rem.push(mi),
                }
            }

            for mi in rem {
                println!("removing match {mi}");
                let m: &Match = self.matches.get(*mi).unwrap();
                for i in m.player_indices {
                    self.player_map.get_mut(&i).unwrap().remove(&mi);
                }
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
