use crate::model::{Player, State};

pub const GAME_PER_PLAYER: usize = 8;
pub const POOL_COUNT: usize = 14;
pub const PLAYER_PER_GEN: usize = POOL_COUNT * (GAME_PER_PLAYER + 1);

pub struct Match {
    players: [Player; 2],
    scores: [Vec<f32>; 2],
}

pub struct Pool {
    matches: [Match; GAME_PER_PLAYER * (GAME_PER_PLAYER + 1) / 2],
}

pub struct Generation {
    pools: [Pool; POOL_COUNT],
}

impl Match {
    pub fn new(white: Player, black: Player) -> Self {
        Self {
            players: [white, black],
            scores: [vec![], vec![]],
        }
    }
}

impl Pool {
    pub fn init(state: &State) -> Self {
        let matches = std::array::from_fn(|_| {
            Match::new(Player::new(state).unwrap(), Player::new(state).unwrap())
        });

        Self { matches }
    }
}

impl Generation {
    pub fn init(state: &State) -> Self {
        let pools = std::array::from_fn(|_| Pool::init(state));

        Self { pools }
    }

    pub fn get_players(self) -> Vec<Player> {
        let mut res = Vec::with_capacity(PLAYER_PER_GEN);

        for p in self.pools {
            for m in p.matches {
                res.extend(m.players);
            }
        }

        res
    }
}
