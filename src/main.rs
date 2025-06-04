mod ai;
mod chess;

use rand_distr::Distribution;

#[derive(Debug)]
pub struct Move {
    src: chess::Position,
    dst: chess::Position,
}

pub fn new_chess_network() -> Result<ai::Network, rand_distr::NormalError> {
    let mut layers = vec![
        ai::Layer::new(768, 512, ai::Activation::Relu),
        ai::Layer::new(512, 1024, ai::Activation::Relu),
        ai::Layer::new(1024, 4096, ai::Activation::Softmax { temperature: 0.6 }),
    ];

    layers[0].randomize(ai::WeightInit::He)?;
    layers[1].randomize(ai::WeightInit::He)?;
    layers[2].randomize(ai::WeightInit::Xavier)?;

    Ok(layers)
}

pub fn encode_board(board: &chess::Board) -> Vec<f64> {
    let mut res = vec![0.0; 768];

    for x in 0..8 {
        for y in 0..8 {
            let chess::Piece { color, piece_type } = match board.get_square(x, y) {
                Some(v) => v,
                None => continue,
            };

            res[(x as usize * 8 + y as usize) * 12 + piece_type as usize + color as usize * 6] =
                1.0;
        }
    }

    res
}

pub fn move_from_index(idx: usize) -> Move {
    let src_index = idx / 64;
    let dst_index = idx % 64;

    Move {
        src: ((src_index % 8) as u8, (src_index / 8) as u8),
        dst: ((dst_index % 8) as u8, (dst_index / 8) as u8),
    }
}

pub fn next_move(
    network: &mut ai::Network,
    game: &mut chess::Game,
) -> Result<Move, rand::distr::weighted::Error> {
    let encoded_board = encode_board(&game.board);
    ai::forward(network, &encoded_board);

    let idx = network.len() - 1;
    let logits = &mut network[idx].outputs;

    for (m, logit) in logits
        .iter_mut()
        .enumerate()
        .map(|(idx, v)| (move_from_index(idx), v))
    {
        if !game.safe_move(m.src.0, m.src.1, m.dst.0, m.dst.1) {
            *logit = -f64::INFINITY;
        }
    }

    ai::softmax(logits);

    let dist = rand::distr::weighted::WeightedIndex::new(&logits[..])?;
    let rng = rand::rng();
    for v in dist.sample_iter(rng) {
        let m = move_from_index(v);
        if game.safe_move(m.src.0, m.src.1, m.dst.0, m.dst.1) {
            return Ok(m);
        }
    }

    panic!();
}

fn main() {
    let mut game = chess::Game::default();
    let mut network = new_chess_network().unwrap();

    let next_move = next_move(&mut network, &mut game);
    println!("{:?}", next_move);
}
