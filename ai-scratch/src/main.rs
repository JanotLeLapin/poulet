use poulet_ai_scratch::{
    genetic::Generation,
    model::{Player, State},
};

use clap::{Command, arg, value_parser};

fn cli() -> Command {
    Command::new("poulet")
        .about("A toy generic algorithm that learns to play chess")
        .subcommand_required(true)
        .subcommands([
            Command::new("train")
                .arg(
                    arg!(-g --generation [GEN] "Generation where the algorithm left off")
                        .default_value("0")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    arg!(-p --population [POP] "Population size")
                        .default_value("64")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    arg!(-m --matches [MATCHES] "Match count per individual")
                        .default_value("16")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    arg!(--interval [INTERVAL] "Generation save interval")
                        .default_value("10")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    arg!(--elite [ELITE] "Number of elite individuals for each generation")
                        .default_value("8")
                        .value_parser(value_parser!(usize)),
                ),
            Command::new("test")
                .arg(arg!(<PLAYER> "Path to player model"))
                .arg(arg!(--games [GAMES] "Game count").value_parser(value_parser!(usize))),
        ])
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = State::new().await;

    let m = cli().get_matches();
    match m.subcommand() {
        Some(("train", sub)) => {
            let mut g: usize = *sub.get_one("generation").unwrap();
            let pop_size: usize = *sub.get_one("population").unwrap();
            let match_count: usize = *sub.get_one("matches").unwrap();
            let save_interval: usize = *sub.get_one("interval").unwrap();
            let elite_size: usize = *sub.get_one("elite").unwrap();

            println!(
                "starting training with population size = {pop_size}, match count = {match_count}"
            );

            let mut generation;
            let mut elite;

            if g == 0 {
                generation = Generation::init(&state, pop_size);
                generation.generate_matches(match_count);
                generation.play().await;
                elite = generation.get_elite(elite_size);
            } else {
                elite = (0..elite_size)
                    .map(|i| {
                        Player::load(&state, &format!("./ai-scratch/models/model-{g}-{i}.params"))
                            .unwrap()
                    })
                    .collect();
            }

            loop {
                println!("--- GENERATION {g} ---");
                generation = Generation::populate(&state, elite, pop_size);
                generation.generate_matches(match_count);
                generation.play().await;
                elite = generation.get_elite(elite_size);
                g += 1;

                if g % save_interval == 0 {
                    elite.iter().enumerate().for_each(|(i, p)| {
                        p.save(&format!("./ai-scratch/models/model-{g}-{i}.params"))
                            .unwrap()
                    });
                }
            }
        }
        Some(("test", sub)) => {
            // let player_path: &String = sub.get_one("PLAYER").unwrap();
            // let game_count: usize = *sub.get_one("GAMES").unwrap_or(&16);

            // let mut players = Vec::with_capacity(2);
            // players.push(Player::load(&state, player_path)?);
            // players.push(Player::init(&state)?);

            // let mut scores = vec![0.0, 0.0];

            // for i in 0..game_count {
            //     let a = i % 2;
            //     let b = (i + 1) % 2;
            //     let mut m = Match::new(a, b);
            //     println!("starting game {i} {a} {b}");
            //     m.match_loop(&players);

            //     scores[a] += (m.white_score / 20.0) + 0.5;
            //     scores[b] += (m.black_score / 20.0) + 0.5;
            // }

            // println!("Your score: {}", scores[0]);
            // println!("Random score: {}", scores[1]);
            // println!(
            //     "Win rate: {}%",
            //     (scores[0] as f32) / ((scores[0] + scores[1]) as f32) * 100.0
            // );
        }
        _ => {
            cli().print_help()?;
        }
    }

    Ok(())
}
