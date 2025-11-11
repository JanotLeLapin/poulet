use poulet_ai_scratch::{
    genetic::Generation,
    model::{Player, State},
};

use clap::{Arg, Command, arg, value_parser};

fn cli() -> Command {
    Command::new("poulet")
        .about("A toy generic algorithm that learns to play chess")
        .subcommand_required(true)
        .subcommands([
            Command::new("train")
                .arg(
                    Arg::new("gen-start")
                        .long("gen-start")
                        .help("Start generation")
                        .default_value("0")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    Arg::new("gen-end")
                        .long("gen-end")
                        .help("End generation")
                        .value_parser(value_parser!(usize)),
                )
                .arg(
                    Arg::new("mut-dev")
                        .long("mut-dev")
                        .help("Mutation standard deviation")
                        .default_value("0.04")
                        .value_parser(value_parser!(f32)),
                )
                .arg(
                    Arg::new("burst-mut-rate")
                        .long("burst-mut-rate")
                        .help("Burst mutation population rate")
                        .default_value("0.0")
                        .value_parser(value_parser!(f32)),
                )
                .arg(
                    Arg::new("burst-mut-dev")
                        .long("burst-mut-dev")
                        .help("Burst mutation standard deviation")
                        .default_value("0.2")
                        .value_parser(value_parser!(f32)),
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
                .arg(
                    arg!(--games [GAMES] "Game count")
                        .default_value("64")
                        .value_parser(value_parser!(usize)),
                ),
        ])
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = State::new().await;

    let m = cli().get_matches();
    match m.subcommand() {
        Some(("train", sub)) => {
            let mut g: usize = *sub.get_one("gen-start").unwrap();
            let g_end: usize = *sub.get_one("gen-end").unwrap_or(&usize::MAX);
            let mut_dev: f32 = *sub.get_one("mut-dev").unwrap();
            let burst_mut_rate: f32 = *sub.get_one("burst-mut-rate").unwrap();
            let burst_mut_dev: f32 = *sub.get_one("burst-mut-dev").unwrap();
            let pop_size: usize = *sub.get_one("population").unwrap();
            let match_count: usize = *sub.get_one("matches").unwrap();
            let save_interval: usize = *sub.get_one("interval").unwrap();
            let elite_size: usize = *sub.get_one("elite").unwrap();

            println!(
                r#"
starting training with following params:
mutation deviation = {mut_dev}
burst mutation rate: {}%
burst mutation deviation = {burst_mut_dev}
population size = {pop_size}
match count = {match_count}
save interval = {save_interval}
elite size = {elite_size}
"#,
                burst_mut_rate * 100.0,
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

            while g <= g_end {
                println!("--- GENERATION {g} ---");
                generation = Generation::populate(
                    &state,
                    elite,
                    pop_size,
                    mut_dev,
                    burst_mut_rate,
                    burst_mut_dev,
                    2,
                );
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
            let player_path: &String = sub.get_one("PLAYER").unwrap();
            let game_count: usize = *sub.get_one("games").unwrap_or(&16);

            let mut players = Vec::with_capacity(2);
            players.push(Player::load(&state, player_path)?);
            players.push(Player::init(&state)?);

            let mut wins = vec![0, 0];
            let mut losses = vec![0, 0];

            let mut generation = Generation::new(players);

            for i in 0..game_count {
                generation.create_match(i % 2, (i + 1) % 2);
            }

            generation.play().await;

            for (i, m) in generation.matches.iter().enumerate() {
                // TODO: we should encode the actual game outcome somewhere else
                if m.scores[0] == m.scores[1] {
                    continue;
                }

                let player_is_white = (i % 2) == 0;
                let white_won = m.scores[0] > 0.0;
                if player_is_white == white_won {
                    wins[0] += 1;
                    losses[1] += 1;
                } else {
                    wins[1] += 1;
                    losses[0] += 1;
                }
            }

            println!(
                r#"
Your wins: {}/{game_count} (rate: {}%)
Your losses: {}/{game_count} (rate: {}%)
Your global score: {}

Random wins: {}/{game_count} (rate: {}%)
Random losses: {}/{game_count} (rate: {}%)
Random global score: {}
            "#,
                wins[0],
                (wins[0] as f32) / (game_count as f32) * 100.0,
                losses[0],
                (losses[0] as f32) / (game_count as f32) * 100.0,
                ((game_count - wins[0] - losses[0]) as f32) * 0.5 + (wins[0] as f32)
                    - (losses[0] as f32),
                wins[1],
                (wins[1] as f32) / (game_count as f32) * 100.0,
                losses[1],
                (losses[1] as f32) / (game_count as f32) * 100.0,
                ((game_count - wins[1] - losses[1]) as f32) * 0.5 + (wins[1] as f32)
                    - (losses[1] as f32)
            );
        }
        _ => {
            cli().print_help()?;
        }
    }

    Ok(())
}
