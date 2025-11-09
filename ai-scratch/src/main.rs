use poulet_ai_scratch::{
    genetic::Generation,
    model::{Player, State},
};

use clap::{Command, arg, value_parser};

fn cli() -> Command {
    Command::new("poulet")
        .about("A toy generic algorithm that learns to play chess")
        .subcommand_required(true)
        .subcommands([Command::new("train")
            .arg(
                arg!(<GEN> "Generation where the algorithm left off")
                    .required(true)
                    .value_parser(value_parser!(usize)),
            )
            .arg(
                arg!(--interval [INTERVAL] "Generation save interval")
                    .value_parser(value_parser!(usize)),
            )
            .arg(
                arg!(--elite [ELITE] "Number of elite individuals for each generation")
                    .value_parser(value_parser!(usize)),
            )])
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let state = State::new().await;

    let m = cli().get_matches();
    match m.subcommand() {
        Some(("train", sub)) => {
            let mut g: usize = *sub.get_one::<usize>("GEN").unwrap();
            let save_interval: usize = *sub.get_one("INTERVAL").unwrap_or(&5);
            let elite_size: usize = *sub.get_one("ELITE").unwrap_or(&8);

            let mut generation;
            let mut elite;

            if g == 0 {
                generation = Generation::init(&state);
                generation.play();
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
                let mut generation = Generation::populate(&state, elite);
                generation.play();
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
        _ => {
            cli().print_help()?;
        }
    }

    Ok(())
}
