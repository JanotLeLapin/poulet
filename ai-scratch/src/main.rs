use poulet_ai_scratch::{
    genetic::Generation,
    model::{Player, State},
};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Generation where the algorithm left off
    #[arg(short, long)]
    generation: u32,

    /// Model save interval
    #[arg(short, long, default_value_t = 5)]
    save_interval: u32,

    /// Number of individuals in the elite
    #[arg(short, long, default_value_t = 8)]
    elite_size: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let state = State::new().await;

    let mut g = args.generation;
    let mut generation;
    let mut elite;

    if g == 0 {
        generation = Generation::init(&state);
        generation.play();
        elite = generation.get_elite(args.elite_size);
    } else {
        elite = (0..args.elite_size)
            .map(|i| {
                Player::load(&state, &format!("./ai-scratch/models/model-{g}-{i}.params")).unwrap()
            })
            .collect();
    }

    loop {
        println!("--- GENERATION {g} ---");
        let mut generation = Generation::populate(&state, elite);
        generation.play();
        elite = generation.get_elite(args.elite_size);
        g += 1;

        if g % args.save_interval == 0 {
            elite.iter().enumerate().for_each(|(i, p)| {
                p.save(&format!("./ai-scratch/models/model-{g}-{i}.params"))
                    .unwrap()
            });
        }
    }
}
