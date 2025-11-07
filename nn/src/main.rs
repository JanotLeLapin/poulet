use burn::backend::Wgpu;

mod model;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = model::ModelConfig::new().init::<MyBackend>(&device);

    println!("{model:#?}");
}
