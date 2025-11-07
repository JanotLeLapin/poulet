use burn::{
    nn::{Linear, LinearConfig, Relu},
    prelude::*,
};

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "512")]
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(768, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.hidden_size).init(device),
            linear3: LinearConfig::new(self.hidden_size, 4096).init(device),
            activation: Relu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, state: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(state);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear3.forward(x);
        x
    }
}
