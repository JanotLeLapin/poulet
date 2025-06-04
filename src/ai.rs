use rand_distr::{Distribution, Normal};

pub enum Activation {
    None,
    Relu,
    Softmax { temperature: f32 },
}

pub enum WeightInit {
    He,
    Xavier,
}

pub struct Layer {
    pub input_size: u64,
    pub output_size: u64,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    pub outputs: Vec<f64>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(input_size: u64, output_size: u64, activation: Activation) -> Self {
        Self {
            input_size,
            output_size,
            weights: vec![1.0; (input_size * output_size) as usize],
            biases: vec![0.0; output_size as usize],
            outputs: vec![0.0; output_size as usize],
            activation,
        }
    }

    pub fn randomize(&mut self, weight_init: WeightInit) -> Result<(), rand_distr::NormalError> {
        let mut rng = rand::rng();
        let weights_iter = match weight_init {
            WeightInit::He => {
                let std_dev = (2.0 / (self.input_size) as f64).sqrt();
                let normal = Normal::new(0.0, std_dev)?;
                normal.sample_iter(&mut rng)
            }
            WeightInit::Xavier => {
                let std_dev = (2.0 / (self.input_size + self.output_size) as f64).sqrt();
                let normal = Normal::new(0.0, std_dev)?;
                normal.sample_iter(&mut rng)
            }
        };

        for (w, val) in self.weights.iter_mut().zip(weights_iter) {
            *w = val;
        }

        Ok(())
    }

    pub fn forward(&mut self, inputs: &Vec<f64>) {
        for (i, output) in self.outputs.iter_mut().enumerate() {
            let start = i * self.input_size as usize;
            let end = start + self.input_size as usize;

            let sum: f64 = self.weights[start..end]
                .iter()
                .zip(inputs.iter())
                .map(|(w, x)| w * x)
                .sum();

            *output = self.biases[i] + sum;
        }

        match self.activation {
            Activation::None => {}
            Activation::Relu => {
                self.outputs.iter_mut().for_each(|o| {
                    if *o < 0.0 {
                        *o = 0.0;
                    }
                });
            }
            Activation::Softmax { temperature } => {
                softmax(&mut self.outputs);
            }
        }
    }
}

pub type Network = Vec<Layer>;

pub fn softmax(logits: &mut Vec<f64>) {
    let max = match logits.iter().max_by(|a, b| a.partial_cmp(b).unwrap()) {
        Some(v) => *v,
        None => return,
    };

    let mut sum_exp = 0.0;
    for v in logits.iter_mut() {
        *v = (*v - max).exp();
        sum_exp += *v;
    }

    for v in logits.iter_mut() {
        *v /= sum_exp;
    }
}

pub fn forward(network: &mut Network, inputs: &Vec<f64>) {
    let mut tmp = inputs;
    for layer in network {
        layer.forward(tmp);
        tmp = &layer.outputs;
    }
}
