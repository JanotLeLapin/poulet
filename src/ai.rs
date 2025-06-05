use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum Activation {
    None,
    Relu,
    Softmax { temperature: f32 },
}

pub enum WeightInit {
    He,
    Xavier,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Layer {
    pub input_size: u64,
    pub output_size: u64,
    pub weights: Vec<f64>,
    pub biases: Vec<f64>,
    pub activation: Activation,
}

impl Layer {
    pub fn new(input_size: u64, output_size: u64, activation: Activation) -> Self {
        Self {
            input_size,
            output_size,
            weights: vec![1.0; (input_size * output_size) as usize],
            biases: vec![0.0; output_size as usize],
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

    pub fn offspring(&self, other: &Self) -> Result<Self, OffspringError> {
        let mut child = Self::new(self.input_size, self.output_size, self.activation);
        offspring(&self.weights, &other.weights, &mut child.weights)?;
        offspring(&self.biases, &self.biases, &mut child.biases)?;
        Ok(child)
    }

    pub fn forward(&self, inputs: &Vec<f64>, outputs: &mut Vec<f64>) {
        for (i, output) in outputs.iter_mut().enumerate() {
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
                outputs.iter_mut().for_each(|o| {
                    if *o < 0.0 {
                        *o = 0.0;
                    }
                });
            }
            Activation::Softmax { temperature } => {
                softmax(outputs);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Network(pub Vec<Layer>);

#[derive(Debug)]
pub enum NetworkSaveError {
    EncodeError(rmp_serde::encode::Error),
    IOError(std::io::Error),
}

#[derive(Debug)]
pub enum NetworkLoadError {
    DecodeError(rmp_serde::decode::Error),
    IOError(std::io::Error),
}

impl Network {
    pub fn load(filename: &str) -> Result<Self, NetworkLoadError> {
        rmp_serde::from_slice(
            &std::fs::read(filename).map_err(|err| NetworkLoadError::IOError(err))?[..],
        )
        .map_err(|err| NetworkLoadError::DecodeError(err))
    }

    pub fn save(&self, filename: &str) -> Result<(), NetworkSaveError> {
        std::fs::write(
            filename,
            rmp_serde::to_vec(self).map_err(|err| NetworkSaveError::EncodeError(err))?,
        )
        .map_err(|err| NetworkSaveError::IOError(err))
    }

    pub fn offspring(&self, other: &Self) -> Result<Self, OffspringError> {
        let mut res = Vec::with_capacity(self.0.len());

        for (a, b) in self.0.iter().zip(&other.0) {
            res.push(a.offspring(b)?);
        }

        Ok(Self(res))
    }

    pub fn get_buffer(&self) -> (Vec<f64>, Vec<f64>) {
        let max_output_size = self
            .0
            .iter()
            .map(|layer| layer.output_size as usize)
            .max()
            .unwrap_or(0);

        (
            Vec::with_capacity(max_output_size),
            Vec::with_capacity(max_output_size),
        )
    }

    pub fn forward(&self, inputs: &Vec<f64>, layer: (&mut Vec<f64>, &mut Vec<f64>)) {
        let (input, output) = layer;

        input.extend_from_slice(inputs);

        for layer in &self.0 {
            output.resize(layer.output_size as usize, 0.0);
            layer.forward(input, output);
            std::mem::swap(input, output);
        }
    }
}

impl TryFrom<Vec<u8>> for Network {
    type Error = rmp_serde::decode::Error;

    fn try_from(value: Vec<u8>) -> Result<Self, Self::Error> {
        rmp_serde::from_slice(&value[..])
    }
}

pub fn softmax(logits: &mut Vec<f64>) {
    let max = match logits
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    {
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

#[derive(Debug)]
pub enum OffspringError {
    UniformError(rand::distr::uniform::Error),
    WeightedIndexError(rand::distr::weighted::Error),
    NormalError(rand_distr::NormalError),
}

pub fn offspring(a: &Vec<f64>, b: &Vec<f64>, dst: &mut Vec<f64>) -> Result<(), OffspringError> {
    let crossover_rng = rand::rng();
    let crossover = crossover_rng.sample_iter(
        rand::distr::Uniform::new(0.0, 1.0).map_err(|err| OffspringError::UniformError(err))?,
    );

    let mutation_rng = rand::rng();
    let mutation = mutation_rng.sample_iter(
        rand::distr::weighted::WeightedIndex::new([0.8, 0.15, 0.05])
            .map_err(|err| OffspringError::WeightedIndexError(err))?,
    );

    let smooth_mutation_rng = rand::rng();
    let smooth_mutation = smooth_mutation_rng.sample_iter(
        rand_distr::Normal::new(0.0, 0.02).map_err(|err| OffspringError::NormalError(err))?,
    );

    let burst_mutation_rng = rand::rng();
    let burst_mutation = burst_mutation_rng.sample_iter(
        rand_distr::Normal::new(0.0, 0.2).map_err(|err| OffspringError::NormalError(err))?,
    );

    for ((((((v, av), bv), alpha), mutation_type), smooth), burst) in dst
        .iter_mut()
        .zip(a)
        .zip(b)
        .zip(crossover)
        .zip(mutation)
        .zip(smooth_mutation)
        .zip(burst_mutation)
    {
        *v = av * alpha + bv * (1.0 - alpha);
        match mutation_type {
            1 => {
                *v += smooth;
            }
            2 => {
                *v += burst;
            }
            _ => {}
        }
    }

    Ok(())
}
