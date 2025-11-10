use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use poulet_ai_common::decode_move;
use poulet_chess::Game;
use rand_distr::Distribution;

use serde::{Deserialize, Serialize};
use wgpu::{
    BufferDescriptor,
    util::{BufferInitDescriptor, DeviceExt},
};

pub const BATCH_SIZE: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    input_size: u32,
    output_size: u32,
    _pad: [u32; 2],
}

pub struct State {
    pub instance: Arc<wgpu::Instance>,
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub shader: Arc<wgpu::ShaderModule>,
    pub pipeline: Arc<wgpu::ComputePipeline>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct DenseLayerParams {
    input_size: usize,
    output_size: usize,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

pub struct DenseLayer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: Arc<wgpu::ComputePipeline>,
    params_buffer: wgpu::Buffer,
    weights_buffer: wgpu::Buffer,
    biases_buffer: wgpu::Buffer,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    temp_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    input_size: usize,
    output_size: usize,
}

pub struct Player {
    hidden_layer_a_params: DenseLayerParams,
    hidden_layer_b_params: DenseLayerParams,
    output_layer_params: DenseLayerParams,

    hidden_layer_a: DenseLayer,
    hidden_layer_b: DenseLayer,
    output_layer: DenseLayer,
}

impl State {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(&Default::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Introduction Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("dense"),
            compilation_options: Default::default(),
            cache: Default::default(),
        });

        let device = Arc::new(device);

        tokio::spawn({
            let device = device.clone();
            async move {
                loop {
                    let _ = device.poll(wgpu::PollType::Poll);
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            }
        });

        Self {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device,
            queue: Arc::new(queue),
            shader: Arc::new(shader),
            pipeline: Arc::new(pipeline),
        }
    }
}

impl DenseLayerParams {
    pub fn crossover_and_mutate(&mut self, other: &Self, alpha: f32, deviation: f32) {
        let mut rng = rand::rng();
        let distr = rand_distr::Normal::new(0.0, deviation).unwrap();

        let self_weights = &mut self.weights;
        let self_biases = &mut self.biases;

        let other_weights = &other.weights;
        let other_biases = &other.biases;

        for (self_params, other_params) in
            [(self_weights, other_weights), (self_biases, other_biases)]
        {
            self_params
                .iter_mut()
                .zip(other_params)
                .for_each(|(self_p, other_p)| {
                    *self_p = (*self_p * alpha + other_p * (1.0 - alpha)) + distr.sample(&mut rng);
                });
        }
    }
}

impl DenseLayer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipeline: Arc<wgpu::ComputePipeline>,
        params: &DenseLayerParams,
    ) -> Self {
        let params_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&Params {
                input_size: params.input_size as u32,
                output_size: params.output_size as u32,
                _pad: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let biases_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("biases"),
            contents: bytemuck::cast_slice(&params.biases),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let weights_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("weights"),
            contents: bytemuck::cast_slice(&params.weights),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let input_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("input"),
            size: (BATCH_SIZE * params.input_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let temp_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("temp"),
            size: (BATCH_SIZE * params.output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: (BATCH_SIZE * params.output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: biases_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline,
            params_buffer,
            weights_buffer,
            biases_buffer,
            input_buffer,
            output_buffer,
            temp_buffer,
            bind_group,
            input_size: params.input_size,
            output_size: params.output_size,
        }
    }

    pub async fn forward(&self, input: &[Vec<f32>]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut encoder = self.device.create_command_encoder(&Default::default());

        let mut input_data = Vec::with_capacity(input.len() * self.input_size as usize);
        for vec in input {
            input_data.extend_from_slice(&vec);
        }

        self.queue
            .write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(&input_data));

        {
            let num_dispatches = self.output_size.div_ceil(64) as u32;
            let batch_size = input.len() as u32;

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, batch_size, 1);
        }

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.temp_buffer,
            0,
            self.output_buffer.size(),
        );

        self.queue.submit([encoder.finish()]);

        let res = {
            let (tx, rx) = tokio::sync::oneshot::channel();

            self.temp_buffer
                .map_async(wgpu::MapMode::Read, .., move |result| {
                    tx.send(result).unwrap()
                });

            rx.await??;

            let output_data = self.temp_buffer.get_mapped_range(..);

            let slice: &[f32] = bytemuck::cast_slice(&output_data);
            slice
                .chunks(self.output_size)
                .map(|chunk| chunk.to_vec())
                .collect()
        };

        self.temp_buffer.unmap();

        Ok(res)
    }
}

impl Player {
    pub fn new(
        state: &State,
        hidden_layer_a_params: DenseLayerParams,
        hidden_layer_b_params: DenseLayerParams,
        output_layer_params: DenseLayerParams,
    ) -> Self {
        let hidden_layer_a = DenseLayer::new(
            state.device.clone(),
            state.queue.clone(),
            state.pipeline.clone(),
            &hidden_layer_a_params,
        );

        let hidden_layer_b = DenseLayer::new(
            state.device.clone(),
            state.queue.clone(),
            state.pipeline.clone(),
            &hidden_layer_b_params,
        );

        let output_layer = DenseLayer::new(
            state.device.clone(),
            state.queue.clone(),
            state.pipeline.clone(),
            &output_layer_params,
        );

        Self {
            hidden_layer_a_params,
            hidden_layer_b_params,
            output_layer_params,
            hidden_layer_a,
            hidden_layer_b,
            output_layer,
        }
    }

    pub fn init(state: &State) -> anyhow::Result<Self> {
        let rng = rand::rng();
        let distr_a = rand_distr::Normal::new(0.0, (2.0 / 768.0f32).sqrt())?;
        let distr_b = rand_distr::Normal::new(0.0, (2.0 / 512.0f32).sqrt())?;

        let hidden_layer_a_params = DenseLayerParams {
            input_size: 768,
            output_size: 512,
            weights: distr_a.sample_iter(rng.clone()).take(768 * 512).collect(),
            biases: (0..512).map(|_| 0.0 as f32).collect(),
        };

        let hidden_layer_b_params = DenseLayerParams {
            input_size: 512,
            output_size: 512,
            weights: distr_b.sample_iter(rng.clone()).take(512 * 512).collect(),
            biases: (0..512).map(|_| 0.0 as f32).collect(),
        };

        let output_layer_params = DenseLayerParams {
            input_size: 512,
            output_size: 4096,
            weights: distr_b.sample_iter(rng.clone()).take(512 * 4096).collect(),
            biases: (0..4096).map(|_| 0.0 as f32).collect(),
        };

        Ok(Self::new(
            state,
            hidden_layer_a_params,
            hidden_layer_b_params,
            output_layer_params,
        ))
    }

    pub fn load(state: &State, path: &str) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let mut layers: Vec<DenseLayerParams> = postcard::from_bytes(&bytes)?;
        layers.reverse();

        assert_eq!(3, layers.len(), "player model has 3 layers!");

        Ok(Self::new(
            state,
            layers.pop().unwrap(),
            layers.pop().unwrap(),
            layers.pop().unwrap(),
        ))
    }

    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let a = &self.hidden_layer_a_params;
        let b = &self.hidden_layer_b_params;
        let out = &self.output_layer_params;
        let layers = vec![a, b, out];
        let bytes: Vec<u8> = postcard::to_allocvec(&layers)?;
        std::fs::write(path, &bytes)?;

        Ok(())
    }

    pub fn crossover_and_mutate(
        &self,
        state: &State,
        other: &Self,
        alpha: f32,
        deviation: f32,
    ) -> Self {
        let mut a = self.hidden_layer_a_params.clone();
        let mut b = self.hidden_layer_b_params.clone();
        let mut out = self.output_layer_params.clone();
        out.crossover_and_mutate(&other.output_layer_params, alpha, deviation);
        a.crossover_and_mutate(&other.hidden_layer_a_params, alpha, deviation);
        b.crossover_and_mutate(&other.hidden_layer_b_params, alpha, deviation);

        Self::new(state, a, b, out)
    }

    pub async fn forward(&self, input: &[Vec<f32>]) -> anyhow::Result<Vec<Vec<f32>>> {
        let out = self.hidden_layer_a.forward(input).await?;
        let out = self.hidden_layer_b.forward(&out).await?;
        let out = self.output_layer.forward(&out).await?;
        Ok(out)
    }
}

pub fn decode_move_unflipped(i: usize, unflip: bool) -> (u8, u8, u8, u8) {
    if unflip {
        let (src_file, src_rank, dst_file, dst_rank) = decode_move(i);
        (src_file, 7 - src_rank, dst_file, 7 - dst_rank)
    } else {
        decode_move(i)
    }
}

pub fn predict_move(
    game: &mut Game,
    unflip: bool,
    mut logits: Vec<f32>,
    temperature: f32,
) -> Result<(f32, u8, u8, u8, u8), rand::distr::weighted::Error> {
    for i in 0..4096 {
        let (src_file, src_rank, dst_file, dst_rank) = decode_move_unflipped(i, unflip);
        if !game
            .board
            .get_square(src_file, src_rank)
            .map(|p| game.can_move(p.color))
            .unwrap_or(false)
            || !game.safe_move(
                src_file as u8,
                src_rank as u8,
                dst_file as u8,
                dst_rank as u8,
            )
        {
            logits[i] = f32::NEG_INFINITY;
        }
    }

    let max: f32 = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|v| ((v - max) / temperature).exp()).sum();
    logits
        .iter_mut()
        .for_each(|v| *v = (*v - max).exp() / exp_sum);

    let mut rng = rand::rng();
    let distr = rand::distr::weighted::WeightedIndex::new(&logits)?;
    let i = distr.sample(&mut rng);

    let (src_file, src_rank, dst_file, dst_rank) = decode_move_unflipped(i, unflip);

    Ok((logits[i], src_file, src_rank, dst_file, dst_rank))
}
