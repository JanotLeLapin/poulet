use std::sync::Arc;

use poulet_ai_common::decode_move;
use poulet_chess::Game;
use rand_distr::Distribution;

use flume::bounded;
use wgpu::{
    BufferDescriptor,
    util::{BufferInitDescriptor, DeviceExt},
};

pub struct State {
    pub instance: Arc<wgpu::Instance>,
    pub adapter: Arc<wgpu::Adapter>,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub shader: Arc<wgpu::ShaderModule>,
    pub pipeline: Arc<wgpu::ComputePipeline>,
}

pub struct DenseLayer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pipeline: Arc<wgpu::ComputePipeline>,
    weights_buffer: wgpu::Buffer,
    biases_buffer: wgpu::Buffer,
    input_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    temp_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    output_size: usize,
}

pub struct Player {
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

        Self {
            instance: Arc::new(instance),
            adapter: Arc::new(adapter),
            device: Arc::new(device),
            queue: Arc::new(queue),
            shader: Arc::new(shader),
            pipeline: Arc::new(pipeline),
        }
    }
}

impl DenseLayer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        pipeline: Arc<wgpu::ComputePipeline>,
        weights: &[f32],
        biases: &[f32],
        input_size: usize,
        output_size: usize,
    ) -> Self {
        let biases_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("biases"),
            contents: bytemuck::cast_slice(&biases),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let weights_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("weights"),
            contents: bytemuck::cast_slice(&weights),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        });

        let input_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("input"),
            size: (input_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let temp_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("temp"),
            size: output_size as u64 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size as u64 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
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
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline,
            weights_buffer,
            biases_buffer,
            input_buffer,
            output_buffer,
            temp_buffer,
            bind_group,
            output_size,
        }
    }

    pub async fn forward(&self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        let mut encoder = self.device.create_command_encoder(&Default::default());

        self.queue
            .write_buffer(&self.input_buffer, 0, bytemuck::cast_slice(&input));

        {
            let num_dispatches = self.output_size.div_ceil(64) as u32;

            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(num_dispatches, 1, 1);
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
            let (tx, rx) = bounded(1);

            self.temp_buffer
                .map_async(wgpu::MapMode::Read, .., move |result| {
                    tx.send(result).unwrap()
                });

            self.device.poll(wgpu::PollType::wait_indefinitely())?;

            rx.recv_async().await??;

            let output_data = self.temp_buffer.get_mapped_range(..);

            bytemuck::cast_slice(&output_data).to_vec()
        };

        self.temp_buffer.unmap();

        Ok(res)
    }
}

impl Player {
    pub fn new(state: &State) -> anyhow::Result<Self> {
        let rng = rand::rng();
        let distr_a = rand_distr::Normal::new(0.0, (2.0 / 768.0f32).sqrt())?;
        let distr_b = rand_distr::Normal::new(0.0, (2.0 / 512.0f32).sqrt())?;

        let weights_data_a: Vec<f32> = distr_a.sample_iter(rng.clone()).take(768 * 512).collect();
        let biases_data_a: Vec<f32> = (0..512).map(|_| 0.0 as f32).collect();

        let weights_data_b: Vec<f32> = distr_b.sample_iter(rng.clone()).take(512 * 512).collect();
        let biases_data_b: Vec<f32> = (0..512).map(|_| 0.0 as f32).collect();

        let weights_data_output: Vec<f32> =
            distr_b.sample_iter(rng.clone()).take(512 * 4096).collect();
        let biases_data_output: Vec<f32> = (0..4096).map(|_| 0.0 as f32).collect();

        let hidden_layer_a = DenseLayer::new(
            state.device.clone(),
            state.queue.clone(),
            state.pipeline.clone(),
            &weights_data_a,
            &biases_data_a,
            768,
            512,
        );

        let hidden_layer_b = DenseLayer::new(
            state.device.clone(),
            state.queue.clone(),
            state.pipeline.clone(),
            &weights_data_b,
            &biases_data_b,
            512,
            512,
        );

        let output_layer = DenseLayer::new(
            state.device.clone(),
            state.queue.clone(),
            state.pipeline.clone(),
            &weights_data_output,
            &biases_data_output,
            512,
            4096,
        );

        Ok(Self {
            hidden_layer_a,
            hidden_layer_b,
            output_layer,
        })
    }

    pub async fn forward(&self, input: &[f32]) -> anyhow::Result<Vec<f32>> {
        let out = self.hidden_layer_a.forward(input).await?;
        let out = self.hidden_layer_b.forward(&out).await?;
        let out = self.output_layer.forward(&out).await?;
        Ok(out)
    }
}

pub fn predict_move(game: &mut Game, mut logits: Vec<f32>) -> (f32, u8, u8, u8, u8) {
    for i in 0..4096 {
        let (src_file, src_rank, dst_file, dst_rank) = decode_move(i);
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
    let exp_sum: f32 = logits.iter().map(|v| (v - max).exp()).sum();
    logits
        .iter_mut()
        .for_each(|v| *v = (*v - max).exp() / exp_sum);

    let mut rng = rand::rng();
    let distr = rand::distr::weighted::WeightedIndex::new(&logits).unwrap();
    let i = distr.sample(&mut rng);

    let (src_file, src_rank, dst_file, dst_rank) = decode_move(i);

    (logits[i], src_file, src_rank, dst_file, dst_rank)
}
