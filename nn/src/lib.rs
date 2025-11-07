use std::sync::Arc;

use flume::bounded;
use poulet_chess::{Board, Color, Piece, PieceType};
use wgpu::{
    BufferDescriptor,
    util::{BufferInitDescriptor, DeviceExt},
};

pub struct State {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub shader: wgpu::ShaderModule,
    pub pipeline: wgpu::ComputePipeline,
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
            instance,
            adapter,
            device,
            queue,
            shader,
            pipeline,
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

pub fn encode_board(board: &Board, neurons: &mut Vec<f32>) {
    let piece_iter = vec![Color::White, Color::Black].into_iter().flat_map(|c| {
        vec![
            PieceType::Pawn,
            PieceType::Bishop,
            PieceType::Knight,
            PieceType::Rook,
            PieceType::Queen,
            PieceType::King,
        ]
        .into_iter()
        .map(move |t| Piece::new(c, t))
    });

    for (i, piece) in piece_iter.enumerate() {
        for x in 0..8 {
            for y in 0..8 {
                let neuron_idx = i * 64 + (y * 8 + x);
                neurons[neuron_idx] = if board.get_square(x as u8, y as u8) == Some(piece) {
                    1.0
                } else {
                    0.0
                };
            }
        }
    }
}
