use flume::bounded;
use rand::distr::Distribution;
use wgpu::util::{BufferInitDescriptor, DeviceExt};

struct LayerParams {
    input_size: u32,
    output_size: u32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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

    let rng = rand::rng();
    let distr = rand::distr::Uniform::new(-1.0, 1.0)?;

    let params = LayerParams {
        input_size: 768,
        output_size: 512,
    };

    let input_data: Vec<f32> = distr
        .sample_iter(rng.clone())
        .take(params.input_size as usize)
        .collect();
    let biases_data: Vec<f32> = (0..params.input_size).map(|v| v as f32).collect();
    let weights_data: Vec<f32> = distr
        .sample_iter(rng)
        .take(params.input_size as usize * params.output_size as usize)
        .collect();

    let input_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("input"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });

    let biases_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("biases"),
        contents: bytemuck::cast_slice(&biases_data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });

    let weights_buffer = device.create_buffer_init(&BufferInitDescriptor {
        label: Some("weights"),
        contents: bytemuck::cast_slice(&weights_data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });

    let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("temp"),
        size: params.output_size as u64 * std::mem::size_of::<f32>() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: params.output_size as u64 * std::mem::size_of::<f32>() as u64,
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

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        // We specified 64 threads per workgroup in the shader, so we need to compute how many
        // workgroups we need to dispatch.
        let num_dispatches = (params.output_size).div_ceil(64) as u32;

        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_dispatches, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &temp_buffer, 0, output_buffer.size());

    queue.submit([encoder.finish()]);

    {
        // The mapping process is async, so we'll need to create a channel to get
        // the success flag for our mapping
        let (tx, rx) = bounded(1);

        // We send the success or failure of our mapping via a callback
        temp_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap()
        });

        // The callback we submitted to map async will only get called after the
        // device is polled or the queue submitted
        device.poll(wgpu::PollType::wait_indefinitely())?;

        // We check if the mapping was successful here
        rx.recv_async().await??;

        // We then get the bytes that were stored in the buffer
        let output_data = temp_buffer.get_mapped_range(..);

        // Now we have the data on the CPU we can do what ever we want to with it
        let output_slice: &[f32] = bytemuck::cast_slice(&output_data);
        println!("input = {input_data:?}");
        println!("weights = {weights_data:?}");
        println!("output = {output_slice:?}");

        println!(
            "input len = {}, weights len = {}, output len = {}",
            input_data.len(),
            weights_data.len(),
            output_slice.len()
        );
    }

    // We need to unmap the buffer to be able to use it again
    temp_buffer.unmap();

    Ok(())
}
