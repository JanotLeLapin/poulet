struct Params {
    input_size: u32,
    output_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> biases: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> input_batch: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_batch: array<f32>;

@compute
@workgroup_size(64, 1, 1)
fn dense(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let output_neuron_idx = global_id.x;
    let batch_idx = global_id.y;

    let input_size = params.input_size;
    let output_size = params.output_size;

    if (output_neuron_idx >= output_size) {
        return;
    }

    let weights_start_offset = output_neuron_idx * input_size;
    let bias_idx = output_neuron_idx;

    let input_start_offset = batch_idx * input_size;
    let output_idx = batch_idx * output_size + output_neuron_idx;

    var sum = 0.0;
    for (var i = 0u; i < input_size; i = i + 1u) {
        sum = sum + input_batch[input_start_offset + i] * weights[weights_start_offset + i];
    }

    sum = sum + biases[bias_idx];
    output_batch[output_idx] = max(0.0, sum);
}
