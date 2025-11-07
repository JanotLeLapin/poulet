@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> biases: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute
@workgroup_size(64)
fn dense(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let id = global_invocation_id.x;
    let input_size = arrayLength(&input);
    let output_size = arrayLength(&output);

    if (id >= output_size) {
        return;
    }

    var sum = 0.0;
    for (var i = 0u; i < input_size; i = i + 1u) {
        let wi = id * input_size + i;
        sum = sum + input[i] * weights[wi];
    }

    sum = sum + biases[id];
    output[id] = max(0.0, sum);
}
