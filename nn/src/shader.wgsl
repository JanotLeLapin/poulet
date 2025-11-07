@group(0) @binding(0) var<storage, read> x1: array<u32>;
@group(0) @binding(1) var<storage, read> x2: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

@compute
@workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_invocation_id: vec3<u32>
) {
    let index = global_invocation_id.x;
    output[index] = x1[index] + x2[index];
}
