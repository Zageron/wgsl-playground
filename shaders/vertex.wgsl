struct VertexInput {
    @location(0) position: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Uniforms {
    tex_size: vec2<f32>,
    resolution: vec2<f32>,
};

struct InstanceInput {
    @location(1) column_offset: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(in: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;
    var pos = in.position / uniforms.resolution * uniforms.tex_size;
    var uv = floor(pos);

    var clamp_a = clamp((1.0 - fract(pos)).x, 0.0, 1.0);
    var clamp_b = clamp((1.0 - fract(pos)).y, 0.0, 1.0);

    var normalized_offset = instance.column_offset / uniforms.resolution.x;
    uv = uv + 1.0 - vec2<f32>(clamp_a, clamp_b);
    uv.x = uv.x + normalized_offset;
    out.position = vec4<f32>(uv, 0.0, 1.0);
    out.uv = fma(in.position, vec2<f32>(0.5, -0.5), vec2<f32>(0.5, 0.5));

    return out;
}
