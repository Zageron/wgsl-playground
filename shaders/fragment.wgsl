struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

struct Uniforms {
    index: u32,
};

@group(0) @binding(1)
var<uniform> uniforms: Uniforms;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var x_loc: f32 = 64.0 / 256.0;
    var y_loc: f32 = 64.0 / 192.0;

    var x_idx = uniforms.index / 3u;
    var y_idx = uniforms.index % 3u;

    var uv = in.uv;
    uv.x = (uv.x * x_loc) + (f32(x_idx) * x_loc);
    uv.y = (uv.y * y_loc) + (f32(y_idx) * y_loc);
    return textureSample(t_diffuse, s_diffuse, uv);
}
