import package::fullscreen_vertex_shader::FullscreenVertexOutput;
@if(SRGB_TO_LINEAR) import bevy_render::color_operations::srgb_to_linear;
@if(OKLAB_TO_LINEAR) import bevy_render::color_operations::oklab_to_linear_rgb;

@group(0) @binding(0) var in_texture: texture_2d<f32>;
@group(0) @binding(1) var in_sampler: sampler;
// The brightness of `1.0` in nits, in `x`, for a window that shows HDR output.
@if(HDR_OUTPUT) @group(0) @binding(2) var<uniform> paper_white: vec4<f32>;

@if(HDR_PQ) {
// The linear BT.709 colors as linear BT.2020 colors.
const BT709_TO_BT2020 = mat3x3<f32>(
    vec3<f32>(0.6274039, 0.0690973, 0.0163914),
    vec3<f32>(0.3292830, 0.9195404, 0.0880133),
    vec3<f32>(0.0433131, 0.0113623, 0.8955953),
);

// The PQ (SMPTE ST 2084) encoding of absolute luminances in nits.
fn pq_encode(nits: vec3<f32>) -> vec3<f32> {
    let y = clamp(nits / 10000.0, vec3<f32>(0.0), vec3<f32>(1.0));
    let p = pow(y, vec3<f32>(0.1593017578125));
    return pow((vec3<f32>(0.8359375) + 18.8515625 * p) / (vec3<f32>(1.0) + 18.6875 * p), vec3<f32>(78.84375));
}
}

@fragment
fn fs_main(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    var color = textureSample(in_texture, in_sampler, in.uv);
    @if(SRGB_TO_LINEAR)
    color = vec4(srgb_to_linear(color.rgb), color.a);
    @if(OKLAB_TO_LINEAR)
    color = vec4(oklab_to_linear_rgb(color.rgb), color.a);
    @if(HDR_SCRGB)
    color = vec4(color.rgb * (paper_white.x / 80.0), color.a);
    @if(HDR_PQ)
    color = vec4(pq_encode(max(BT709_TO_BT2020 * color.rgb, vec3<f32>(0.0)) * paper_white.x), color.a);
    return color;
}
