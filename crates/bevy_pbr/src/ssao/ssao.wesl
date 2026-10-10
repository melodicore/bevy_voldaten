//! Visibility Bitmask Ambient Occlusion (VBAO)
//! Paper: ttps://ar5iv.labs.arxiv.org/html/2301.11376
//!
//! Source code heavily based on XeGTAO v1.30 from Intel
//! https://github.com/GameTechDev/XeGTAO/blob/0d177ce06bfa642f64d8af4de1197ad1bcb862d4/Source/Rendering/Shaders/XeGTAO.hlsli
//!
//! Source code based on the existing XeGTAO implementation and
//! https://cdrinmatane.github.io/posts/ssaovb-code/
//!
//! Source code base on SSRT3 implementation
//! https://github.com/cdrinmatane/SSRT3

import bevy_render::{
    view::View,
    globals::Globals,
    maths::{PI, HALF_PI},
};

@group(0) @binding(0) var preprocessed_depth: texture_2d<f32>;
@group(0) @binding(1) var normals: texture_2d<f32>;
@group(0) @binding(2) var hilbert_index_lut: texture_2d<u32>;
@if(USE_R16FLOAT)
@group(0) @binding(3) var ambient_occlusion: texture_storage_2d<r16float, write>;
@else
@group(0) @binding(3) var ambient_occlusion: texture_storage_2d<r32float, write>;
@group(0) @binding(4) var depth_differences: texture_storage_2d<r32uint, write>;
@group(0) @binding(5) var<uniform> globals: Globals;
struct SsaoSettings {
    radius: f32,
    thickness: f32,
}
@group(0) @binding(6) var<uniform> settings: SsaoSettings;
@group(1) @binding(0) var point_clamp_sampler: sampler;
@group(1) @binding(1) var linear_clamp_sampler: sampler;
@group(1) @binding(2) var<uniform> view: View;

fn load_noise(pixel_coordinates: vec2<i32>) -> vec2<f32> {
    var index = textureLoad(hilbert_index_lut, pixel_coordinates % 64, 0).r;

@if(TEMPORAL_JITTER)
    index += 288u * (globals.frame_count % 64u);

    // R2 sequence - http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences
    return fract(0.5 + f32(index) * vec2<f32>(0.75487766624669276005, 0.5698402909980532659114));
}

/// Calculate differences in depth between neighbor pixels (later used by the spatial denoiser pass to preserve object edges)
fn calculate_neighboring_depth_differences(pixel_coordinates: vec2<i32>) -> f32 {
    // Sample the pixel's depth and 4 depths around it
    let uv = vec2<f32>(pixel_coordinates) / view.viewport.zw;
    let depths_upper_left = textureGather(0, preprocessed_depth, point_clamp_sampler, uv);
    let depths_bottom_right = textureGather(0, preprocessed_depth, point_clamp_sampler, uv, vec2<i32>(1i, 1i));
    let depth_center = depths_upper_left.y;
    let depth_left = depths_upper_left.x;
    let depth_top = depths_upper_left.z;
    let depth_bottom = depths_bottom_right.x;
    let depth_right = depths_bottom_right.z;

    // Calculate the depth differences (large differences represent object edges)
    var edge_info = vec4<f32>(depth_left, depth_right, depth_top, depth_bottom) - depth_center;
    let slope_left_right = (edge_info.y - edge_info.x) * 0.5;
    let slope_top_bottom = (edge_info.w - edge_info.z) * 0.5;
    let edge_info_slope_adjusted = edge_info + vec4<f32>(slope_left_right, -slope_left_right, slope_top_bottom, -slope_top_bottom);
    edge_info = min(abs(edge_info), abs(edge_info_slope_adjusted));
    let bias = 0.25; // Using the bias and then saturating nudges the values a bit
    let scale = depth_center * 0.011; // Weight the edges by their distance from the camera
    edge_info = saturate((1.0 + bias) - edge_info / scale); // Apply the bias and scale, and invert edge_info so that small values become large, and vice versa

    // Pack the edge info into the texture
    let edge_info_packed = vec4<u32>(pack4x8unorm(edge_info), 0u, 0u, 0u);
    textureStore(depth_differences, pixel_coordinates, edge_info_packed);

    return depth_center;
}

fn load_normal_view_space(uv_normals: vec2<f32>) -> vec3<f32> {
    var world_normal = textureSampleLevel(normals, point_clamp_sampler, uv_normals, 0.0).xyz;
    world_normal = (world_normal * 2.0) - 1.0;
    let view_from_world = mat3x3<f32>(
        view.view_from_world[0].xyz,
        view.view_from_world[1].xyz,
        view.view_from_world[2].xyz,
    );
    return view_from_world * world_normal;
}

fn reconstruct_view_space_position(depth: f32, uv: vec2<f32>) -> vec3<f32> {
    let clip_xy = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
    let t = view.view_from_clip * vec4<f32>(clip_xy, depth, 1.0);
    let view_xyz = t.xyz / t.w;
    return view_xyz;
}

fn load_and_reconstruct_view_space_position(uv: vec2<f32>, sample_mip_level: f32) -> vec3<f32> {
    let depth = textureSampleLevel(preprocessed_depth, linear_clamp_sampler, uv, sample_mip_level).r;
    return reconstruct_view_space_position(depth, uv);
}

// Occlusion strength by how far the occluder rises above the pixel's own tangent plane (view-space
// units): a sample at the plane occludes at `OCCLUDER_MIN_STRENGTH`, ramping (smoothstep) to full
// strength at `OCCLUDER_FULL_HEIGHT`. This softens shallow self-occlusion (gentle creases, small
// steps and bumps) while a real occluder such as the other wall of a corner keeps full strength.
// Height is measured along the normal, not the view direction, so a flat face receding from the
// camera doesn't count as a depth change. Each sector takes the strongest sample covering it,
// quantized to `STRENGTH_LEVELS` bitmasks (mask `k` holds the sectors of strength > k / levels).
const OCCLUDER_FULL_HEIGHT = 0.375;
const OCCLUDER_MIN_STRENGTH = 1.0 / 3.0;
const STRENGTH_LEVELS = 6u;

fn updateSectors(
    min_horizon: f32,
    max_horizon: f32,
    samples_per_slice: f32,
    bitmask: u32,
) -> u32 {
    let start_horizon = u32(min_horizon * samples_per_slice);
    // A sample on the pixel's own tangent plane lands on a sector boundary, so `ceil` would turn
    // any error there into a whole occluded sector. The prepass normal (`Rgb10a2Unorm`, world
    // space) is quantized to ~1e-3 rad, and that error changes as an object rotates, so flat faces
    // flickered between occluded and not. Spans under `SECTOR_TOLERANCE` of a sector are dropped.
    const SECTOR_TOLERANCE = 0.05;
    let angle_horizon = u32(max(ceil((max_horizon - min_horizon) * samples_per_slice - SECTOR_TOLERANCE), 0.0));

    return insertBits(bitmask, 0xFFFFFFFFu, start_horizon, angle_horizon);
}

fn processSample(
    delta_position: vec3<f32>,
    view_vec: vec3<f32>,
    pixel_normal: vec3<f32>,
    sampling_direction: f32,
    n: vec2<f32>,
    samples_per_slice: f32,
    bitmasks: ptr<function, array<u32, STRENGTH_LEVELS>>,
) {
    let delta_position_back_face = delta_position - view_vec * settings.thickness;

    // Exact `acos`, not `fast_acos`: a sample on the pixel's own tangent plane lands exactly on a
    // sector boundary, and `updateSectors`' `ceil` turns any positive error there into a whole
    // occluded sector. `fast_acos`'s error (up to ~9e-3 rad, varying with the argument) doesn't
    // cancel between this and `n`, so flat surfaces self-occluded.
    var front_back_horizon = vec2(
        acos(clamp(dot(normalize(delta_position), view_vec), -1.0, 1.0)),
        acos(clamp(dot(normalize(delta_position_back_face), view_vec), -1.0, 1.0)),
    );

    front_back_horizon = saturate(fma(vec2(sampling_direction), -front_back_horizon * (1.0 / PI), n));
    front_back_horizon = select(front_back_horizon.xy, front_back_horizon.yx, sampling_direction >= 0.0);

    let sectors = updateSectors(front_back_horizon.x, front_back_horizon.y, samples_per_slice, 0u);
    let height = dot(delta_position, pixel_normal);
    let strength = mix(OCCLUDER_MIN_STRENGTH, 1.0, smoothstep(0.0, OCCLUDER_FULL_HEIGHT, height));
    let levels = u32(round(strength * f32(STRENGTH_LEVELS)));
    // Constant trip count, so the loop unrolls and the masks stay in registers (a strength-dependent
    // bound makes the array index dynamic, which can push the array out to scratch memory).
    for (var level = 0u; level < STRENGTH_LEVELS; level += 1u) {
        (*bitmasks)[level] |= select(0u, sectors, level < levels);
    }
}

@compute
@workgroup_size(8, 8, 1)
fn ssao(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let slice_count = f32(constants::SLICE_COUNT);
    let samples_per_slice_side = f32(constants::SAMPLES_PER_SLICE_SIDE);

    let pixel_coordinates = vec2<i32>(global_id.xy);
    let uv = (vec2<f32>(pixel_coordinates) + 0.5) / view.viewport.zw;

    var pixel_depth = calculate_neighboring_depth_differences(pixel_coordinates);
    pixel_depth += 0.00001; // Avoid depth precision issues

    let pixel_position = reconstruct_view_space_position(pixel_depth, uv);
    let normals_size = vec2<f32>(textureDimensions(normals));
    let uv_normals = (vec2<f32>(pixel_coordinates) + 0.5 + view.viewport.xy) / normals_size;
    let pixel_normal = load_normal_view_space(uv_normals);
    let view_vec = normalize(-pixel_position);

    let noise = load_noise(pixel_coordinates);
    let safe_radius = max(settings.radius, 0.0001);
    // Per-axis UV radius. A single (x-derived) scale for both axes makes the sampled screen line
    // anisotropic on a non-square view, so diagonal slices sample off their own slice plane.
    let sample_scale = (-0.5 * safe_radius * vec2<f32>(view.clip_from_view[0][0], view.clip_from_view[1][1])) / pixel_position.z;

    var visibility = 0.0;
    var occluded_sample_count = 0.0;
    for (var slice_t = 0.0; slice_t < slice_count; slice_t += 1.0) {
        let slice = slice_t + noise.x;
        let phi = (PI / slice_count) * slice;
        let omega = vec2<f32>(cos(phi), sin(phi));

        let direction = vec3<f32>(omega.xy, 0.0);
        let orthographic_direction = direction - (dot(direction, view_vec) * view_vec);
        // Normalized: off-center `direction` isn't perpendicular to `view_vec`, and an
        // unnormalized axis projects the normal onto the wrong plane.
        let axis = normalize(cross(direction, view_vec));
        let projected_normal = pixel_normal - axis * dot(pixel_normal, axis);
        let projected_normal_length = length(projected_normal);

        let sign_norm = sign(dot(orthographic_direction, projected_normal));
        let cos_norm = saturate(dot(projected_normal, view_vec) / projected_normal_length);
        let n = vec2((HALF_PI - sign_norm * acos(cos_norm)) * (1.0 / PI));

        var bitmasks = array<u32, STRENGTH_LEVELS>();

        let sample_mul = vec2<f32>(omega.x, -omega.y) * sample_scale;
        for (var sample_t = 0.0; sample_t < samples_per_slice_side; sample_t += 1.0) {
            var sample_noise = (slice_t + sample_t * samples_per_slice_side) * 0.6180339887498948482;
            sample_noise = fract(noise.y + sample_noise);

            var s = (sample_t + sample_noise) / samples_per_slice_side;
            s *= s; // https://github.com/GameTechDev/XeGTAO#sample-distribution
            let sample = s * sample_mul;

            // * view.viewport.zw gets us from [0, 1] to [0, viewport_size], which is needed for this to get the correct mip levels
            let sample_mip_level = clamp(log2(length(sample * view.viewport.zw)) - 3.3, 0.0, 5.0); // https://github.com/GameTechDev/XeGTAO#memory-bandwidth-bottleneck
            let sample_position_1 = load_and_reconstruct_view_space_position(uv + sample, sample_mip_level);
            let sample_position_2 = load_and_reconstruct_view_space_position(uv - sample, sample_mip_level);

            let sample_difference_1 = sample_position_1 - pixel_position;
            let sample_difference_2 = sample_position_2 - pixel_position;

            processSample(sample_difference_1, view_vec, pixel_normal, -1.0, n, samples_per_slice_side * 2.0, &bitmasks);
            processSample(sample_difference_2, view_vec, pixel_normal, 1.0, n, samples_per_slice_side * 2.0, &bitmasks);
        }

        var occluded_levels = 0u;
        for (var level = 0u; level < STRENGTH_LEVELS; level += 1u) {
            occluded_levels += countOneBits(bitmasks[level]);
        }
        occluded_sample_count += f32(occluded_levels) / f32(STRENGTH_LEVELS);
    }

    visibility = 1.0 - occluded_sample_count / (slice_count * 2.0 * samples_per_slice_side);

    visibility = clamp(visibility, 0.03, 1.0);

    textureStore(ambient_occlusion, pixel_coordinates, vec4<f32>(visibility, 0.0, 0.0, 0.0));
}
