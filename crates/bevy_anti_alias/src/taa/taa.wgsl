// References:
// https://www.elopezr.com/temporal-aa-and-the-quest-for-the-holy-trail
// http://behindthepixels.io/assets/files/TemporalAA.pdf
// http://leiy.cc/publications/TAA/TAA_EG2020_Talk.pdf
// https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING

// Controls how much to blend between the current and past samples
// Lower numbers = less of the current sample and more of the past sample = more smoothing
// Values chosen empirically
const DEFAULT_HISTORY_BLEND_RATE: f32 = 0.1; // Default blend rate to use when no confidence in history
const MIN_HISTORY_BLEND_RATE: f32 = 0.015; // Minimum blend rate allowed, to ensure at least some of the current sample is used

@group(0) @binding(0) var view_target: texture_2d<f32>;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var motion_vectors: texture_2d<f32>;
@group(0) @binding(3) var depth: texture_depth_2d;
@group(0) @binding(4) var nearest_sampler: sampler;
@group(0) @binding(5) var linear_sampler: sampler;
#ifdef ALPHA
// Per-pixel confidence history. In ALPHA mode the main history's alpha channel holds the resolved
// alpha instead, so the confidence counter lives here.
@group(0) @binding(6) var history_confidence_tex: texture_2d<f32>;
#endif

struct Output {
    @location(0) view_target: vec4<f32>,
    @location(1) history: vec4<f32>,
#ifdef ALPHA
    @location(2) confidence: f32,
#endif
};

// TAA is ideally applied after tonemapping, but before post processing
// Post processing wants to go before tonemapping, which conflicts
// Solution: Put TAA before tonemapping, tonemap TAA input, apply TAA, invert-tonemap TAA output
// https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING, slide 20
// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve
fn rcp(x: f32) -> f32 { return 1.0 / x; }
fn max3(x: vec3<f32>) -> f32 { return max(x.r, max(x.g, x.b)); }
fn tonemap(color: vec3<f32>) -> vec3<f32> { return color * rcp(max3(color) + 1.0); }
fn reverse_tonemap(color: vec3<f32>) -> vec3<f32> { return color * rcp(1.0 - max3(color)); }

// The following 3 functions are from Playdead (MIT-licensed)
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
fn RGB_to_YCoCg(rgb: vec3<f32>) -> vec3<f32> {
    let y = (rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0);
    let co = (rgb.r / 2.0) - (rgb.b / 2.0);
    let cg = (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0);
    return vec3(y, co, cg);
}

fn YCoCg_to_RGB(ycocg: vec3<f32>) -> vec3<f32> {
    let r = ycocg.x + ycocg.y - ycocg.z;
    let g = ycocg.x + ycocg.z;
    let b = ycocg.x - ycocg.y - ycocg.z;
    return saturate(vec3(r, g, b));
}

fn clip_towards_aabb_center(history_color: vec3<f32>, current_color: vec3<f32>, aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;
    let v_clip = history_color - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max3(a_unit);
    if ma_unit > 1.0 {
        return p_clip + (v_clip / ma_unit);
    } else {
        return history_color;
    }
}

// ALPHA mode resolves premultiplied colour AND alpha, so the history sample carries alpha too.
fn sample_history(u: f32, v: f32) -> vec4<f32> {
    return textureSample(history, linear_sampler, vec2(u, v));
}

fn sample_view_alpha(uv: vec2<f32>) -> f32 {
    return textureSample(view_target, nearest_sampler, uv).a;
}

fn sample_view_target(uv: vec2<f32>) -> vec3<f32> {
    var sample = textureSample(view_target, nearest_sampler, uv).rgb;
#ifdef TONEMAP
    sample = tonemap(sample);
#endif
    return RGB_to_YCoCg(sample);
}

@fragment
fn taa(@location(0) uv: vec2<f32>) -> Output {
    let texture_size = vec2<f32>(textureDimensions(view_target));
    let texel_size = 1.0 / texture_size;

    // Fetch the current sample
    let original_color = textureSample(view_target, nearest_sampler, uv);
    var current_color = original_color.rgb;
    var current_alpha = original_color.a;
#ifdef TONEMAP
    current_color = tonemap(current_color);
#endif

#ifndef RESET
    // Pick the closest motion_vector from 5 samples (reduces aliasing on the edges of moving entities)
    // https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING, slide 27
    let offset = texel_size * 2.0;
    let d_uv_tl = uv + vec2(-offset.x, offset.y);
    let d_uv_tr = uv + vec2(offset.x, offset.y);
    let d_uv_bl = uv + vec2(-offset.x, -offset.y);
    let d_uv_br = uv + vec2(offset.x, -offset.y);
    var closest_uv = uv;
    let d_tl = textureSample(depth, nearest_sampler, d_uv_tl);
    let d_tr = textureSample(depth, nearest_sampler, d_uv_tr);
    var closest_depth = textureSample(depth, nearest_sampler, uv);
    let d_bl = textureSample(depth, nearest_sampler, d_uv_bl);
    let d_br = textureSample(depth, nearest_sampler, d_uv_br);
    if d_tl > closest_depth {
        closest_uv = d_uv_tl;
        closest_depth = d_tl;
    }
    if d_tr > closest_depth {
        closest_uv = d_uv_tr;
        closest_depth = d_tr;
    }
    if d_bl > closest_depth {
        closest_uv = d_uv_bl;
        closest_depth = d_bl;
    }
    if d_br > closest_depth {
        closest_uv = d_uv_br;
    }
    let closest_motion_vector = textureSample(motion_vectors, nearest_sampler, closest_uv).rg;

    // Reproject to find the equivalent sample from the past
    // Uses 5-sample Catmull-Rom filtering (reduces blurriness)
    // Catmull-Rom filtering: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
    // Ignoring corners: https://www.activision.com/cdn/research/Dynamic_Temporal_Antialiasing_and_Upsampling_in_Call_of_Duty_v4.pdf#page=68
    // Technically we should renormalize the weights since we're skipping the corners, but it's basically the same result
    let history_uv = uv - closest_motion_vector;
    let sample_position = history_uv * texture_size;
    let texel_center = floor(sample_position - 0.5) + 0.5;
    let f = sample_position - texel_center;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let texel_position_0 = (texel_center - 1.0) * texel_size;
    let texel_position_3 = (texel_center + 2.0) * texel_size;
    let texel_position_12 = (texel_center + (w2 / w12)) * texel_size;
    var history_sample = sample_history(texel_position_12.x, texel_position_0.y) * w12.x * w0.y;
    history_sample += sample_history(texel_position_0.x, texel_position_12.y) * w0.x * w12.y;
    history_sample += sample_history(texel_position_12.x, texel_position_12.y) * w12.x * w12.y;
    history_sample += sample_history(texel_position_3.x, texel_position_12.y) * w3.x * w12.y;
    history_sample += sample_history(texel_position_12.x, texel_position_3.y) * w12.x * w3.y;
    var history_color = history_sample.rgb;
#ifdef ALPHA
    var history_alpha = history_sample.a;
#endif

    // Constrain past sample with 3x3 YCoCg variance clipping (reduces ghosting)
    // YCoCg: https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING, slide 33
    // Variance clipping: https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
    let s_tl = sample_view_target(uv + vec2(-texel_size.x,  texel_size.y));
    let s_tm = sample_view_target(uv + vec2( 0.0,           texel_size.y));
    let s_tr = sample_view_target(uv + vec2( texel_size.x,  texel_size.y));
    let s_ml = sample_view_target(uv + vec2(-texel_size.x,  0.0));
    let s_mm = RGB_to_YCoCg(current_color);
    let s_mr = sample_view_target(uv + vec2( texel_size.x,  0.0));
    let s_bl = sample_view_target(uv + vec2(-texel_size.x, -texel_size.y));
    let s_bm = sample_view_target(uv + vec2( 0.0,          -texel_size.y));
    let s_br = sample_view_target(uv + vec2( texel_size.x, -texel_size.y));
    let moment_1 = s_tl + s_tm + s_tr + s_ml + s_mm + s_mr + s_bl + s_bm + s_br;
    let moment_2 = (s_tl * s_tl) + (s_tm * s_tm) + (s_tr * s_tr) + (s_ml * s_ml) + (s_mm * s_mm) + (s_mr * s_mr) + (s_bl * s_bl) + (s_bm * s_bm) + (s_br * s_br);
    let mean = moment_1 / 9.0;
    let variance = (moment_2 / 9.0) - (mean * mean);
    let std_deviation = sqrt(max(variance, vec3(0.0)));
    history_color = RGB_to_YCoCg(history_color);
    history_color = clip_towards_aabb_center(history_color, s_mm, mean - std_deviation, mean + std_deviation);
    history_color = YCoCg_to_RGB(history_color);

#ifdef ALPHA
    // Keep the reprojected alpha within the current 3x3 neighbourhood's alpha range: a moved or
    // vanished edge can't leave a ghost trail, while an edge texel whose neighbours span 0..1
    // (jitter alternately covers/uncovers it) keeps the full accumulated coverage.
    let a_tl = sample_view_alpha(uv + vec2(-texel_size.x,  texel_size.y));
    let a_tm = sample_view_alpha(uv + vec2( 0.0,           texel_size.y));
    let a_tr = sample_view_alpha(uv + vec2( texel_size.x,  texel_size.y));
    let a_ml = sample_view_alpha(uv + vec2(-texel_size.x,  0.0));
    let a_mm = original_color.a;
    let a_mr = sample_view_alpha(uv + vec2( texel_size.x,  0.0));
    let a_bl = sample_view_alpha(uv + vec2(-texel_size.x, -texel_size.y));
    let a_bm = sample_view_alpha(uv + vec2( 0.0,          -texel_size.y));
    let a_br = sample_view_alpha(uv + vec2( texel_size.x, -texel_size.y));
    let alpha_min = min(min(min(a_tl, a_tm), min(a_tr, a_ml)), min(min(a_mm, a_mr), min(a_bl, min(a_bm, a_br))));
    let alpha_max = max(max(max(a_tl, a_tm), max(a_tr, a_ml)), max(max(a_mm, a_mr), max(a_bl, max(a_bm, a_br))));
    history_alpha = clamp(history_alpha, alpha_min, alpha_max);
#endif

    // How confident we are that the history is representative of the current frame
#ifdef ALPHA
    var history_confidence = textureSample(history_confidence_tex, nearest_sampler, uv).r;
#else
    var history_confidence = textureSample(history, nearest_sampler, uv).a;
#endif
    let pixel_motion_vector = abs(closest_motion_vector) * texture_size;
    if pixel_motion_vector.x < 0.01 && pixel_motion_vector.y < 0.01 {
        // Increment when pixels are not moving
        history_confidence += 10.0;
    } else {
        // Else reset
        history_confidence = 1.0;
    }

    // Blend current and past sample
    // Use more of the history if we're confident in it (reduces noise when there is no motion)
    // https://hhoppe.com/supersample.pdf, section 4.1
    var current_color_factor = clamp(1.0 / history_confidence, MIN_HISTORY_BLEND_RATE, DEFAULT_HISTORY_BLEND_RATE);

    // Reject history when motion vectors point off screen
    if any(saturate(history_uv) != history_uv) {
        current_color_factor = 1.0;
        history_confidence = 1.0;
    }

    current_color = mix(history_color, current_color, current_color_factor);
#ifdef ALPHA
    current_alpha = mix(history_alpha, current_alpha, current_color_factor);
#endif
#endif // #ifndef RESET


    // Write output to history and view target
    var out: Output;
#ifdef RESET
    let history_confidence = 1.0 / MIN_HISTORY_BLEND_RATE;
#endif
#ifdef ALPHA
    // Premultiplied colour + resolved alpha in the history, confidence in its own texture.
    out.history = vec4(current_color, current_alpha);
    out.confidence = history_confidence;
#else
    out.history = vec4(current_color, history_confidence);
#endif
#ifdef TONEMAP
    current_color = reverse_tonemap(current_color);
#endif
#ifdef ALPHA
    out.view_target = vec4(current_color, current_alpha);
#else
    out.view_target = vec4(current_color, original_color.a);
#endif
    return out;
}
