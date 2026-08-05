#define_import_path bevy_pbr::light_falloff

// Extension point for shaping point/spot light distance attenuation beyond
// plain inverse-square falloff (e.g. capping intensity near the light with a
// soft knee). The default here is a no-op passthrough.
//
// A downstream project can override this behaviour without patching this
// crate: register your own `Shader` asset whose source also declares
// `#define_import_path bevy_pbr::light_falloff` and defines a function with
// this exact name and signature. Bevy's shader cache resolves `#import`s by
// this logical path string, not by which crate/asset supplied it, so the
// last such module registered (e.g. via `load_shader_library!` in a plugin
// added after `PbrPlugin`) wins and this default is shadowed everywhere
// `apply_light_falloff` is called — no changes to bevy_pbr required.
fn apply_light_falloff(x: f32) -> f32 {
    return x;
}
