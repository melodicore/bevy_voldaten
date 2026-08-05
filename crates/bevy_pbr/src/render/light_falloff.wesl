// Extension point for shaping point/spot light distance attenuation beyond
// plain inverse-square falloff (e.g. capping intensity near the light with a
// soft knee). The default here is a no-op passthrough.
//
// A downstream project can override this behaviour without patching this
// crate: register your own WESL `Shader` asset whose path is
// `embedded://bevy_pbr/render/light_falloff.wesl` (so its module path is
// `bevy_pbr::render::light_falloff`) and which defines a function with this
// exact name and signature. Bevy's shader cache resolves imports by module
// path, not by which crate/asset supplied it, so the last such module
// registered wins and this default is shadowed everywhere
// `apply_light_falloff` is called — no changes to bevy_pbr required.
fn apply_light_falloff(x: f32) -> f32 {
    return x;
}
