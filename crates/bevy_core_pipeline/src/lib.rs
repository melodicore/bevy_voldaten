#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![forbid(unsafe_code)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

pub mod blit;
pub mod core_2d;
pub mod core_3d;
pub mod deferred;
pub mod fullscreen_material;
pub mod mip_generation;
pub mod oit;
pub mod prepass;
pub mod schedule;
pub mod skybox;
pub mod tonemapping;
pub mod upscaling;

use bevy_ecs::schedule::IntoScheduleConfigs;
pub use bevy_light::Skybox;
pub use fullscreen_vertex_shader::FullscreenShader;
pub use schedule::{Core2d, Core2dSystems, Core3d, Core3dSystems};

mod fullscreen_vertex_shader;

use crate::schedule::{
    camera_driver, handle_uncovered_swap_chains, submit_pending_command_buffers,
};
use crate::{
    blit::BlitPlugin, core_2d::Core2dPlugin, core_3d::Core3dPlugin,
    deferred::copy_lighting_id::CopyDeferredLightingIdPlugin, mip_generation::MipGenerationPlugin,
    prepass::BackgroundMotionVectorsPlugin, tonemapping::TonemappingPlugin,
    upscaling::UpscalingPlugin,
};
use bevy_app::{App, Plugin};
use bevy_asset::embedded_asset;
use bevy_render::renderer::{RenderGraph, RenderGraphSystems};
use bevy_render::RenderApp;
use oit::OrderIndependentTransparencyPlugin;

pub struct CorePipelinePlugin {
    /// Forwarded to [`Core3dPlugin::enable_deferred_prepass`] — see that field's doc comment.
    pub enable_deferred_prepass: bool,
    /// Controls if [`CopyDeferredLightingIdPlugin`] is added. Only meaningful alongside deferred
    /// rendering (see `enable_deferred_prepass`) — its own node is a plain `ViewQuery` (not
    /// gated on `DeferredPrepass` the way the other deferred nodes are) but still early-returns
    /// cheaply with no deferred prepass in play, so disabling it only saves the per-view executor
    /// hand-off, not any real body cost.
    pub enable_copy_deferred_lighting_id: bool,
    /// Controls if [`OrderIndependentTransparencyPlugin`] is added.
    ///
    /// **History, for anyone re-touching this:** first found to be an unsafe disable, confirmed
    /// by testing — same hazard class `bevy_pbr::PbrPlugin::enable_screen_space_reflections` hit:
    /// `bevy_pbr::render::mesh_view_bindings::prepare_mesh_view_bind_groups` unconditionally read
    /// `Res<OitBuffers>` regardless of whether any view actually used OIT, panicking at runtime
    /// ("Resource does not exist") the first time any mesh was rendered. **Fixed**: that param is
    /// now `Option<Res<OitBuffers>>`, and its per-view usage (already correctly gated behind the
    /// view's own optional `OrderIndependentTransparencySettingsOffset` component, which is never
    /// present without this plugin) additionally checks the buffer is `Some` before touching it
    /// — mirrors the pre-existing `atmosphere_buffer`/`atmosphere_sampler` `Option<Res<_>>`
    /// pattern in that same function.
    pub enable_oit: bool,
    /// Controls if [`MipGenerationPlugin`] is added. Its `ViewDepthPyramid` output is consumed
    /// by GPU-driven occlusion culling and meshlets.
    ///
    /// **History, for anyone re-touching this:** first found to be an unsafe disable, confirmed
    /// by testing — its `DownsampleShaders`/`DownsamplingConstants` resources were *also* read
    /// unconditionally by `bevy_pbr::light_probe::generate::initialize_generated_environment_map_resources`
    /// (`LightProbePlugin`, always registered by `PbrPlugin`), regardless of whether any light
    /// probe with generated environment maps actually existed in the scene — panicking at
    /// startup ("Resource does not exist" for `Res<DownsampleShaders>`). **Fixed**: that
    /// system's `downsample_shaders` param is now `Option<Res<DownsampleShaders>>`, with an
    /// early return when absent — there is nothing for it to set up without the shaders that
    /// only this plugin provides, so a project with no light probe needing a generated
    /// environment map (this one included) safely skips the whole system.
    ///
    /// **Second, related panic found after the above fix** (same module, still confirmed by
    /// testing): with `initialize_generated_environment_map_resources` skipping its work, three
    /// *more* resources it would otherwise have inserted (`GeneratorBindGroupLayouts`,
    /// `GeneratorSamplers`, `DownsamplingConfig`) also never exist — and
    /// `prepare_generated_environment_map_bind_groups` read all three as bare `Res<_>`, panicking
    /// at *runtime* this time (its `light_probes: Query<_>` param is always valid even when
    /// empty, so that alone wasn't the blocker). **Fixed the same way**: all three are now
    /// `Option<Res<_>>` with an early return, mirroring `GeneratorPipelines`'s own pre-existing
    /// `Option<Res<_>>` + early-return idiom already used elsewhere in the same file
    /// (`downsampling_system`/`filtering_system`) — this fix pattern was already established in
    /// this exact module before any of ours, we just hadn't found every consumer yet.
    pub enable_mip_generation: bool,
}

impl Default for CorePipelinePlugin {
    fn default() -> Self {
        Self {
            enable_deferred_prepass: true,
            enable_copy_deferred_lighting_id: true,
            enable_oit: true,
            enable_mip_generation: true,
        }
    }
}

impl Plugin for CorePipelinePlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "fullscreen_vertex_shader/fullscreen.wgsl");

        app.add_plugins((
            Core2dPlugin,
            Core3dPlugin {
                enable_deferred_prepass: self.enable_deferred_prepass,
            },
        ))
        .add_plugins((BlitPlugin, TonemappingPlugin, UpscalingPlugin));

        if self.enable_copy_deferred_lighting_id {
            app.add_plugins(CopyDeferredLightingIdPlugin);
        }
        if self.enable_oit {
            app.add_plugins(OrderIndependentTransparencyPlugin);
        }
        if self.enable_mip_generation {
            app.add_plugins(MipGenerationPlugin);
        }
        // Always on — TAA (bevy_anti_alias::taa) requires accurate motion vectors for background
        // (depth == 0) pixels or it ghosts/smears on any camera rotation over the background; see
        // this plugin's own doc comment.
        app.add_plugins(BackgroundMotionVectorsPlugin);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app.init_resource::<FullscreenShader>().add_systems(
            RenderGraph,
            (
                camera_driver.in_set(RenderGraphSystems::Render),
                (submit_pending_command_buffers, handle_uncovered_swap_chains)
                    .chain()
                    .in_set(RenderGraphSystems::Submit),
            ),
        );
    }
}
