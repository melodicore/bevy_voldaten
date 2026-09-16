#![expect(missing_docs, reason = "Not all docs are written yet, see #3492.")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![forbid(unsafe_code)]
#![doc(
    html_logo_url = "https://bevy.org/assets/icon.png",
    html_favicon_url = "https://bevy.org/assets/icon.png"
)]

extern crate alloc;

#[cfg(feature = "meshlet")]
mod meshlet;
pub mod wireframe;

/// Experimental features that are not yet finished. Please report any issues you encounter!
///
/// Expect bugs, missing features, compatibility issues, low performance, and/or future breaking changes.
#[cfg(feature = "meshlet")]
pub mod experimental {
    /// Render high-poly 3d meshes using an efficient GPU-driven method.
    /// See [`MeshletPlugin`](meshlet::MeshletPlugin) and [`MeshletMesh`](meshlet::MeshletMesh) for details.
    pub mod meshlet {
        pub use crate::meshlet::*;
    }
}

mod atmosphere;
mod cluster;
pub mod contact_shadows;
#[cfg(feature = "bevy_gltf")]
mod gltf;
use bevy_light::cluster::GlobalClusterSettings;
use bevy_render::{
    sync_component::SyncComponent,
    view::{
        RenderExtractedShadowMapVisibleEntities, RenderShadowLodOrigin,
        RenderShadowMapVisibleEntities,
    },
};
pub use contact_shadows::{
    ContactShadows, ContactShadowsBuffer, ContactShadowsPlugin, ContactShadowsUniform,
    ViewContactShadowsUniformOffset,
};
pub mod decal;
pub mod deferred;
pub mod diagnostic;
mod extended_material;
mod fog;
mod light_probe;
mod lightmap;
mod material;
mod material_bind_groups;
mod medium;
mod mesh_material;
mod parallax;
mod pbr_material;
mod prepass;
mod render;
mod ssao;
mod ssr;
mod transmission;
mod volumetric_fog;

use bevy_color::{Color, LinearRgba};

pub use atmosphere::*;
use bevy_light::{
    AmbientLight, DirectionalLight, PointLight, RectLight, ShadowFilteringMethod,
    ShadowRotationCompensation, SpotLight,
};
use bevy_shader::{load_shader_library, ShaderRef};
pub use cluster::*;
pub use decal::clustered::ClusteredDecalPlugin;
pub use extended_material::*;
pub use fog::*;
pub use light_probe::*;
pub use lightmap::*;
pub use material::*;
pub use material_bind_groups::*;
pub use medium::*;
pub use mesh_material::*;
pub use parallax::*;
pub use pbr_material::*;
pub use prepass::*;
pub use render::*;
pub use ssao::*;
pub use ssr::*;
pub use transmission::*;
pub use volumetric_fog::VolumetricFogPlugin;

/// The PBR prelude.
///
/// This includes the most common types in this crate, re-exported for your convenience.
pub mod prelude {
    #[doc(hidden)]
    pub use crate::{
        contact_shadows::ContactShadowsPlugin,
        fog::{DistanceFog, FogFalloff},
        material::{Material, MaterialPlugin},
        mesh_material::MeshMaterial3d,
        parallax::ParallaxMappingMethod,
        pbr_material::StandardMaterial,
        ssao::ScreenSpaceAmbientOcclusionPlugin,
    };
}

use crate::gpu::GpuClusteringPlugin;
use crate::{deferred::DeferredPbrLightingPlugin, gpu::extract_clusters_for_gpu_clustering};
use bevy_app::prelude::*;
use bevy_asset::{AssetApp, AssetPath, Assets, Handle, RenderAssetUsages};
use bevy_core_pipeline::mip_generation::experimental::depth::early_downsample_depth;
use bevy_core_pipeline::schedule::{Core3d, Core3dSystems, ShadowMapSchedule};
use bevy_ecs::prelude::*;
use bevy_image::{Image, ImageSampler};
use bevy_material::AlphaMode;
use bevy_render::{
    camera::sort_cameras,
    extract_resource::ExtractResourcePlugin,
    render_resource::{
        Extent3d, TextureDataOrder, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, TextureViewDescriptor, TextureViewDimension,
    },
    sync_component::SyncComponentPlugin,
    ExtractSchedule, GpuResourceAppExt, Render, RenderApp, RenderDebugFlags, RenderStartup,
    RenderSystems,
};

use std::path::PathBuf;

fn shader_ref(path: PathBuf) -> ShaderRef {
    ShaderRef::Path(AssetPath::from_path_buf(path).with_source("embedded"))
}

/// Sets up the entire PBR infrastructure of bevy.
pub struct PbrPlugin {
    /// Controls if the prepass is enabled for the [`StandardMaterial`].
    /// For more information about what a prepass is, see the [`bevy_core_pipeline::prepass`] docs.
    pub prepass_enabled: bool,
    /// Controls if [`DeferredPbrLightingPlugin`] is added.
    pub add_default_deferred_lighting_plugin: bool,
    /// Controls if GPU [`MeshUniform`] building is enabled.
    ///
    /// This requires compute shader support and so will be forcibly disabled if
    /// the platform doesn't support those.
    pub use_gpu_instance_buffer_builder: bool,
    /// Debugging flags that can optionally be set when constructing the renderer.
    pub debug_flags: RenderDebugFlags,
    /// Builds and inserts `StandardMaterial` when loading glTF files
    pub gltf_enable_standard_materials: bool,
    /// Controls if [`VolumetricFogPlugin`] (and its `Core3d` node) is added. Purely opt-in via
    /// the per-camera [`VolumetricFog`](bevy_light::VolumetricFog) component — a project that
    /// never inserts it can disable this to skip the node's per-view executor hand-off.
    pub enable_volumetric_fog: bool,
    /// Controls if [`ScreenSpaceReflectionsPlugin`] (and its `Core3d` node) is added. Purely
    /// opt-in via the per-camera `ScreenSpaceReflections` component.
    ///
    /// **History, for anyone re-touching this:** first found to be an unsafe disable, confirmed
    /// by testing — unlike volumetric fog and atmosphere,
    /// `bevy_pbr::render::mesh_view_bindings::prepare_mesh_view_bind_groups` (a core system that
    /// runs for every mesh, not just a camera carrying the `ScreenSpaceReflections` component)
    /// unconditionally read `Res<ScreenSpaceReflectionsBuffer>`, which only this plugin
    /// initializes — panicking at runtime ("Resource does not exist") the first time any mesh
    /// was rendered, regardless of whether SSR was actually used anywhere. **Fixed**: that
    /// param is now `Option<Res<_>>`, and its per-view usage (already correctly gated behind the
    /// view's own optional `ViewScreenSpaceReflectionsUniformOffset` component, which is never
    /// present without this plugin) additionally checks the buffer is `Some` before touching it
    /// — mirrors the pre-existing `atmosphere_buffer`/`atmosphere_sampler` `Option<Res<_>>`
    /// pattern in that same function.
    pub enable_screen_space_reflections: bool,
    /// Controls if [`AtmospherePlugin`] (and [`ScatteringMediumPlugin`], its sole dependent) are
    /// added. Purely opt-in via the per-camera `Atmosphere` component.
    ///
    /// **History, for anyone re-touching this:** this was first found to be a *shader*-level
    /// hazard, not a Rust resource one, unlike `enable_screen_space_reflections`/`enable_oit`
    /// below — `mesh_view_bindings.wgsl` (a shared library `#import`ed by essentially every PBR
    /// shader) had an **unconditional** `#import bevy_pbr::atmosphere::types as atmosphere_types`
    /// at the top of the file, while the import's only *usage* further down was already guarded
    /// by `#ifdef ATMOSPHERE`. Disabling this plugin without the import itself also being
    /// conditional broke every PBR pipeline's compile, confirmed by testing (every mesh in the
    /// game failed to render, only UI text visible, plus a continuous flood of failed/retried
    /// pipeline-compile log lines). **Fixed**: the import in `mesh_view_bindings.wgsl` is now
    /// itself inside `#ifdef ATMOSPHERE` (mirroring the `OIT_ENABLED`-gated import a few lines
    /// above it in the same file, and `pbr.wgsl`'s `OIT_ENABLED`-gated `oit_draw` import — an
    /// established pattern in this codebase, not a novel workaround), and `PbrPlugin::build`
    /// unconditionally calls `atmosphere::load_atmosphere_types_shader` regardless of this field
    /// (see that function's own doc comment for why — belt-and-braces so the module path stays
    /// registered even though `ATMOSPHERE` can now never be set with this field `false`, making
    /// the import itself unreachable at that point anyway).
    pub enable_atmosphere: bool,
}

impl Default for PbrPlugin {
    fn default() -> Self {
        Self {
            prepass_enabled: true,
            add_default_deferred_lighting_plugin: true,
            use_gpu_instance_buffer_builder: true,
            debug_flags: RenderDebugFlags::default(),
            gltf_enable_standard_materials: true,
            enable_volumetric_fog: true,
            enable_screen_space_reflections: true,
            enable_atmosphere: true,
        }
    }
}

/// A resource that stores the spatio-temporal blue noise texture.
#[derive(Resource)]
pub struct Bluenoise {
    /// Texture handle for spatio-temporal blue noise
    pub texture: Handle<Image>,
}

/// LTC (Linearly Transformed Cosines) LUT textures for area light shading.
///
/// It is a texture array containing 2 LUT textures:
/// The first entry encodes the 4 non-trivial elements of the inverse GGX LTC matrix.
/// The second entry encodes amplitude and Fresnel-related weights.
///
/// [LUT source and fitting code](https://github.com/selfshadow/ltc_code/blob/master/fit/results)
#[derive(Resource, Clone)]
pub struct AreaLightLuts {
    pub image: Handle<Image>,
}

// See https://github.com/bevyengine/bevy/pull/23737 for information on how the LUT was generated.
/// The split-sum approximation LUT (`F_AB`) indexed by (`NdotV`, `perceptual_roughness`).
#[derive(Resource, Clone)]
pub struct DfgLut {
    pub texture: Handle<Image>,
}

impl Plugin for PbrPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "render/pbr_types.wgsl");
        load_shader_library!(app, "render/pbr_bindings.wgsl");
        load_shader_library!(app, "render/utils.wgsl");
        load_shader_library!(app, "render/clustered_forward.wgsl");
        load_shader_library!(app, "render/light_falloff.wgsl");
        load_shader_library!(app, "render/pbr_lighting.wgsl");
        load_shader_library!(app, "render/shadows.wgsl");
        load_shader_library!(app, "deferred/pbr_deferred_types.wgsl");
        load_shader_library!(app, "deferred/pbr_deferred_functions.wgsl");
        load_shader_library!(app, "render/shadow_sampling.wgsl");
        load_shader_library!(app, "render/pbr_functions.wgsl");
        load_shader_library!(app, "render/rgb9e5.wgsl");
        load_shader_library!(app, "render/pbr_ambient.wgsl");
        load_shader_library!(app, "render/pbr_fragment.wgsl");
        load_shader_library!(app, "render/pbr.wgsl");
        load_shader_library!(app, "render/pbr_prepass_functions.wgsl");
        load_shader_library!(app, "render/pbr_prepass.wgsl");
        load_shader_library!(app, "render/parallax_mapping.wgsl");
        load_shader_library!(app, "render/view_transformations.wgsl");

        // Setup dummy shaders for when MeshletPlugin is not used to prevent shader import errors.
        load_shader_library!(app, "meshlet/dummy_visibility_buffer_resolve.wgsl");

        // Always loaded regardless of `self.enable_atmosphere` — see
        // `atmosphere::load_atmosphere_types_shader`'s own doc comment.
        load_atmosphere_types_shader(app);

        app.register_asset_reflect::<StandardMaterial>()
            .init_resource::<DefaultOpaqueRendererMethod>()
            // See `PointLightShadowAtlasReservedCapacity`'s own doc comment (`render/light.rs`) —
            // an application sets this once it knows the true total point-light count, before any
            // light has cast a shadow, so `PointLightShadowAtlasCache`'s first-ever allocation is
            // already sized for everyone.
            .init_resource::<PointLightShadowAtlasReservedCapacity>()
            // Main-world side of the cascade-fit-view override — see `CascadeViewOverride`'s own
            // doc comment (`render/light.rs`). A plain resource, not `ExtractResource` (needs
            // entity remapping on extraction, which a derive can't do) — mirrored into the
            // render world by `extract_cascade_view_override`.
            .init_resource::<CascadeViewOverride>()
            .add_plugins((
                MeshRenderPlugin {
                    use_gpu_instance_buffer_builder: self.use_gpu_instance_buffer_builder,
                    debug_flags: self.debug_flags,
                },
                MaterialsPlugin {
                    debug_flags: self.debug_flags,
                },
                MaterialPlugin::<StandardMaterial> {
                    debug_flags: self.debug_flags,
                    ..Default::default()
                },
                ScreenSpaceAmbientOcclusionPlugin,
                FogPlugin,
                ExtractResourcePlugin::<DefaultOpaqueRendererMethod>::default(),
                SyncComponentPlugin::<ShadowFilteringMethod, Self>::default(),
                LightmapPlugin,
                LightProbePlugin,
                GpuMeshPreprocessPlugin {
                    use_gpu_instance_buffer_builder: self.use_gpu_instance_buffer_builder,
                },
                ScreenSpaceTransmissionPlugin,
                ClusteredDecalPlugin,
                ContactShadowsPlugin,
            ))
            .add_plugins((
                decal::ForwardDecalPlugin,
                SyncComponentPlugin::<DirectionalLight, Self>::default(),
                SyncComponentPlugin::<PointLight, Self>::default(),
                SyncComponentPlugin::<SpotLight, Self>::default(),
                SyncComponentPlugin::<RectLight, Self>::default(),
                SyncComponentPlugin::<AmbientLight, Self>::default(),
            ))
            .add_plugins((
                GpuClusteringPlugin,
                ExtractResourcePlugin::<PointLightShadowAtlasReservedCapacity>::default(),
            ));

        if self.enable_volumetric_fog {
            app.add_plugins(VolumetricFogPlugin);
        }
        if self.enable_screen_space_reflections {
            app.add_plugins(ScreenSpaceReflectionsPlugin);
        }
        if self.enable_atmosphere {
            app.add_plugins((ScatteringMediumPlugin, AtmospherePlugin));
        }

        #[cfg(feature = "bevy_gltf")]
        if self.gltf_enable_standard_materials {
            gltf::add_gltf(app);
        }

        if self.add_default_deferred_lighting_plugin {
            app.add_plugins(DeferredPbrLightingPlugin);
        }

        // Initialize the default material handle.
        app.world_mut()
            .resource_mut::<Assets<StandardMaterial>>()
            .insert(
                &Handle::<StandardMaterial>::default(),
                StandardMaterial {
                    base_color: Color::srgb(1.0, 0.0, 0.5),
                    ..Default::default()
                },
            )
            .unwrap();

        let has_bluenoise = app
            .get_sub_app(RenderApp)
            .is_some_and(|render_app| render_app.world().is_resource_added::<Bluenoise>());

        if !has_bluenoise {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            #[cfg(feature = "bluenoise_texture")]
            let handle = {
                let mut image = Image::from_buffer(
                    include_bytes!("bluenoise/stbn.ktx2"),
                    bevy_image::ImageType::Extension("ktx2"),
                    bevy_image::CompressedImageFormats::NONE,
                    false,
                    ImageSampler::Default,
                    RenderAssetUsages::RENDER_WORLD,
                )
                .expect("Failed to decode embedded blue-noise texture");
                image.texture_descriptor.label = Some("bluenoise");
                images.add(image)
            };

            #[cfg(not(feature = "bluenoise_texture"))]
            let handle = { images.add(stbn_placeholder()) };

            if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
                render_app
                    .world_mut()
                    .insert_resource(Bluenoise { texture: handle });
            }
        }

        let has_area_light_luts = app
            .get_sub_app(RenderApp)
            .is_some_and(|render_app| render_app.world().is_resource_added::<AreaLightLuts>());

        if !has_area_light_luts {
            let mut images = app.world_mut().resource_mut::<Assets<Image>>();
            #[cfg(feature = "area_light_luts")]
            let handle = {
                let mut image = Image::from_buffer(
                    include_bytes!("ltc/ltc.ktx2"),
                    bevy_image::ImageType::Extension("ktx2"),
                    bevy_image::CompressedImageFormats::NONE,
                    false,
                    ImageSampler::linear(),
                    RenderAssetUsages::RENDER_WORLD,
                )
                .expect("Failed to decode embedded LTC LUTs");
                image.texture_descriptor.label = Some("area_light_luts");
                images.add(image)
            };
            #[cfg(not(feature = "area_light_luts"))]
            let handle = images.add(area_light_luts_placeholder());

            let area_light_luts = AreaLightLuts { image: handle };
            if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
                render_app.world_mut().insert_resource(area_light_luts);
            }
        }

        let has_dfg_lut = app
            .get_sub_app(RenderApp)
            .is_some_and(|render_app| render_app.world().is_resource_added::<DfgLut>());

        if !has_dfg_lut {
            #[cfg(feature = "dfg_lut")]
            let texture = app.world_mut().resource_mut::<Assets<Image>>().add(
                Image::from_buffer(
                    include_bytes!("environment_map/dfg.ktx2"),
                    bevy_image::ImageType::Extension("ktx2"),
                    bevy_image::CompressedImageFormats::NONE,
                    false,
                    ImageSampler::linear(),
                    RenderAssetUsages::RENDER_WORLD,
                )
                .expect("Failed to decode embedded DFG LUT"),
            );
            #[cfg(not(feature = "dfg_lut"))]
            let texture = Handle::default();

            if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
                render_app.world_mut().insert_resource(DfgLut { texture });
            }
        }

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // Extract the required data from the main world
        render_app
            .add_systems(
                RenderStartup,
                (
                    init_shadow_samplers,
                    init_global_clusterable_object_meta,
                    init_fallback_bindless_resources,
                ),
            )
            .add_systems(
                ExtractSchedule,
                (
                    extract_clusters_for_cpu_clustering
                        .run_if(not(gpu_clustering_is_enabled_during_extraction)),
                    extract_clusters_for_gpu_clustering
                        .run_if(gpu_clustering_is_enabled_during_extraction),
                ),
            )
            .add_systems(
                ExtractSchedule,
                (
                    extract_lights,
                    extract_shadow_rotation_compensation,
                    extract_cascade_view_override,
                    extract_ambient_light_resource,
                    extract_ambient_light,
                    extract_shadow_filtering_method,
                    extract_shadow_lod_origin,
                    late_sweep_material_instances,
                ),
            )
            .add_systems(
                Render,
                (
                    prepare_lights
                        .in_set(RenderSystems::CreateViews)
                        .after(sort_cameras),
                    prepare_clusters_for_cpu_clustering
                        .in_set(RenderSystems::PrepareResources)
                        .run_if(
                            |global_cluster_settings: Res<GlobalClusterSettings>| -> bool {
                                global_cluster_settings.gpu_clustering.is_none()
                            },
                        ),
                ),
            )
            .init_gpu_resource::<LightMeta>()
            // Persistent point-light shadow atlas backing `CachedShadowMap` shadow caching. Uses
            // `init_gpu_resource` (not `init_resource`) so it is reset to `Default` (empty texture,
            // reallocated on next `prepare_lights`) on render-device-loss recovery.
            .init_gpu_resource::<PointLightShadowAtlasCache>()
            .init_gpu_resource::<RenderMaterialBindings>()
            .init_resource::<RenderShadowLodOrigin>()
            // Render-world copy, kept in sync every frame by `extract_shadow_rotation_compensation`
            // (a manual extract system rather than the usual `ExtractResourcePlugin`, since
            // `ShadowRotationCompensation` is defined in `bevy_light`, which has no `bevy_render`
            // dependency to derive `ExtractResource` against — see that type's own doc comment).
            .init_resource::<ShadowRotationCompensation>()
            // Render-world copy of `CascadeViewOverride`, kept in sync every frame by
            // `extract_cascade_view_override` — see that resource's own doc comment.
            .init_resource::<RenderCascadeViewOverride>()
            .allow_ambiguous_resource::<RenderMaterialBindings>();

        render_app.world_mut().add_observer(add_light_view_entities);
        render_app
            .world_mut()
            .add_observer(remove_light_view_entities);
        render_app
            .world_mut()
            .add_observer(remove_point_and_spot_light_view_entities);

        render_app
            .add_systems(
                Core3d,
                (
                    // `shared_shadow_pass` is deliberately *not* registered here — see this
                    // block's own `ShadowMapSchedule` registration below for why: point/spot
                    // light shadow views (the only thing `shared_shadow_pass` ever renders,
                    // per its own doc comment) are routed to `ShadowMapSchedule` now, not
                    // `Core3d`, and `per_view_shadow_pass`'s `ViewQuery<&ViewLightEntities>`
                    // never matches one of those views' own root entity anyway (that component
                    // lives on the *camera* view listing its lights, not on a light's own
                    // shadow-view entity) — so keeping `shared_shadow_pass` registered here too
                    // would just be dead weight paying the per-view hand-off for a system that
                    // can structurally never find a matching view through this schedule again.
                    per_view_shadow_pass::<EARLY_SHADOW_PASS>
                        .after(early_prepass_build_indirect_parameters)
                        .before(early_downsample_depth)
                        .before(per_view_shadow_pass::<LATE_SHADOW_PASS>),
                    per_view_shadow_pass::<LATE_SHADOW_PASS>
                        .after(late_prepass_build_indirect_parameters)
                        .before(main_build_indirect_parameters)
                        .before(Core3dSystems::MainPass),
                ),
            )
            // See `ShadowMapSchedule`'s own doc comment (`bevy_core_pipeline::schedule`).
            // `shared_shadow_pass::<LATE_SHADOW_PASS>` is omitted — its own body
            // (`view_shadow_pass::<IS_LATE>`) is `if IS_LATE && !occlusion_culling { return; }`,
            // an unconditional no-op without `OcclusionCulling`, confirmed by reading the source.
            // Ordered relative to `GpuMeshPreprocessPlugin`'s own `ShadowMapSchedule` systems
            // (registered earlier in this same `PbrPlugin::build`, via the nested
            // `app.add_plugins(GpuMeshPreprocessPlugin { .. })` call above) rather than being
            // part of that same `.chain()` — mirrors this exact system's own position in the
            // `Core3d` chain just above (`.after(early_prepass_build_indirect_parameters)`,
            // `.before(early_downsample_depth)`, i.e. between the early and late gpu-preprocess
            // phases), not a guess.
            .add_systems(
                ShadowMapSchedule,
                shared_shadow_pass::<EARLY_SHADOW_PASS>
                    .after(early_prepass_build_indirect_parameters)
                    .before(late_gpu_preprocess),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        let global_cluster_settings = make_global_cluster_settings(render_app.world());
        app.insert_resource(global_cluster_settings);
    }
}

pub fn stbn_placeholder() -> Image {
    let format = TextureFormat::Rgba8Unorm;
    let data = vec![255, 0, 255, 255];
    Image {
        data: Some(data),
        data_order: TextureDataOrder::default(),
        texture_descriptor: TextureDescriptor {
            size: Extent3d::default(),
            format,
            dimension: TextureDimension::D2,
            label: Some("bluenoise_placeholder"),
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        sampler: ImageSampler::Default,
        texture_view_descriptor: None,
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        copy_on_resize: false,
    }
}

pub fn area_light_luts_placeholder() -> Image {
    let format = TextureFormat::Rgba16Float;
    let data = vec![0; 16];
    Image {
        data: Some(data),
        data_order: TextureDataOrder::default(),
        texture_descriptor: TextureDescriptor {
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 2,
            },
            format,
            dimension: TextureDimension::D2,
            label: Some("area_light_luts_placeholder"),
            mip_level_count: 1,
            sample_count: 1,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        sampler: ImageSampler::Default,
        texture_view_descriptor: Some(TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        }),
        asset_usage: RenderAssetUsages::RENDER_WORLD,
        copy_on_resize: false,
    }
}

impl SyncComponent<PbrPlugin> for DirectionalLight {
    type Target = (
        Self,
        ExtractedDirectionalLight,
        RenderExtractedShadowMapVisibleEntities,
        RenderShadowMapVisibleEntities,
        DirectionalLightViewEntities,
    );
}
impl SyncComponent<PbrPlugin> for PointLight {
    type Target = (
        Self,
        ExtractedPointLight,
        RenderExtractedShadowMapVisibleEntities,
        RenderShadowMapVisibleEntities,
        PointAndSpotLightViewEntities,
    );
}
impl SyncComponent<PbrPlugin> for SpotLight {
    type Target = (
        Self,
        ExtractedPointLight,
        RenderExtractedShadowMapVisibleEntities,
        RenderShadowMapVisibleEntities,
        PointAndSpotLightViewEntities,
    );
}
impl SyncComponent<PbrPlugin> for RectLight {
    type Target = (Self, ExtractedRectLight);
}
impl SyncComponent<PbrPlugin> for AmbientLight {
    type Target = Self;
}
impl SyncComponent<PbrPlugin> for ShadowFilteringMethod {
    type Target = Self;
}
