//! The core rendering pipelines schedules. These schedules define the "default" render graph
//! for 2D and 3D rendering in Bevy.
//!
//! Rendering in Bevy is "camera driven", meaning that for each camera in the world, its
//! associated rendering schedule is executed. This allows different cameras to have different
//! rendering pipelines, for example a 3D camera with post-processing effects and a 2D camera
//! with a simple clear and sprite rendering.
//!
//! The [`camera_driver`] system is responsible for iterating over all cameras in the world
//! and executing their associated schedules. In this way, the schedule for each camera is a
//! sub-schedule or sub-graph of the root render graph schedule.
use core::fmt::{self, Display, Formatter};

use bevy_camera::{ClearColor, NormalizedRenderTarget};
use bevy_ecs::{
    entity::EntityHashSet,
    prelude::*,
    schedule::{InternedScheduleLabel, IntoScheduleConfigs, Schedule, ScheduleLabel, SystemSet},
};
#[cfg(feature = "trace")]
use bevy_log::info_span;
use bevy_reflect::Reflect;
use bevy_render::{
    camera::{ExtractedCamera, SortedCameras},
    render_resource::{
        CommandEncoderDescriptor, LoadOp, Operations, RenderPassColorAttachment,
        RenderPassDescriptor, StoreOp,
    },
    renderer::{CurrentView, PendingCommandBuffers, RenderDevice, RenderQueue},
    view::ExtractedWindows,
};

/// Schedule label for the Core 3D rendering pipeline.
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Core3d;

/// System sets for the Core 3D rendering pipeline, defining the main stages of rendering.
/// These stages include and run in the following order:
/// - `Prepass`: Initial rendering operations, such as depth pre-pass.
/// - `MainPass`: The primary rendering operations, including drawing opaque and transparent objects.
/// - `EarlyPostProcess`: Initial post processing effects.
/// - `PostProcess`: Final rendering operations, such as post-processing effects.
///
/// Additional systems can be added to these sets to customize the rendering pipeline, or additional
/// sets can be created relative to these core sets.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Core3dSystems {
    Prepass,
    MainPass,
    EarlyPostProcess,
    PostProcess,
}

impl Core3d {
    pub fn base_schedule() -> Schedule {
        use bevy_ecs::schedule::ScheduleBuildSettings;
        use Core3dSystems::*;

        let mut schedule = Schedule::new(Self);

        schedule.set_build_settings(ScheduleBuildSettings {
            auto_insert_apply_deferred: false,
            ..Default::default()
        });

        schedule.configure_sets((Prepass, MainPass, EarlyPostProcess, PostProcess).chain());

        schedule
    }
}

/// Schedule label for the Core 2D rendering pipeline.
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Core2d;

/// System sets for the Core 2D rendering pipeline, defining the main stages of rendering.
/// These stages include and run in the following order:
/// - `Prepass`: Initial rendering operations, such as depth pre-pass.
/// - `MainPass`: The primary rendering operations, including drawing 2D sprites and meshes.
/// - `EarlyPostProcess`: Initial post processing effects.
/// - `PostProcess`: Final rendering operations, such as post-processing effects.
///
/// Additional systems can be added to these sets to customize the rendering pipeline, or additional
/// sets can be created relative to these core sets.
#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Core2dSystems {
    Prepass,
    MainPass,
    EarlyPostProcess,
    PostProcess,
}

impl Core2d {
    pub fn base_schedule() -> Schedule {
        use bevy_ecs::schedule::ScheduleBuildSettings;
        use Core2dSystems::*;

        let mut schedule = Schedule::new(Self);

        schedule.set_build_settings(ScheduleBuildSettings {
            auto_insert_apply_deferred: false,
            ..Default::default()
        });

        schedule.configure_sets((Prepass, MainPass, EarlyPostProcess, PostProcess).chain());

        schedule
    }
}

/// Schedule label for a minimal render-graph schedule used by point/spot light shadow map views
/// (see `RootNonCameraView`, below), instead of the full `Core3d` schedule.
///
/// A shadow-map view never draws to `Core3dSystems::MainPass`/`PostProcess` at all — it only ever
/// writes its own depth-only shadow map via `bevy_pbr::render::light::{per_view_shadow_pass,
/// shared_shadow_pass}` (plus the GPU mesh-preprocessing steps those depend on) — so routing it
/// through all 43 of `Core3d`'s node systems pays a per-view executor hand-off (measured ~5.6us
/// per node in this project's own Tracy trace analysis, `docs/render_orchestration_handoff.md`'s
/// Lever 2) for ~35 nodes that are pure no-ops for this view type, on top of the ~8 that matter.
/// `camera_driver` runs whatever schedule a `RootNonCameraView` names, generically — this is a
/// second, much smaller such schedule alongside `Core2d`/`Core3d`, not a special case in the
/// driver itself.
///
/// Node set and ordering, re-derived (not copy-pasted) from `bevy_pbr`'s own registration of
/// these systems into `Core3d`, since that registration's `.after()`/`.before()` constraints
/// reference `Core3d`-only systems (`early_prepass`, `late_prepass`, `early_downsample_depth`,
/// `late_deferred_prepass`) that don't exist in this schedule and so can't just be copied
/// verbatim:
/// - `late_prepass_build_indirect_parameters` and the `LATE_SHADOW_PASS` (`IS_LATE = true`)
///   instantiations of `per_view_shadow_pass`/`shared_shadow_pass` are deliberately **not**
///   registered here at all (not merely reordered) — confirmed by reading their own gating,
///   not inferred: `late_prepass_build_indirect_parameters`'s `run_if` requires
///   `With<OcclusionCulling>` on some entity, and `view_shadow_pass::<IS_LATE>`'s own body is
///   `if IS_LATE && !occlusion_culling { return; }` — both are unconditional no-ops for this
///   project, which uses no `OcclusionCulling` component anywhere. Omitting them here changes no
///   behavior versus today's `Core3d`-routed version; it only removes hand-off for systems that
///   already never did anything.
/// - Every other node in the original registration (`clear_indirect_parameters_metadata`,
///   `unpack_bins`, `early_gpu_preprocess`, `early_prepass_build_indirect_parameters`,
///   `late_gpu_preprocess`, `main_build_indirect_parameters`, and the `EARLY_SHADOW_PASS`
///   instantiations of `per_view_shadow_pass`/`shared_shadow_pass`) is kept, chained in one
///   single sequential order that preserves every relative-order constraint the original
///   registration actually established between kept nodes (see the two call sites' own comments
///   in `bevy_pbr/src/render/gpu_preprocess.rs`/`light.rs` for exactly which). Where the original
///   established no constraint between two kept nodes, this chain still picks *some* order
///   (harmless — a real dependency would misbehave either way if this reasoning were wrong; an
///   absent one is unaffected by which arbitrary order is picked).
#[derive(ScheduleLabel, Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ShadowMapSchedule;

/// Holds the entity of windows that are a render target for a camera
#[derive(Resource)]
struct CameraWindows(EntityHashSet);

/// A render-world marker component for a view that corresponds to neither a
/// camera nor a camera-associated shadow map.
///
/// This is used for point light and spot light shadow maps, since these aren't
/// associated with views.
#[derive(Clone, Copy, Component, Debug, Reflect)]
#[reflect(Clone, Component)]
#[reflect(from_reflect = false)]
pub struct RootNonCameraView(#[reflect(ignore)] pub InternedScheduleLabel);

/// The default entry point for camera driven rendering added to the root [`bevy_render::renderer::RenderGraph`]
/// schedule. This system iterates over all cameras in the world, executing their associated
/// rendering schedules defined by the [`bevy_render::camera::CameraRenderGraph`] component.
///
/// After executing all camera schedules, it submits any pending command buffers to the GPU
/// and clears any swap chains that were not covered by a camera. Users can order any additional
/// operations (e.g. one-off compute passes) before or after this system in the root render
/// graph schedule.
pub fn camera_driver(world: &mut World) {
    // Gather up all cameras and auxiliary views not associated with a camera.
    let root_views: Vec<_> = {
        let mut auxiliary_views = world.query_filtered::<Entity, With<RootNonCameraView>>();
        let sorted = world.resource::<SortedCameras>();
        auxiliary_views
            .iter(world)
            .map(RootView::Auxiliary)
            .chain(sorted.0.iter().map(|c| RootView::Camera {
                entity: c.entity,
                order: c.order,
            }))
            .collect()
    };

    let mut camera_windows = EntityHashSet::default();

    for root_view in root_views {
        let mut run_schedule = true;
        let (schedule, view_entity);

        match root_view {
            RootView::Camera {
                entity: camera_entity,
                ..
            } => {
                let Some(camera) = world.get::<ExtractedCamera>(camera_entity) else {
                    continue;
                };

                schedule = camera.schedule;
                let target = camera.target.clone();

                if let Some(NormalizedRenderTarget::Window(window_ref)) = &target {
                    let window_entity = window_ref.entity();
                    let windows = world.resource::<ExtractedWindows>();
                    if windows
                        .windows
                        .get(&window_entity)
                        .is_some_and(|w| w.physical_width > 0 && w.physical_height > 0)
                    {
                        camera_windows.insert(window_entity);
                    } else {
                        run_schedule = false;
                    }
                }

                view_entity = camera_entity;
            }

            RootView::Auxiliary(auxiliary_view_entity) => {
                let Some(root_view) = world.get::<RootNonCameraView>(auxiliary_view_entity) else {
                    continue;
                };

                view_entity = auxiliary_view_entity;
                schedule = root_view.0;
            }
        }

        if run_schedule {
            world.insert_resource(CurrentView(view_entity));

            #[cfg(feature = "trace")]
            let _span =
                bevy_log::info_span!("camera_schedule", camera = root_view.to_string()).entered();

            world.run_schedule(schedule);
        }
    }
    world.remove_resource::<CurrentView>();

    world.insert_resource(CameraWindows(camera_windows));
}

/// A view not associated with any other camera.
enum RootView {
    /// A camera.
    Camera { entity: Entity, order: isize },

    /// An auxiliary view not associated with a camera.
    ///
    /// This is currently used for point and spot light shadow maps.
    Auxiliary(Entity),
}

impl Display for RootView {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match *self {
            RootView::Camera { entity, order } => write!(f, "Camera {} ({:?})", order, entity),
            RootView::Auxiliary(entity) => write!(f, "Auxiliary View {:?}", entity),
        }
    }
}

pub(crate) fn submit_pending_command_buffers(world: &mut World) {
    let mut pending = world.resource_mut::<PendingCommandBuffers>();
    #[cfg(feature = "trace")]
    let buffer_count = pending.len();
    let buffers = pending.take();

    if !buffers.is_empty() {
        #[cfg(feature = "trace")]
        let _span = info_span!("queue_submit", count = buffer_count).entered();
        let queue = world.resource::<RenderQueue>();
        queue.submit(buffers);
    }
}

pub(crate) fn handle_uncovered_swap_chains(world: &mut World) {
    let windows_to_clear: Vec<_> = {
        let clear_color = world.resource::<ClearColor>().0.to_linear();
        let Some(camera_windows) = world.remove_resource::<CameraWindows>() else {
            return;
        };
        let windows = world.resource::<ExtractedWindows>();
        windows
            .iter()
            .filter_map(|(window_entity, window)| {
                if camera_windows.0.contains(window_entity) {
                    return None;
                }
                let swap_chain_texture = window.swap_chain_texture_view.as_ref()?;
                Some((swap_chain_texture.clone(), clear_color))
            })
            .collect()
    };

    if windows_to_clear.is_empty() {
        return;
    }

    let render_device = world.resource::<RenderDevice>();
    let render_queue = world.resource::<RenderQueue>();

    let mut encoder = render_device.create_command_encoder(&CommandEncoderDescriptor::default());

    for (swap_chain_texture, clear_color) in &windows_to_clear {
        #[cfg(feature = "trace")]
        let _span = bevy_log::info_span!("no_camera_clear_pass").entered();

        let pass_descriptor = RenderPassDescriptor {
            label: Some("no_camera_clear_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: swap_chain_texture,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear((*clear_color).into()),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        encoder.begin_render_pass(&pass_descriptor);
    }

    render_queue.submit([encoder.finish()]);
}
