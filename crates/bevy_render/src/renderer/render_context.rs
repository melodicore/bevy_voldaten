use super::WgpuWrapper;
use crate::diagnostic::internal::DiagnosticsRecorder;
use crate::render_phase::TrackedRenderPass;
use crate::render_resource::{CommandEncoder, RenderPassDescriptor};
use crate::renderer::RenderDevice;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::change_detection::Tick;
use bevy_ecs::component::ComponentId;
use bevy_ecs::prelude::*;
use bevy_ecs::query::{FilteredAccessSet, QueryData, QueryFilter, QueryState};
use bevy_ecs::system::{
    Deferred, SystemBuffer, SystemMeta, SystemParam, SystemParamValidationError,
};
use bevy_ecs::world::unsafe_world_cell::UnsafeWorldCell;
use bevy_ecs::world::DeferredWorld;
#[cfg(feature = "trace")]
use bevy_log::info_span;
use alloc::sync::Arc;
use core::marker::PhantomData;
use std::sync::{Mutex, MutexGuard};
use wgpu::CommandBuffer;

/// A finished command buffer, or an encoder whose `finish()` is deferred to
/// [`PendingCommandBuffers::take`].
///
/// `wgpu` records passes lazily and does the actual backend encoding (barriers, resource
/// tracking, the native command calls) in `CommandEncoder::finish`, which makes it the expensive
/// step. Finishing every encoder serially on the render thread, at the end of each render system,
/// put all of that on the critical path; deferring it lets `take` finish independent encoders in
/// parallel.
enum PendingCommand {
    Buffer(CommandBuffer),
    Encoder(CommandEncoder),
}

impl PendingCommand {
    fn finish(self) -> CommandBuffer {
        match self {
            PendingCommand::Buffer(buffer) => buffer,
            PendingCommand::Encoder(encoder) => encoder.finish(),
        }
    }
}

/// Buffers and encoders in submission order, plus the currently *open* encoder, which comes after
/// all of them.
///
/// **Shared, not per-system (voldaten fork).** Every `RenderContext` records into the one open
/// encoder instead of creating its own, so consecutive render systems — a camera's whole `Core3d`
/// invocation, say — append to a single command buffer rather than producing one each. wgpu's
/// per-command-buffer cost (submit-time validation and barrier recording, and later
/// `drop command buffer trackers` in `Device::maintain`, each ~10µs) was being paid ~116 times a
/// frame. `camera_driver` closes the open encoder after every view
/// ([`PendingCommandBuffers::close_open_encoder`]), so there is still one encoder per view for
/// [`PendingCommandBuffers::take`] to finish in parallel.
///
/// A `RenderContext` takes the open encoder out of here the first time it records and puts it back
/// when it is dropped at the end of its system — no lock is held while recording. That is only
/// order-correct because render systems that record never run concurrently: `Core2d`/`Core3d` (app
/// side) and `ShadowMapSchedule` (fork) use the single-threaded executor, and `camera_driver` runs
/// views one after another. A second encoder opened concurrently would still be submitted, just
/// not in a defined order relative to the first (see `RenderContext`'s `Drop`).
#[derive(Default)]
struct SharedCommands {
    commands: Vec<PendingCommand>,
    open: Option<CommandEncoder>,
}

impl SharedCommands {
    /// Moves the open encoder (if any) to the end of `commands`.
    fn close_open_encoder(&mut self) {
        if let Some(encoder) = self.open.take() {
            self.commands.push(PendingCommand::Encoder(encoder));
        }
    }
}

type SharedCommandsHandle = Arc<Mutex<WgpuWrapper<SharedCommands>>>;

fn lock_shared(shared: &SharedCommandsHandle) -> MutexGuard<'_, WgpuWrapper<SharedCommands>> {
    shared
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A resource that holds command buffers and encoders that are pending submission to the render queue.
#[derive(Resource)]
pub struct PendingCommandBuffers(SharedCommandsHandle);

/// Reuses the existing shared list (emptied) when the resource already exists. It is
/// (re)inserted by `init_gpu_resource` in `RenderStartup` — including after device recovery — and
/// every `RenderContextState` holds on to the list it was initialized with, so replacing it with a
/// fresh one would leave those recording into a list nothing ever submits.
impl FromWorld for PendingCommandBuffers {
    fn from_world(world: &mut World) -> Self {
        match world.get_resource::<PendingCommandBuffers>() {
            Some(existing) => {
                let shared = existing.0.clone();
                **lock_shared(&shared) = SharedCommands::default();
                Self(shared)
            }
            None => Self(Arc::new(Mutex::new(WgpuWrapper::new(SharedCommands::default())))),
        }
    }
}

impl PendingCommandBuffers {
    pub fn push(&mut self, buffers: impl IntoIterator<Item = CommandBuffer>) {
        let mut shared = lock_shared(&self.0);
        shared.close_open_encoder();
        shared
            .commands
            .extend(buffers.into_iter().map(PendingCommand::Buffer));
    }

    /// Queues an unfinished encoder; it is finished in [`Self::take`], keeping its place in
    /// submission order.
    pub fn push_encoder(&mut self, encoder: CommandEncoder) {
        let mut shared = lock_shared(&self.0);
        shared.close_open_encoder();
        shared.commands.push(PendingCommand::Encoder(encoder));
    }

    /// Ends the shared open encoder that `RenderContext`s record into, so the next recording
    /// starts a new one. Called by `camera_driver` after every view, keeping one encoder per view
    /// for [`Self::take`] to finish in parallel.
    pub fn close_open_encoder(&mut self) {
        lock_shared(&self.0).close_open_encoder();
    }

    /// Takes every pending command buffer in submission order, finishing any pending encoders.
    ///
    /// Encoders are independent of one another, so when there is more than one they are finished
    /// in parallel on the [`ComputeTaskPool`](bevy_tasks::ComputeTaskPool); the returned order is
    /// the order they were queued in regardless.
    pub fn take(&mut self) -> Vec<CommandBuffer> {
        let commands = {
            let mut shared = lock_shared(&self.0);
            shared.close_open_encoder();
            core::mem::take(&mut shared.commands)
        };
        let encoder_count = commands
            .iter()
            .filter(|command| matches!(command, PendingCommand::Encoder(_)))
            .count();

        #[cfg(feature = "trace")]
        let _span = info_span!("finish_command_encoders", count = encoder_count).entered();

        // wgpu types are only `Send` off the web.
        #[cfg(not(target_arch = "wasm32"))]
        if encoder_count > 1 {
            return bevy_tasks::ComputeTaskPool::get().scope(|scope| {
                for command in commands {
                    scope.spawn(async move { command.finish() });
                }
            });
        }

        commands.into_iter().map(PendingCommand::finish).collect()
    }

    pub fn is_empty(&self) -> bool {
        let shared = lock_shared(&self.0);
        shared.commands.is_empty() && shared.open.is_none()
    }

    pub fn len(&self) -> usize {
        let shared = lock_shared(&self.0);
        shared.commands.len() + usize::from(shared.open.is_some())
    }
}

struct RenderContextStateInner {
    /// The shared open encoder, while this system has it checked out (see [`SharedCommands`]).
    command_encoder: Option<CommandEncoder>,
    render_device: Option<RenderDevice>,
}

/// Per-system state of the [`RenderContext`] system parameter: the shared command list (see
/// [`SharedCommands`]) and, while the system runs, the encoder it has checked out of it.
pub struct RenderContextState {
    shared: SharedCommandsHandle,
    inner: WgpuWrapper<RenderContextStateInner>,
}

impl FromWorld for RenderContextState {
    fn from_world(world: &mut World) -> Self {
        let shared = world
            .get_resource_or_init::<PendingCommandBuffers>()
            .0
            .clone();
        Self {
            shared,
            inner: WgpuWrapper::new(RenderContextStateInner {
                command_encoder: None,
                render_device: None,
            }),
        }
    }
}

impl RenderContextState {
    /// Checks the shared open encoder out (or starts a new one) on first use.
    fn command_encoder(&mut self) -> &mut CommandEncoder {
        if self.inner.command_encoder.is_none() {
            let open = lock_shared(&self.shared).open.take();
            let render_device = self
                .inner
                .render_device
                .clone()
                .expect("RenderDevice must be set before accessing command_encoder");
            self.inner.command_encoder = Some(open.unwrap_or_else(|| {
                render_device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
            }));
        }
        self.inner.command_encoder.as_mut().unwrap()
    }

    /// Ends the encoder this system is recording into (the checked-out one, else the shared open
    /// one), so whatever is queued next comes after it.
    fn close_encoder(&mut self) {
        let mut shared = lock_shared(&self.shared);
        if let Some(encoder) = self.inner.command_encoder.take() {
            shared.close_open_encoder();
            shared.commands.push(PendingCommand::Encoder(encoder));
        } else {
            shared.close_open_encoder();
        }
    }

    /// Returns the checked-out encoder to the shared slot, to be continued by the next recording
    /// system.
    fn check_in(&mut self) {
        if let Some(encoder) = self.inner.command_encoder.take() {
            let mut shared = lock_shared(&self.shared);
            // Only possible if another system recorded concurrently; order between the two is then
            // undefined, but nothing is lost.
            shared.close_open_encoder();
            shared.open = Some(encoder);
        }
        self.inner.render_device = None;
    }

    /// Finishes this system's checked-out encoder, if any. Buffers queued through
    /// [`RenderContext::add_command_buffer`] have already gone to [`PendingCommandBuffers`].
    pub fn finish(&mut self) -> Vec<CommandBuffer> {
        self.inner
            .command_encoder
            .take()
            .map(|encoder| encoder.finish())
            .into_iter()
            .collect()
    }
}

impl SystemBuffer for RenderContextState {
    fn queue(&mut self, _system_meta: &SystemMeta, _world: DeferredWorld) {
        // Nothing left to do: everything was recorded into the shared command list while the
        // system ran, and `RenderContext`'s `Drop` already returned the encoder.
        debug_assert!(
            self.inner.command_encoder.is_none(),
            "RenderContext encoder was not checked back in"
        );
    }
}

/// A system parameter that provides access to a command encoder and render device for issuing
/// rendering commands inside any system running beneath the root [`super::RenderGraph`] schedule in the
/// [`super::render_system`] system.
#[derive(SystemParam)]
pub struct RenderContext<'w, 's> {
    state: Deferred<'s, RenderContextState>,
    render_device: Res<'w, RenderDevice>,
    diagnostics_recorder: Option<Res<'w, DiagnosticsRecorder>>,
}

impl Drop for RenderContext<'_, '_> {
    /// Hands the encoder this system recorded into back to the shared slot, so the next render
    /// system appends to it (see [`SharedCommands`]).
    fn drop(&mut self) {
        self.state.check_in();
    }
}

impl<'w, 's> RenderContext<'w, 's> {
    fn ensure_device(&mut self) {
        if self.state.inner.render_device.is_none() {
            self.state.inner.render_device = Some(self.render_device.clone());
        }
    }

    /// Returns the render device associated with this render context.
    pub fn render_device(&self) -> &RenderDevice {
        &self.render_device
    }

    /// Returns the diagnostics recorder, if available.
    pub fn diagnostic_recorder(&self) -> Option<Res<'w, DiagnosticsRecorder>> {
        self.diagnostics_recorder.as_ref().map(Res::clone)
    }

    /// Returns the current command encoder, creating one if it does not already exist.
    pub fn command_encoder(&mut self) -> &mut CommandEncoder {
        self.ensure_device();
        self.state.command_encoder()
    }

    /// Begins a tracked render pass with the given descriptor.
    pub fn begin_tracked_render_pass<'a>(
        &'a mut self,
        descriptor: RenderPassDescriptor<'_>,
    ) -> TrackedRenderPass<'a> {
        self.ensure_device();

        let render_pass = self.state.command_encoder().begin_render_pass(&descriptor);
        TrackedRenderPass::new(&self.render_device, render_pass)
    }

    /// Adds a finished command buffer to be submitted later.
    pub fn add_command_buffer(&mut self, command_buffer: CommandBuffer) {
        self.state.close_encoder();
        lock_shared(&self.state.shared)
            .commands
            .push(PendingCommand::Buffer(command_buffer));
    }
}

/// A system parameter that can be used to explicitly flush pending command buffers to the render queue.
/// This is typically not necessary, as command buffers are automatically flushed at the end of each
/// render system. However, in some cases it may be useful to flush command buffers earlier.
#[derive(SystemParam)]
pub struct FlushCommands<'w> {
    pending: ResMut<'w, PendingCommandBuffers>,
    queue: Res<'w, super::RenderQueue>,
}

impl<'w> FlushCommands<'w> {
    /// Flushes all pending command buffers to the render queue.
    pub fn flush(&mut self) {
        let buffers = self.pending.take();
        if !buffers.is_empty() {
            // Mirrors `submit_pending_command_buffers`'s `queue_submit` span, so mid-frame submits
            // are visible in traces too.
            #[cfg(feature = "trace")]
            let _span = info_span!("flush_queue_submit", count = buffers.len()).entered();
            self.queue.submit(buffers);
        }
    }
}

/// The entity corresponding to the current view being rendered.
#[derive(Resource, Debug, Clone, Copy, PartialEq, Eq, Deref, DerefMut)]
pub struct CurrentView(pub Entity);

/// A query that fetches components for the entity corresponding to the current view being rendered,
/// as defined by the [`CurrentView`] resource, equivalent to `query.get(current_view.entity())`.
pub struct ViewQuery<'w, 's, D: QueryData, F: QueryFilter = ()> {
    entity: Entity,
    item: D::Item<'w, 's>,
    _filter: PhantomData<F>,
}

impl<'w, 's, D: QueryData, F: QueryFilter> ViewQuery<'w, 's, D, F> {
    #[inline]
    pub fn entity(&self) -> Entity {
        self.entity
    }

    #[inline]
    pub fn into_inner(self) -> D::Item<'w, 's> {
        self.item
    }
}

pub struct ViewQueryState<D: QueryData, F: QueryFilter> {
    resource_id: ComponentId,
    query_state: QueryState<D, F>,
}

// SAFETY: ViewQuery accesses the CurrentView resource (read) and query components.
// Access is properly registered in init_access.
unsafe impl<'a, D: QueryData + 'static, F: QueryFilter + 'static> SystemParam
    for ViewQuery<'a, '_, D, F>
{
    type State = ViewQueryState<D, F>;
    type Item<'w, 's> = ViewQuery<'w, 's, D, F>;

    fn init_state(world: &mut World) -> Self::State {
        ViewQueryState {
            resource_id: world
                .components_registrator()
                .register_component::<CurrentView>(),
            query_state: QueryState::new(world),
        }
    }

    fn init_access(
        state: &Self::State,
        system_meta: &mut SystemMeta,
        component_access_set: &mut FilteredAccessSet,
        world: &mut World,
    ) {
        component_access_set.add_resource_read(state.resource_id);

        <Query<'_, '_, D, F> as SystemParam>::init_access(
            &state.query_state,
            system_meta,
            component_access_set,
            world,
        );
    }

    #[inline]
    unsafe fn get_param<'w, 's>(
        state: &'s mut Self::State,
        _system_meta: &SystemMeta,
        world: UnsafeWorldCell<'w>,
        _change_tick: Tick,
    ) -> Result<Self::Item<'w, 's>, SystemParamValidationError> {
        // SAFETY: We have registered resource read access in init_access
        let current_view = unsafe { world.get_resource::<CurrentView>() };

        let Some(current_view) = current_view else {
            return Err(SystemParamValidationError::skipped::<Self>(
                "CurrentView resource not present",
            ));
        };

        let entity = current_view.entity();

        // SAFETY: Query state access is properly registered in init_access.
        // The caller ensures the world matches the one used in init_state.
        let item = unsafe { state.query_state.get_unchecked(world, entity) }.map_err(|_| {
            SystemParamValidationError::skipped::<Self>("Current view entity does not match query")
        })?;

        Ok(ViewQuery {
            entity,
            item,
            _filter: PhantomData,
        })
    }
}

// SAFETY: ViewQuery with ReadOnlyQueryData only reads from the world.
unsafe impl<'w, 's, D: bevy_ecs::query::ReadOnlyQueryData + 'static, F: QueryFilter + 'static>
    bevy_ecs::system::ReadOnlySystemParam for ViewQuery<'w, 's, D, F>
{
}
