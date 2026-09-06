//! Snapshot collection for the diagnostics dashboard.
//!
//! Gathers GPU, plugin, WDBX, scheduler, and memory facts into a
//! [`DashboardState`] for the renderers. GPU fields keep `abi-gpu`'s honest
//! disclosure: `accelerated` stays false when native kernels are not linked.

use std::sync::Arc;

use abi_core::{MemoryTracker, Scheduler, TaskPriority};
use abi_plugins::PluginManager;

use super::{DashboardState, Options, PANES};

pub(super) fn dashboard_health(ds: &DashboardState) -> &'static str {
    if ds.scheduler_failed > 0 || ds.memory_leaked > 0 {
        return "attention";
    }
    if ds.gpu_accelerated && ds.gpu_linked {
        return "nominal";
    }
    // No GPU kernels active (or only CPU SIMD): report health as "cpu".
    "cpu"
}

pub(super) fn collect_state(options: &Options) -> DashboardState {
    let gpu = abi_gpu::detect_backend();
    let mut manager = PluginManager::new();
    manager.load_bundled();
    let plugin_names: Vec<String> = manager.list().iter().map(|p| p.name.clone()).collect();
    let plugin_count = manager.plugin_count();

    let tracker = Arc::new(MemoryTracker::new());
    let scheduler = Scheduler::new().with_memory_tracker(Arc::clone(&tracker));
    scheduler.submit("dashboard-init", TaskPriority::Normal, Box::new(|| Ok(())));
    scheduler.submit("wdbx-snapshot", TaskPriority::Low, Box::new(|| Ok(())));
    let _ = scheduler.run_all();
    let stats = scheduler.stats();
    let mem = tracker.snapshot();

    DashboardState {
        gpu_backend: gpu.backend.name().to_string(),
        gpu_accelerated: gpu.accelerated,
        gpu_linked: abi_gpu::metal_kernels::kernels_linked(),
        plugin_count,
        plugin_names,
        // Ephemeral probe store — never opens the user's durable path.
        wdbx_blocks: 0,
        wdbx_vectors: 0,
        wdbx_entries: 0,
        wdbx_spatial_records: 0,
        scheduler_source: "CLI dashboard (live)",
        scheduler_running: stats.running,
        scheduler_pending: stats.pending,
        scheduler_completed: stats.completed,
        scheduler_failed: stats.failed,
        memory_source: "MemoryTracker (live)",
        memory_peak: mem.peak_usage,
        memory_current: mem.current_usage,
        // Empty probe tasks do not allocate through the tracker.
        memory_leaked: 0,
        selected_pane: options.initial_pane.min(PANES.len() - 1),
        refresh_interval_ms: options.refresh_interval_ms,
        compact: options.compact,
        color: options.color,
        interactive: false,
    }
}
