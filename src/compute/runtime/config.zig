//! Engine configuration
//!
//! Centralized configuration for compute engine initialization.

const std = @import("std");

const build_options = @import("build_options");

pub const EngineConfig = struct {
    worker_count: u32,
    drain_mode: enum {
        drain,
        discard,
    } = .drain,
    metrics_buffer_size: usize = 1024,
    topology_flags: u32 = 0,
    queue_capacity: usize = 1024,
    drain_spin_iterations: u32 = 1000,
    idle_spin_iterations: u32 = 100,
    error_callback: ?*const fn (worker_id: u32, task_id: u64, err: anyerror) void = null,
    gpu_backend: ?enum {
        cuda,
        vulkan,
        metal,
    } = null,
};

pub const DEFAULT_CONFIG = EngineConfig{
    .worker_count = 0,
    .drain_mode = .drain,
    .metrics_buffer_size = 1024,
    .topology_flags = 0,
    .queue_capacity = 1024,
    .drain_spin_iterations = 1000,
    .idle_spin_iterations = 100,
    .error_callback = null,
    .gpu_backend = null,
};
