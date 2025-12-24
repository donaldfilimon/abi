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
    enable_gpu: bool = build_options.enable_gpu,
    enable_network: bool = false,
    enable_profiling: bool = false,
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
    .enable_gpu = build_options.enable_gpu,
    .enable_network = false,
    .enable_profiling = false,
    .gpu_backend = null,
};
