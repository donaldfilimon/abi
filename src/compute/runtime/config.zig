//! Engine configuration
//!
//! Centralized configuration for compute engine initialization.

const std = @import("std");

pub const EngineConfig = struct {
    worker_count: u32,
    drain_mode: enum {
        drain,
        discard,
    } = .drain,
    metrics_buffer_size: usize = 1024,
    topology_flags: u32 = 0,
};

pub const DEFAULT_CONFIG = EngineConfig{
    .worker_count = 0,
    .drain_mode = .drain,
    .metrics_buffer_size = 1024,
    .topology_flags = 0,
};
