//! Compute Feature Module
//!
//! CPU/GPU compute engine for performance-critical workloads

const std = @import("std");
const lifecycle = @import("../lifecycle.zig");

pub const main = @import("main.zig");
pub const benchmark = @import("benchmark.zig");
pub const demo = @import("demo.zig");

/// Initialize the compute feature module
pub const init = lifecycle.init;

/// Deinitialize the compute feature module
pub const deinit = lifecycle.deinit;
