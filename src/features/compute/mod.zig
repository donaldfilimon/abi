//! Compute Feature Module
//!
//! CPU/GPU compute engine for performance-critical workloads

const std = @import("std");

pub const main = @import("main.zig");
pub const benchmark = @import("benchmark.zig");
pub const demo = @import("demo.zig");

/// Initialize the compute feature module
pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator; // Currently no global compute state to initialize
}

/// Deinitialize the compute feature module
pub fn deinit() void {
    // Currently no global compute state to cleanup
}
