//! GPU runtime execution layer for managing compute contexts and buffers.
//! Provides runtime context lifecycle, buffer allocation, built-in and
//! custom operations, and health/metrics collection.

const std = @import("std");

pub const context = @import("context.zig");
pub const buffers = @import("buffers.zig");
pub const ops_builtin = @import("ops_builtin.zig");
pub const ops_custom = @import("ops_custom.zig");
pub const health_metrics = @import("health_metrics.zig");

test {
    std.testing.refAllDecls(@This());
}
