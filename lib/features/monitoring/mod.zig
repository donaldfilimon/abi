//! Monitoring Feature Module
//!
//! System monitoring, performance profiling, and observability

const std = @import("std");

// Health monitoring
pub const health = @import("health.zig");

// Performance monitoring and profiling
pub const performance = @import("performance.zig");
pub const performance_profiler = @import("performance_profiler.zig");
pub const memory_tracker = @import("memory_tracker.zig");

// Tracing and sampling
pub const tracing = @import("tracing.zig");
pub const sampling = @import("sampling.zig");

// Metrics and monitoring
pub const prometheus = @import("prometheus.zig");
pub const regression = @import("regression.zig");
pub const metrics = @import("metrics.zig");
pub const logging = @import("logging.zig");

/// Initialize the monitoring feature module
pub fn init(allocator: std.mem.Allocator) !void {
    _ = allocator; // Currently no global monitoring state to initialize
}

/// Deinitialize the monitoring feature module
pub fn deinit() void {
    // Currently no global monitoring state to cleanup
}

// Legacy compatibility removed - circular import fixed

test {
    std.testing.refAllDecls(@This());
}
