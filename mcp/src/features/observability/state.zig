//! Observability Module State
//!
//! Lifecycle management for the observability module: initialization,
//! teardown, and runtime status queries.

const std = @import("std");
const build_options = @import("build_options");
const types = @import("types.zig");

const MonitoringError = types.MonitoringError;

var initialized = std.atomic.Value(bool).init(false);

pub fn init(_: std.mem.Allocator) !void {
    if (!isEnabled()) return MonitoringError.MonitoringDisabled;
    initialized.store(true, .release);
}

pub fn deinit() void {
    initialized.store(false, .release);
}

pub fn isEnabled() bool {
    return build_options.feat_observability;
}

pub fn isInitialized() bool {
    return initialized.load(.acquire);
}
