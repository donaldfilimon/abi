//! Telemetry Feature
//!
//! Lightweight event emission and structured observability hooks. Provides
//! cheap always-available (when enabled) primitives for recording named events
//! and counters from scheduler, wdbx, ai, and connectors. Complements the
//! opt-in `metrics` feature.

pub const types = @import("types.zig");

const std = @import("std");

pub const TelemetryError = types.TelemetryError;
pub const Error = types.Error;

pub fn isEnabled() bool {
    return true;
}

/// Record a named event (fire-and-forget). In the real implementation this may
/// forward to a ring buffer, logger, or registered sinks.
pub fn record(name: []const u8) void {
    // Placeholder: real impl could validate name, timestamp it, etc.
    _ = name;
}

/// Increment a named counter by delta (no-op if name empty).
pub fn increment(name: []const u8, delta: u64) void {
    if (name.len == 0) return;
    _ = delta;
}

test {
    std.testing.refAllDecls(@This());
}

test "telemetry records events without error when enabled" {
    try std.testing.expect(isEnabled());
    // Should not panic or error on call
    record("scheduler.task.submitted");
    increment("tasks.total", 1);
    increment("", 5); // tolerated
}
