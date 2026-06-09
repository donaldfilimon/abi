//! Telemetry stub -- disabled at compile time.

const std = @import("std");
pub const types = @import("types.zig");

pub const TelemetryError = types.TelemetryError;
pub const Error = types.Error;

pub fn isEnabled() bool {
    return false;
}

/// Record is a no-op when the feature is compiled out.
pub fn record(name: []const u8) void {
    _ = name;
}

/// Increment is a no-op when the feature is compiled out.
pub fn increment(name: []const u8, delta: u64) void {
    _ = name;
    _ = delta;
}

test {
    std.testing.refAllDecls(@This());
}

test "telemetry stub reports disabled" {
    try std.testing.expect(!isEnabled());
    // Calls are safe no-ops
    record("test.event");
    increment("test.counter", 1);
}
