//! Telemetry stub -- disabled at compile time.
//!
//! Mirrors the public surface of `mod.zig` so callers compile unchanged, but
//! every operation is an inert no-op and every reader returns an empty result.

const std = @import("std");
pub const types = @import("types.zig");

pub const TelemetryError = types.TelemetryError;
pub const Error = types.Error;

pub const SLOT_CAPACITY = 0;
pub const NAME_CAPACITY = 0;

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

pub fn counterValue(name: []const u8) u64 {
    _ = name;
    return 0;
}

pub fn totalEvents() u64 {
    return 0;
}

pub fn distinctCounters() usize {
    return 0;
}

pub fn droppedEvents() u64 {
    return 0;
}

pub fn reset() void {}

test {
    std.testing.refAllDecls(@This());
}

test "telemetry stub reports disabled and stays inert" {
    try std.testing.expect(!isEnabled());
    record("test.event");
    increment("test.counter", 1);
    reset();
    try std.testing.expectEqual(@as(u64, 0), counterValue("test.counter"));
    try std.testing.expectEqual(@as(u64, 0), totalEvents());
    try std.testing.expectEqual(@as(usize, 0), distinctCounters());
    try std.testing.expectEqual(@as(u64, 0), droppedEvents());
}
