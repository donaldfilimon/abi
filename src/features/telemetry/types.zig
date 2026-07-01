//! Shared types for the telemetry feature.
//!
//! Both `mod.zig` (real implementation) and `stub.zig` (disabled no-op)
//! import from here so that type definitions are not duplicated.

const std = @import("std");

/// Errors returned by telemetry operations.
pub const TelemetryError = error{
    FeatureDisabled,
    OutOfMemory,
};

pub const Error = TelemetryError;

/// One retained counter, name-owned by the caller of `snapshot`.
pub const CounterSnapshot = struct {
    name: []const u8,
    value: u64,
};

/// Point-in-time aggregate view of the telemetry table.
pub const Summary = struct {
    total: u64,
    distinct: usize,
    dropped: u64,
};

test {
    std.testing.refAllDecls(@This());
}
