//! Platform-Aware Time Utilities
//!
//! Provides cross-platform time functionality that works on all targets
//! including WASM where std.time.Instant is not available.

const std = @import("std");
const builtin = @import("builtin");

/// Check if we're on a platform that supports std.time.Instant
pub const has_instant = !isWasmTarget();

fn isWasmTarget() bool {
    return builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64;
}

/// Platform-aware Instant type
/// On native platforms, this is std.time.Instant
/// On WASM, this is a stub that returns 0
pub const Instant = if (has_instant) std.time.Instant else struct {
    counter: u64,

    pub fn now() error{Unsupported}!@This() {
        return .{ .counter = 0 };
    }

    pub fn since(self: @This(), earlier: @This()) u64 {
        _ = self;
        _ = earlier;
        return 0;
    }
};

/// Get current instant, returns null on failure or unsupported platform
pub fn now() ?Instant {
    return Instant.now() catch null;
}

/// Get elapsed nanoseconds since a previous instant
/// Returns 0 on WASM or if either instant is null
pub fn elapsed(start: ?Instant, end: ?Instant) u64 {
    if (!has_instant) return 0;
    const s = start orelse return 0;
    const e = end orelse return 0;
    return e.since(s);
}

/// Get elapsed milliseconds since a previous instant
pub fn elapsedMs(start: ?Instant, end: ?Instant) u64 {
    return elapsed(start, end) / std.time.ns_per_ms;
}

/// Get elapsed seconds since a previous instant
pub fn elapsedSec(start: ?Instant, end: ?Instant) u64 {
    return elapsed(start, end) / std.time.ns_per_s;
}

/// Get a timestamp in nanoseconds (monotonic, from app start)
/// Returns 0 on WASM
pub fn timestampNs() u64 {
    if (!has_instant) return 0;
    const instant = Instant.now() catch return 0;
    // Use timestamp field directly for simplicity
    return instant.timestamp;
}

/// Get a timestamp in milliseconds
pub fn timestampMs() u64 {
    return timestampNs() / std.time.ns_per_ms;
}

/// Get a timestamp in seconds
pub fn timestampSec() u64 {
    return timestampNs() / std.time.ns_per_s;
}

test "instant on native platform" {
    if (has_instant) {
        const start = now();
        try std.testing.expect(start != null);
        const end = now();
        try std.testing.expect(end != null);
        // elapsed should be >= 0
        try std.testing.expect(elapsed(start, end) >= 0);
    }
}
