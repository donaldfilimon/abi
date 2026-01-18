//! Platform-Aware Time Utilities
//!
//! Provides cross-platform time functionality that works on all targets
//! including WASM where std.time.Instant is not available.
//!
//! On WASM: Uses std.crypto.random for entropy-based seeding since
//! monotonic timers are unavailable. Timing functions return 0.

const std = @import("std");
const builtin = @import("builtin");

/// Check if we're on a platform that supports std.time.Instant
pub const has_instant = !isWasmTarget();

fn isWasmTarget() bool {
    return builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64;
}

/// Thread-safe counter for WASM fallback uniqueness
var wasm_counter: std.atomic.Value(u64) = .{ .raw = 0 };

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

/// Get a seed value suitable for PRNG initialization.
/// On native platforms: uses monotonic timestamp for uniqueness.
/// On WASM: uses std.crypto.random for true randomness.
/// This should be used instead of timestampNs() for seeding PRNGs.
pub fn getSeed() u64 {
    if (has_instant) {
        // On native platforms, timestamp provides good uniqueness
        return timestampNs();
    } else {
        // On WASM, use cryptographic randomness + counter for uniqueness
        var seed_bytes: [8]u8 = undefined;
        std.crypto.random.bytes(&seed_bytes);
        const random_part = std.mem.readInt(u64, &seed_bytes, .little);
        // Add counter to ensure uniqueness even if crypto.random has issues
        const counter = wasm_counter.fetchAdd(1, .monotonic);
        return random_part ^ counter;
    }
}

/// Get a unique ID (useful for node IDs, etc.)
/// On WASM, uses cryptographic randomness for uniqueness.
pub fn getUniqueId() u64 {
    var prng = std.Random.DefaultPrng.init(getSeed());
    return prng.random().int(u64);
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

test "getSeed returns unique values" {
    const seed1 = getSeed();
    const seed2 = getSeed();
    const seed3 = getSeed();
    // Seeds should be different (with very high probability)
    try std.testing.expect(seed1 != seed2 or seed2 != seed3);
}

test "getUniqueId returns unique values" {
    const id1 = getUniqueId();
    const id2 = getUniqueId();
    const id3 = getUniqueId();
    // IDs should be different
    try std.testing.expect(id1 != id2);
    try std.testing.expect(id2 != id3);
}
