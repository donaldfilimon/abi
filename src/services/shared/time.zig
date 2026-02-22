//! Platform-Aware Time Utilities
//!
//! Provides cross-platform time functionality that works on all targets
//! including WASM where timing APIs are not available.
//!
//! On WASM: Uses entropy when available; otherwise falls back to a counter
//! since monotonic timers are unavailable. Timing functions return 0.

const std = @import("std");
const builtin = @import("builtin");

/// Check if we're on a platform that supports timing APIs
pub const has_instant = !isWasmTarget();

fn isWasmTarget() bool {
    return builtin.cpu.arch == .wasm32 or builtin.cpu.arch == .wasm64;
}

/// Thread-safe counter for WASM fallback uniqueness
const CounterInt = if (isWasmTarget()) u32 else u64;
var wasm_counter: std.atomic.Value(CounterInt) = .{ .raw = 0 };

/// Platform-aware Instant type
/// Uses OS monotonic clocks on supported platforms.
pub const Instant = struct {
    nanos: u128, // Monotonic nanoseconds

    pub fn now() error{Unsupported}!@This() {
        if (isWasmTarget()) {
            return error.Unsupported;
        }

        if (builtin.os.tag == .windows) {
            var qpc: std.os.windows.LARGE_INTEGER = undefined;
            var freq: std.os.windows.LARGE_INTEGER = undefined;
            if (std.os.windows.ntdll.RtlQueryPerformanceCounter(&qpc) == 0) {
                return error.Unsupported;
            }
            if (std.os.windows.ntdll.RtlQueryPerformanceFrequency(&freq) == 0 or freq <= 0) {
                return error.Unsupported;
            }
            const ticks: u64 = @intCast(qpc);
            const hz: u64 = @intCast(freq);
            const nanos = (@as(u128, ticks) * std.time.ns_per_s) / @as(u128, hz);
            return .{ .nanos = nanos };
        }

        var ts: std.posix.timespec = undefined;
        if (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts)) != .SUCCESS) {
            return error.Unsupported;
        }
        const sec = @as(u128, @intCast(if (ts.sec < 0) 0 else ts.sec));
        const nsec = @as(u128, @intCast(if (ts.nsec < 0) 0 else ts.nsec));
        return .{ .nanos = sec * std.time.ns_per_s + nsec };
    }

    pub fn since(self: @This(), earlier: @This()) u64 {
        if (self.nanos >= earlier.nanos) {
            const diff = self.nanos - earlier.nanos;
            // Clamp to u64 max if it overflows
            return if (diff > std.math.maxInt(u64))
                std.math.maxInt(u64)
            else
                @intCast(diff);
        }
        return 0;
    }
};

/// Timer for measuring elapsed time (std.time.Timer compatible API)
/// Provides nanosecond-precision timing on supported platforms
pub const Timer = struct {
    start_instant: Instant,

    pub fn start() error{Unsupported}!Timer {
        return .{ .start_instant = try Instant.now() };
    }

    pub fn read(self: *Timer) u64 {
        const current = Instant.now() catch return 0;
        return current.since(self.start_instant);
    }

    pub fn reset(self: *Timer) void {
        self.start_instant = Instant.now() catch return;
    }

    pub fn lap(self: *Timer) u64 {
        const elapsed_ns = self.read();
        self.reset();
        return elapsed_ns;
    }
};

/// Get current instant, returns null on failure or unsupported platform
pub fn now() ?Instant {
    return Instant.now() catch |err| {
        std.log.debug("Failed to get current instant: {t}", .{err});
        return null;
    };
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

/// App start time for relative timing (initialized lazily)
var app_start: ?Instant = null;
var app_start_initialized: bool = false;

fn getAppStart() ?Instant {
    if (!app_start_initialized) {
        app_start = Instant.now() catch |err| blk: {
            std.log.debug("Failed to initialize app start time: {t}", .{err});
            break :blk null;
        };
        app_start_initialized = true;
    }
    return app_start;
}

/// Get a timestamp in nanoseconds (monotonic, from app start)
/// Returns 0 on WASM
pub fn timestampNs() u64 {
    if (!has_instant) return 0;
    const start = getAppStart() orelse return 0;
    const now_inst = Instant.now() catch |err| {
        std.log.debug("Failed to get current instant for timestamp: {t}", .{err});
        return 0;
    };
    return now_inst.since(start);
}

/// Get a timestamp in milliseconds
pub fn timestampMs() u64 {
    return timestampNs() / std.time.ns_per_ms;
}

/// Get a timestamp in seconds
pub fn timestampSec() u64 {
    return timestampNs() / std.time.ns_per_s;
}

/// Get current time in unix seconds (approximate, monotonic since app start).
/// On WASM, returns 0 (no timer available).
pub fn unixSeconds() i64 {
    if (!has_instant) return 0;
    return @intCast(timestampSec());
}

/// Get current time in unix seconds (alias for unixSeconds).
pub fn nowSeconds() i64 {
    return unixSeconds();
}

/// Get current time in unix milliseconds (approximate, monotonic).
/// On WASM, returns 0 (no timer available).
pub fn unixMs() i64 {
    if (!has_instant) return 0;
    return @intCast(timestampMs());
}

/// Get current time in unix milliseconds (alias for unixMs).
pub fn nowMs() i64 {
    return unixMs();
}

/// Get current time in nanoseconds (monotonic).
/// On WASM, returns 0 (no timer available).
pub fn nowNanoseconds() i64 {
    if (!has_instant) return 0;
    return @intCast(timestampNs());
}

/// Get current time in milliseconds (alias for compatibility).
pub fn nowMilliseconds() i64 {
    return nowMs();
}

/// Injectable time provider for tests and deterministic clocks.
pub const TimeProvider = struct {
    ctx: ?*anyopaque = null,
    nowMsFn: *const fn (?*anyopaque) i64 = defaultNowMs,

    pub fn nowMs(self: TimeProvider) i64 {
        return self.nowMsFn(self.ctx);
    }
};

fn defaultNowMs(ctx: ?*anyopaque) i64 {
    _ = ctx;
    return nowMs();
}

/// Sleep for a specified number of nanoseconds.
/// On WASM, this is a no-op (can't block in WASM).
pub fn sleepNs(ns: u64) void {
    if (!has_instant) return;
    if (@hasDecl(std.posix, "nanosleep")) {
        // POSIX systems: use nanosleep
        var req = std.posix.timespec{
            .sec = @intCast(ns / std.time.ns_per_s),
            .nsec = @intCast(ns % std.time.ns_per_s),
        };
        var rem: std.posix.timespec = undefined;
        while (true) {
            const result = std.posix.nanosleep(&req, &rem);
            if (result == 0) break;
            req = rem;
        }
    } else if (builtin.os.tag == .windows) {
        // Windows: use NtDelayExecution with negative interval (relative time)
        // Interval is in 100-nanosecond units, negative = relative
        const hns = ns / 100;
        const interval: std.os.windows.LARGE_INTEGER = -@as(i64, @intCast(@min(hns, @as(u64, @intCast(std.math.maxInt(i64))))));
        _ = std.os.windows.ntdll.NtDelayExecution(0, &interval);
    } else {
        // Fallback: busy-wait (avoid if possible)
        const start = now() orelse return;
        while (true) {
            const current = now() orelse return;
            if (current.since(start) >= ns) break;
        }
    }
}

/// Sleep for a specified number of milliseconds.
/// On WASM, this is a no-op (can't block in WASM).
pub fn sleepMs(ms: u64) void {
    const ns = ms * std.time.ns_per_ms;
    sleepNs(ns);
}

/// Get a seed value suitable for PRNG initialization.
/// On native platforms: uses monotonic timestamp for uniqueness.
/// On WASM: uses std.c.arc4random_buf for true randomness.
/// This should be used instead of timestampNs() for seeding PRNGs.
pub fn getSeed() u64 {
    if (has_instant) {
        // On native platforms, timestamp provides good uniqueness
        return timestampNs();
    } else {
        if (builtin.os.tag == .wasi) {
            // On WASI, use cryptographic randomness + counter for uniqueness
            var seed_bytes: [8]u8 = undefined;
            std.c.arc4random_buf(&seed_bytes, seed_bytes.len);
            const random_part = std.mem.readInt(u64, &seed_bytes, .little);
            const counter = @as(u64, wasm_counter.fetchAdd(1, .monotonic));
            return random_part ^ counter;
        }

        // On freestanding WASM, use a counter + address salt fallback.
        const counter = @as(u64, wasm_counter.fetchAdd(1, .monotonic));
        const salt = @as(u64, @intFromPtr(&wasm_counter));
        return counter ^ (salt *% 0x9e3779b97f4a7c15);
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

test "timer basic" {
    if (has_instant) {
        var timer = Timer.start() catch return error.SkipZigTest;
        sleepMs(10);
        const elapsed_ns = timer.read();
        try std.testing.expect(elapsed_ns > 0);
    }
}

test "getSeed returns unique values" {
    const seed1 = getSeed();
    sleepMs(1); // Ensure time passes
    const seed2 = getSeed();
    sleepMs(1);
    const seed3 = getSeed();
    // Seeds should be different (with very high probability)
    try std.testing.expect(seed1 != seed2 or seed2 != seed3);
}

test "getUniqueId returns unique values" {
    const id1 = getUniqueId();
    sleepMs(1);
    const id2 = getUniqueId();
    sleepMs(1);
    const id3 = getUniqueId();
    // IDs should be different (with very high probability given sleep)
    try std.testing.expect(id1 != id2 or id2 != id3);
}
