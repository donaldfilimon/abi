//! Unified rate limiter for request throttling.
//!
//! Implements four rate-limiting algorithms with configurable synchronization
//! strategy, following the same `SyncStrategy` pattern as `circuit_breaker.zig`.
//!
//! ## Algorithms
//!
//! - **Token bucket**: smooth refill, burst-friendly. Default.
//! - **Fixed window**: simple counter reset every window.
//! - **Sliding window**: histogram-based approximate sliding window.
//! - **Leaky bucket**: steady drain, equivalent to token bucket with burst = max.
//!
//! ## Sync Strategies
//!
//! - `.atomic`: Lock-free atomics for high-throughput paths.
//! - `.mutex`: Mutex-protected for multi-threaded use.
//! - `.none`: No synchronization for single-threaded use.
//!
//! ## Usage
//!
//! ```zig
//! var rl = RateLimiter(.none).init(.{
//!     .max_requests = 100,
//!     .window_ns = 1_000_000_000,
//!     .algorithm = .token_bucket,
//! });
//! const result = rl.acquire(now_ns);
//! switch (result) {
//!     .allowed => |info| { /* proceed */ },
//!     .denied => |info| { /* retry after info.retry_after_ns */ },
//! }
//! ```

const std = @import("std");
const circuit_breaker = @import("circuit_breaker.zig");
const sync = @import("../sync.zig");

/// Reuse the same SyncStrategy as circuit_breaker.
pub const SyncStrategy = circuit_breaker.SyncStrategy;

/// Rate-limiting algorithm.
pub const Algorithm = enum {
    token_bucket,
    fixed_window,
    sliding_window,
    leaky_bucket,
};

/// Configuration for the shared rate limiter core.
pub const Config = struct {
    /// Maximum requests allowed per window.
    max_requests: u32 = 100,
    /// Window duration in nanoseconds.
    window_ns: u64 = 1_000_000_000,
    /// Algorithm to use.
    algorithm: Algorithm = .token_bucket,
    /// Burst capacity (token bucket / leaky bucket). Defaults to max_requests.
    burst_capacity: ?u32 = null,
};

/// Result of an acquire attempt.
pub const Result = union(enum) {
    allowed: AllowedInfo,
    denied: DeniedInfo,

    pub const AllowedInfo = struct {
        remaining: u32,
        reset_after_ns: u64,
    };

    pub const DeniedInfo = struct {
        retry_after_ns: u64,
    };

    pub fn isAllowed(self: Result) bool {
        return self == .allowed;
    }
};

/// Parameterized rate limiter.
///
/// `strategy` selects the synchronization approach:
/// - `.atomic` -- lock-free atomics, suitable for multi-threaded hot paths
/// - `.mutex`  -- mutex-protected, for general multi-threaded use
/// - `.none`   -- no sync, caller ensures single-threaded access
pub fn RateLimiter(comptime strategy: SyncStrategy) type {
    return struct {
        const Self = @This();

        /// Fixed-point scale factor for sub-token precision.
        /// Must be large enough so that `(max_requests * FP_SCALE) / window_ns`
        /// does not truncate to zero for typical configurations (e.g. 10 rps
        /// with nanosecond-precision windows).
        const FP_SCALE: u64 = 1_000_000_000;

        config: Config,
        effective_capacity: u32,

        // ── Token bucket / leaky bucket state ──
        tokens_fp: TokensField, // fixed-point tokens * FP_SCALE
        last_refill_ns: TimeField,
        refill_rate_fp: u64, // tokens * FP_SCALE per nanosecond (fixed-point)

        // ── Fixed window state ──
        fw_count: CountField,
        fw_window_start_ns: TimeField,

        // ── Sliding window state (7-bucket histogram) ──
        sw_buckets: [7]CountField,
        sw_bucket_start_ns: TimeField,
        sw_bucket_width_ns: u64,

        // ── Mutex (only for .mutex strategy) ──
        mutex: MutexField,

        // ── Field types depend on strategy ──
        const CountField = switch (strategy) {
            .atomic => std.atomic.Value(u32),
            .mutex, .none => u32,
        };
        const TokensField = switch (strategy) {
            .atomic => std.atomic.Value(u64),
            .mutex, .none => u64,
        };
        const TimeField = switch (strategy) {
            .atomic => std.atomic.Value(u128),
            .mutex, .none => u128,
        };
        const MutexField = switch (strategy) {
            .mutex => sync.Mutex,
            .atomic, .none => void,
        };

        pub fn init(config: Config, now_ns: u128) Self {
            const capacity = config.burst_capacity orelse config.max_requests;
            // For leaky bucket, capacity equals max_requests (steady drain).
            const effective_cap: u32 = if (config.algorithm == .leaky_bucket)
                config.max_requests
            else
                capacity;

            // Fixed-point refill rate: (max_requests * FP_SCALE) / window_ns
            // This avoids floating-point while preserving sub-token precision.
            const refill_rate_fp: u64 = if (config.window_ns > 0)
                (@as(u64, config.max_requests) * FP_SCALE) / config.window_ns
            else
                0;

            const sw_bucket_width: u64 = if (config.window_ns > 0)
                config.window_ns / 7
            else
                0;

            return Self{
                .config = config,
                .effective_capacity = effective_cap,
                .tokens_fp = initTokens(@as(u64, effective_cap) * FP_SCALE),
                .last_refill_ns = initTime(now_ns),
                .refill_rate_fp = refill_rate_fp,
                .fw_count = initCount(0),
                .fw_window_start_ns = initTime(now_ns),
                .sw_buckets = initBuckets(),
                .sw_bucket_start_ns = initTime(now_ns),
                .sw_bucket_width_ns = sw_bucket_width,
                .mutex = initMutex(),
            };
        }

        // ── Init helpers ──

        fn initCount(val: u32) CountField {
            return switch (strategy) {
                .atomic => std.atomic.Value(u32).init(val),
                .mutex, .none => val,
            };
        }

        fn initTokens(val: u64) TokensField {
            return switch (strategy) {
                .atomic => std.atomic.Value(u64).init(val),
                .mutex, .none => val,
            };
        }

        fn initTime(val: u128) TimeField {
            return switch (strategy) {
                .atomic => std.atomic.Value(u128).init(val),
                .mutex, .none => val,
            };
        }

        fn initMutex() MutexField {
            return switch (strategy) {
                .mutex => .{},
                .atomic, .none => {},
            };
        }

        fn initBuckets() [7]CountField {
            var buckets: [7]CountField = undefined;
            for (&buckets) |*b| {
                b.* = initCount(0);
            }
            return buckets;
        }

        // ── Field access helpers ──

        fn loadCount(field: *const CountField) u32 {
            return switch (strategy) {
                .atomic => field.load(.monotonic),
                .mutex, .none => field.*,
            };
        }

        fn storeCount(field: *CountField, val: u32) void {
            switch (strategy) {
                .atomic => field.store(val, .monotonic),
                .mutex, .none => field.* = val,
            }
        }

        fn addCount(field: *CountField, val: u32) void {
            switch (strategy) {
                .atomic => _ = field.fetchAdd(val, .monotonic),
                .mutex, .none => field.* += val,
            }
        }

        fn loadTokens(field: *const TokensField) u64 {
            return switch (strategy) {
                .atomic => field.load(.monotonic),
                .mutex, .none => field.*,
            };
        }

        fn storeTokens(field: *TokensField, val: u64) void {
            switch (strategy) {
                .atomic => field.store(val, .monotonic),
                .mutex, .none => field.* = val,
            }
        }

        fn loadTime(field: *const TimeField) u128 {
            return switch (strategy) {
                .atomic => field.load(.monotonic),
                .mutex, .none => field.*,
            };
        }

        fn storeTime(field: *TimeField, val: u128) void {
            switch (strategy) {
                .atomic => field.store(val, .monotonic),
                .mutex, .none => field.* = val,
            }
        }

        // ── Lock helpers ──

        fn lockIfNeeded(self: *Self) void {
            if (strategy == .mutex) self.mutex.lock();
        }

        fn unlockIfNeeded(self: *Self) void {
            if (strategy == .mutex) self.mutex.unlock();
        }

        // ── Public API ──

        /// Try to acquire a single permit at `now_ns` (nanoseconds).
        pub fn acquire(self: *Self, now_ns: u128) Result {
            return self.acquireN(1, now_ns);
        }

        /// Try to acquire `n` permits at `now_ns`.
        pub fn acquireN(self: *Self, n: u32, now_ns: u128) Result {
            self.lockIfNeeded();
            defer self.unlockIfNeeded();

            return switch (self.config.algorithm) {
                .token_bucket, .leaky_bucket => self.tokenBucketAcquire(n, now_ns),
                .fixed_window => self.fixedWindowAcquire(n, now_ns),
                .sliding_window => self.slidingWindowAcquire(now_ns),
            };
        }

        /// Return the number of available tokens/permits (snapshot).
        pub fn available(self: *Self) u32 {
            self.lockIfNeeded();
            defer self.unlockIfNeeded();

            return switch (self.config.algorithm) {
                .token_bucket, .leaky_bucket => @intCast(loadTokens(&self.tokens_fp) / FP_SCALE),
                .fixed_window => self.config.max_requests -| loadCount(&self.fw_count),
                .sliding_window => blk: {
                    var total: u32 = 0;
                    for (&self.sw_buckets) |*b| total += loadCount(b);
                    break :blk self.config.max_requests -| total;
                },
            };
        }

        /// Reset the limiter to initial state at `now_ns`.
        pub fn reset(self: *Self, now_ns: u128) void {
            self.lockIfNeeded();
            defer self.unlockIfNeeded();

            storeTokens(&self.tokens_fp, @as(u64, self.effective_capacity) * FP_SCALE);
            storeTime(&self.last_refill_ns, now_ns);
            storeCount(&self.fw_count, 0);
            storeTime(&self.fw_window_start_ns, now_ns);
            for (&self.sw_buckets) |*b| storeCount(b, 0);
            storeTime(&self.sw_bucket_start_ns, now_ns);
        }

        // ── Token bucket algorithm ──

        fn tokenBucketAcquire(self: *Self, n: u32, now_ns: u128) Result {
            self.refillTokens(now_ns);

            const current_fp = loadTokens(&self.tokens_fp);
            const needed_fp: u64 = @as(u64, n) * FP_SCALE;

            if (current_fp >= needed_fp) {
                storeTokens(&self.tokens_fp, current_fp - needed_fp);
                return .{ .allowed = .{
                    .remaining = @intCast((current_fp - needed_fp) / FP_SCALE),
                    .reset_after_ns = 0,
                } };
            }

            // Denied: compute wait time
            const deficit_fp = needed_fp - current_fp;
            const wait_ns: u64 = if (self.refill_rate_fp > 0)
                deficit_fp / self.refill_rate_fp
            else
                self.config.window_ns;
            return .{ .denied = .{ .retry_after_ns = wait_ns } };
        }

        fn refillTokens(self: *Self, now_ns: u128) void {
            const last = loadTime(&self.last_refill_ns);
            if (now_ns <= last) return;
            const elapsed_ns: u64 = @intCast(@min(now_ns - last, std.math.maxInt(u64)));
            if (elapsed_ns == 0) return;

            const tokens_to_add_fp = elapsed_ns * self.refill_rate_fp;
            if (tokens_to_add_fp > 0) {
                const cap_fp: u64 = @as(u64, self.effective_capacity) * FP_SCALE;
                const current_fp = loadTokens(&self.tokens_fp);
                storeTokens(&self.tokens_fp, @min(current_fp + tokens_to_add_fp, cap_fp));
                storeTime(&self.last_refill_ns, now_ns);
            }
        }

        // ── Fixed window algorithm ──

        fn fixedWindowAcquire(self: *Self, n: u32, now_ns: u128) Result {
            const ws = loadTime(&self.fw_window_start_ns);
            const window_ns_128: u128 = self.config.window_ns;

            // Reset window if expired
            if (now_ns >= ws + window_ns_128) {
                storeCount(&self.fw_count, 0);
                storeTime(&self.fw_window_start_ns, now_ns);
            }

            const current = loadCount(&self.fw_count);
            if (current + n <= self.config.max_requests) {
                storeCount(&self.fw_count, current + n);
                return .{ .allowed = .{
                    .remaining = self.config.max_requests - current - n,
                    .reset_after_ns = 0,
                } };
            }

            const ws2 = loadTime(&self.fw_window_start_ns);
            const remaining_ns: u64 = @intCast(@min(
                (ws2 + window_ns_128) -| now_ns,
                std.math.maxInt(u64),
            ));
            return .{ .denied = .{ .retry_after_ns = remaining_ns } };
        }

        // ── Sliding window algorithm (7-bucket histogram) ──

        fn slidingWindowAcquire(self: *Self, now_ns: u128) Result {
            self.advanceSlidingBuckets(now_ns);

            var total: u32 = 0;
            for (&self.sw_buckets) |*b| total += loadCount(b);

            if (total < self.config.max_requests) {
                const idx = self.currentSlidingBucketIdx(now_ns);
                addCount(&self.sw_buckets[idx], 1);
                const remaining = self.config.max_requests - total - 1;
                return .{ .allowed = .{
                    .remaining = remaining,
                    .reset_after_ns = 0,
                } };
            }

            const retry_ns: u64 = if (self.sw_bucket_width_ns > 0)
                self.sw_bucket_width_ns
            else
                self.config.window_ns;
            return .{ .denied = .{ .retry_after_ns = retry_ns } };
        }

        fn currentSlidingBucketIdx(self: *const Self, now_ns: u128) usize {
            if (self.sw_bucket_width_ns == 0) return 0;
            const start = loadTime(&self.sw_bucket_start_ns);
            const elapsed = now_ns -| start;
            return @intCast((elapsed / self.sw_bucket_width_ns) % 7);
        }

        fn advanceSlidingBuckets(self: *Self, now_ns: u128) void {
            if (self.sw_bucket_width_ns == 0) return;
            const start = loadTime(&self.sw_bucket_start_ns);
            const elapsed = now_ns -| start;
            const buckets_to_advance = elapsed / self.sw_bucket_width_ns;

            if (buckets_to_advance >= 7) {
                for (&self.sw_buckets) |*b| storeCount(b, 0);
                storeTime(&self.sw_bucket_start_ns, now_ns);
            } else if (buckets_to_advance > 0) {
                var i: usize = 0;
                while (i < buckets_to_advance) : (i += 1) {
                    const base_idx = self.currentSlidingBucketIdx(start);
                    const old_idx = (base_idx + i) % 7;
                    storeCount(&self.sw_buckets[old_idx], 0);
                }
                storeTime(
                    &self.sw_bucket_start_ns,
                    start + buckets_to_advance * self.sw_bucket_width_ns,
                );
            }
        }
    };
}

// ============================================================================
// Convenience aliases
// ============================================================================

/// Single-threaded rate limiter (no synchronization).
pub const SimpleRateLimiter = RateLimiter(.none);
/// Lock-free rate limiter (atomics).
pub const AtomicRateLimiter = RateLimiter(.atomic);
/// Mutex-protected rate limiter.
pub const MutexRateLimiter = RateLimiter(.mutex);

// ============================================================================
// Tests
// ============================================================================

test "RateLimiter(.none) token bucket basic" {
    const now: u128 = 1_000_000_000;
    var rl = SimpleRateLimiter.init(.{
        .max_requests = 5,
        .window_ns = 1_000_000_000,
        .algorithm = .token_bucket,
    }, now);

    // Should allow 5 requests (capacity)
    for (0..5) |_| {
        const r = rl.acquire(now);
        try std.testing.expect(r.isAllowed());
    }
    // 6th should be denied
    const r = rl.acquire(now);
    try std.testing.expect(!r.isAllowed());
}

test "RateLimiter(.none) token bucket refill" {
    const now: u128 = 1_000_000_000;
    var rl = SimpleRateLimiter.init(.{
        .max_requests = 10,
        .window_ns = 1_000_000_000,
        .algorithm = .token_bucket,
    }, now);

    // Exhaust tokens
    for (0..10) |_| _ = rl.acquire(now);
    try std.testing.expect(!rl.acquire(now).isAllowed());

    // Advance half a window -- should have ~5 tokens
    const later = now + 500_000_000;
    try std.testing.expect(rl.acquire(later).isAllowed());
}

test "RateLimiter(.atomic) fixed window basic" {
    const now: u128 = 1_000_000_000;
    var rl = AtomicRateLimiter.init(.{
        .max_requests = 3,
        .window_ns = 1_000_000_000,
        .algorithm = .fixed_window,
    }, now);

    for (0..3) |_| {
        try std.testing.expect(rl.acquire(now).isAllowed());
    }
    try std.testing.expect(!rl.acquire(now).isAllowed());

    // New window
    const later = now + 1_000_000_001;
    try std.testing.expect(rl.acquire(later).isAllowed());
}

test "RateLimiter(.mutex) sliding window basic" {
    const now: u128 = 1_000_000_000;
    var rl = MutexRateLimiter.init(.{
        .max_requests = 4,
        .window_ns = 1_000_000_000,
        .algorithm = .sliding_window,
    }, now);

    for (0..4) |_| {
        try std.testing.expect(rl.acquire(now).isAllowed());
    }
    try std.testing.expect(!rl.acquire(now).isAllowed());
}

test "RateLimiter(.none) leaky bucket" {
    const now: u128 = 1_000_000_000;
    var rl = SimpleRateLimiter.init(.{
        .max_requests = 5,
        .window_ns = 1_000_000_000,
        .algorithm = .leaky_bucket,
    }, now);

    for (0..5) |_| {
        try std.testing.expect(rl.acquire(now).isAllowed());
    }
    try std.testing.expect(!rl.acquire(now).isAllowed());
}

test "RateLimiter(.none) burst capacity" {
    const now: u128 = 1_000_000_000;
    var rl = SimpleRateLimiter.init(.{
        .max_requests = 10,
        .window_ns = 1_000_000_000,
        .algorithm = .token_bucket,
        .burst_capacity = 3,
    }, now);

    // Only 3 burst tokens
    for (0..3) |_| {
        try std.testing.expect(rl.acquire(now).isAllowed());
    }
    try std.testing.expect(!rl.acquire(now).isAllowed());
}

test "RateLimiter(.none) acquireN" {
    const now: u128 = 1_000_000_000;
    var rl = SimpleRateLimiter.init(.{
        .max_requests = 10,
        .window_ns = 1_000_000_000,
        .algorithm = .token_bucket,
    }, now);

    const r = rl.acquireN(5, now);
    try std.testing.expect(r.isAllowed());
    try std.testing.expectEqual(@as(u32, 5), r.allowed.remaining);

    // 6 more should be denied
    const r2 = rl.acquireN(6, now);
    try std.testing.expect(!r2.isAllowed());
}

test "RateLimiter(.none) available" {
    const now: u128 = 1_000_000_000;
    var rl = SimpleRateLimiter.init(.{
        .max_requests = 10,
        .window_ns = 1_000_000_000,
        .algorithm = .token_bucket,
    }, now);

    try std.testing.expectEqual(@as(u32, 10), rl.available());
    _ = rl.acquireN(3, now);
    try std.testing.expectEqual(@as(u32, 7), rl.available());
}

test "RateLimiter(.none) reset" {
    const now: u128 = 1_000_000_000;
    var rl = SimpleRateLimiter.init(.{
        .max_requests = 5,
        .window_ns = 1_000_000_000,
        .algorithm = .token_bucket,
    }, now);

    for (0..5) |_| _ = rl.acquire(now);
    try std.testing.expect(!rl.acquire(now).isAllowed());

    rl.reset(now);
    try std.testing.expect(rl.acquire(now).isAllowed());
    try std.testing.expectEqual(@as(u32, 4), rl.available());
}

test "RateLimiter(.none) denied result has retry_after_ns" {
    const now: u128 = 1_000_000_000;
    var rl = SimpleRateLimiter.init(.{
        .max_requests = 1,
        .window_ns = 1_000_000_000,
        .algorithm = .token_bucket,
    }, now);

    _ = rl.acquire(now);
    const r = rl.acquire(now);
    try std.testing.expect(!r.isAllowed());
    try std.testing.expect(r.denied.retry_after_ns > 0);
}

test "RateLimiter(.none) fixed window denied has retry_after_ns" {
    const now: u128 = 1_000_000_000;
    var rl = SimpleRateLimiter.init(.{
        .max_requests = 1,
        .window_ns = 1_000_000_000,
        .algorithm = .fixed_window,
    }, now);

    _ = rl.acquire(now);
    const r = rl.acquire(now);
    try std.testing.expect(!r.isAllowed());
    try std.testing.expect(r.denied.retry_after_ns > 0);
}

test {
    std.testing.refAllDecls(@This());
}
