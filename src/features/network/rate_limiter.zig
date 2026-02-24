//! Rate limiting for network requests (nanosecond-precision, single-client).
//!
//! Delegates to the shared `services/shared/resilience/rate_limiter.zig` for core
//! algorithms (token bucket, sliding window, fixed window, leaky bucket). This
//! module preserves the existing network-specific public API and adds
//! network-specific extensions like `acquireBlocking`.
//!
//! For HTTP/API-level rate limiting with per-key tracking, bans, whitelist,
//! and auth integration, see `services/shared/security/rate_limit.zig`.

const std = @import("std");
const time = @import("../../services/shared/utils.zig");
const sync = @import("../../services/shared/sync.zig");
const shared_rl = @import("../../services/shared/resilience/rate_limiter.zig");

pub const RateLimitAlgorithm = enum { token_bucket, sliding_window, fixed_window, leaky_bucket };

pub const RateLimiterConfig = struct {
    max_requests: u32 = 100,
    window_ns: u64 = 1_000_000_000, // 1 second
    algorithm: RateLimitAlgorithm = .token_bucket,
    burst_capacity: ?u32 = null,
    enable_queue: bool = false,
    max_queue_size: usize = 100,
};

pub const AcquireResult = union(enum) {
    allowed: AllowedInfo,
    denied: DeniedInfo,
    queued: QueuedInfo,

    pub fn isAllowed(self: AcquireResult) bool {
        return self == .allowed;
    }
};

pub const AllowedInfo = struct { remaining: u32, reset_after_ns: u64, limit: u32 };
pub const DeniedInfo = struct { retry_after_ns: u64, limit: u32, current: u32 };
pub const QueuedInfo = struct { position: usize, estimated_wait_ns: u64 };

/// Shared core type (mutex-protected — network is multi-threaded).
const CoreLimiter = shared_rl.MutexRateLimiter;

/// Convert network algorithm enum to shared algorithm enum.
fn toSharedAlgorithm(algo: RateLimitAlgorithm) shared_rl.Algorithm {
    return switch (algo) {
        .token_bucket => .token_bucket,
        .sliding_window => .sliding_window,
        .fixed_window => .fixed_window,
        .leaky_bucket => .leaky_bucket,
    };
}

/// Convert network RateLimiterConfig to shared Config.
fn toSharedConfig(config: RateLimiterConfig) shared_rl.Config {
    return .{
        .max_requests = config.max_requests,
        .window_ns = config.window_ns,
        .algorithm = toSharedAlgorithm(config.algorithm),
        .burst_capacity = config.burst_capacity,
    };
}

/// Get current time as u128 nanoseconds (from shared time utilities).
fn nowNsU128() u128 {
    const ns = time.nowNanoseconds();
    return if (ns > 0) @intCast(ns) else 0;
}

/// Convert shared Result to network AcquireResult.
fn toNetworkResult(result: shared_rl.Result, config: RateLimiterConfig, avail: u32) AcquireResult {
    return switch (result) {
        .allowed => |info| .{ .allowed = .{
            .remaining = info.remaining,
            .reset_after_ns = info.reset_after_ns,
            .limit = config.burst_capacity orelse config.max_requests,
        } },
        .denied => |info| .{ .denied = .{
            .retry_after_ns = info.retry_after_ns,
            .limit = config.max_requests,
            .current = config.max_requests -| avail,
        } },
    };
}

/// Token bucket rate limiter with optional burst capacity.
pub const TokenBucketLimiter = struct {
    config: RateLimiterConfig,
    core: CoreLimiter,

    pub fn init(config: RateLimiterConfig) TokenBucketLimiter {
        return .{
            .config = config,
            .core = CoreLimiter.init(toSharedConfig(config), nowNsU128()),
        };
    }

    pub fn acquire(self: *TokenBucketLimiter) AcquireResult {
        return self.acquireN(1);
    }

    pub fn acquireN(self: *TokenBucketLimiter, n: u32) AcquireResult {
        const now = nowNsU128();
        const avail = self.core.available();
        const result = self.core.acquireN(n, now);
        return toNetworkResult(result, self.config, avail);
    }

    pub fn acquireBlocking(self: *TokenBucketLimiter) void {
        while (true) {
            switch (self.acquire()) {
                .allowed => return,
                .denied => |info| time.sleepNs(info.retry_after_ns),
                .queued => {},
            }
        }
    }

    pub fn availableTokens(self: *TokenBucketLimiter) u32 {
        return self.core.available();
    }
};

/// Sliding window rate limiter using timestamp log.
/// NOTE: The shared core uses a histogram-based sliding window (7 buckets),
/// which is more memory-efficient than the original timestamp-log approach.
/// The public API is preserved.
pub const SlidingWindowLimiter = struct {
    config: RateLimiterConfig,
    core: CoreLimiter,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, config: RateLimiterConfig) SlidingWindowLimiter {
        return .{
            .config = config,
            .core = CoreLimiter.init(
                toSharedConfig(.{ .max_requests = config.max_requests, .window_ns = config.window_ns, .algorithm = .sliding_window, .burst_capacity = config.burst_capacity }),
                nowNsU128(),
            ),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SlidingWindowLimiter) void {
        // Shared core has no allocations to free for sliding window.
        self.* = undefined;
    }

    pub fn acquire(self: *SlidingWindowLimiter) !AcquireResult {
        const now = nowNsU128();
        const avail = self.core.available();
        const result = self.core.acquire(now);
        return toNetworkResult(result, self.config, avail);
    }

    pub fn currentCount(self: *SlidingWindowLimiter) u32 {
        return self.config.max_requests -| self.core.available();
    }
};

/// Fixed window rate limiter with atomic counter.
pub const FixedWindowLimiter = struct {
    config: RateLimiterConfig,
    core: CoreLimiter,

    pub fn init(config: RateLimiterConfig) FixedWindowLimiter {
        return .{
            .config = config,
            .core = CoreLimiter.init(
                toSharedConfig(.{ .max_requests = config.max_requests, .window_ns = config.window_ns, .algorithm = .fixed_window, .burst_capacity = config.burst_capacity }),
                nowNsU128(),
            ),
        };
    }

    pub fn acquire(self: *FixedWindowLimiter) AcquireResult {
        const now = nowNsU128();
        const avail = self.core.available();
        const result = self.core.acquire(now);
        return toNetworkResult(result, self.config, avail);
    }

    pub fn currentCount(self: *FixedWindowLimiter) u32 {
        return self.config.max_requests -| self.core.available();
    }

    pub fn reset(self: *FixedWindowLimiter) void {
        self.core.reset(nowNsU128());
    }
};

/// Unified rate limiter that dispatches to the algorithm selected by config.
pub const RateLimiter = struct {
    allocator: std.mem.Allocator,
    config: RateLimiterConfig,
    impl: LimiterImpl,

    const LimiterImpl = union(RateLimitAlgorithm) {
        token_bucket: TokenBucketLimiter,
        sliding_window: SlidingWindowLimiter,
        fixed_window: FixedWindowLimiter,
        leaky_bucket: TokenBucketLimiter, // leaky bucket uses token bucket impl
    };

    pub fn init(allocator: std.mem.Allocator, config: RateLimiterConfig) RateLimiter {
        return .{
            .allocator = allocator,
            .config = config,
            .impl = switch (config.algorithm) {
                .token_bucket => .{ .token_bucket = TokenBucketLimiter.init(config) },
                .sliding_window => .{ .sliding_window = SlidingWindowLimiter.init(allocator, config) },
                .fixed_window => .{ .fixed_window = FixedWindowLimiter.init(config) },
                .leaky_bucket => .{ .leaky_bucket = TokenBucketLimiter.init(.{
                    .max_requests = config.max_requests,
                    .window_ns = config.window_ns,
                    .algorithm = .leaky_bucket,
                    .burst_capacity = config.burst_capacity,
                }) },
            },
        };
    }

    pub fn deinit(self: *RateLimiter) void {
        switch (self.impl) {
            .sliding_window => |*sw| sw.deinit(),
            else => {},
        }
        self.* = undefined;
    }

    pub fn acquire(self: *RateLimiter) !AcquireResult {
        return switch (self.impl) {
            .token_bucket => |*tb| tb.acquire(),
            .sliding_window => |*sw| try sw.acquire(),
            .fixed_window => |*fw| fw.acquire(),
            .leaky_bucket => |*lb| lb.acquire(),
        };
    }

    pub fn getStats(self: *RateLimiter) LimiterStats {
        return switch (self.impl) {
            .token_bucket => |*tb| .{ .algorithm = .token_bucket, .limit = self.config.max_requests, .remaining = tb.availableTokens(), .window_ns = self.config.window_ns },
            .sliding_window => |*sw| .{ .algorithm = .sliding_window, .limit = self.config.max_requests, .remaining = self.config.max_requests -| sw.currentCount(), .window_ns = self.config.window_ns },
            .fixed_window => |*fw| .{ .algorithm = .fixed_window, .limit = self.config.max_requests, .remaining = self.config.max_requests -| fw.currentCount(), .window_ns = self.config.window_ns },
            .leaky_bucket => |*lb| .{ .algorithm = .leaky_bucket, .limit = self.config.max_requests, .remaining = lb.availableTokens(), .window_ns = self.config.window_ns },
        };
    }
};

pub const LimiterStats = struct { algorithm: RateLimitAlgorithm, limit: u32, remaining: u32, window_ns: u64 };

test "token bucket basic" {
    var limiter = TokenBucketLimiter.init(.{ .max_requests = 10, .window_ns = 1_000_000_000 });
    for (0..10) |_| {
        const result = limiter.acquire();
        try std.testing.expect(result.isAllowed());
    }
    const result = limiter.acquire();
    try std.testing.expect(!result.isAllowed());
}

test "fixed window basic" {
    var limiter = FixedWindowLimiter.init(.{ .max_requests = 5, .window_ns = 1_000_000_000 });
    for (0..5) |_| {
        const result = limiter.acquire();
        try std.testing.expect(result.isAllowed());
    }
    const result = limiter.acquire();
    try std.testing.expect(!result.isAllowed());
}

test "sliding window basic" {
    const allocator = std.testing.allocator;
    var limiter = SlidingWindowLimiter.init(allocator, .{ .max_requests = 5, .window_ns = 1_000_000_000 });
    defer limiter.deinit();
    for (0..5) |_| {
        const result = try limiter.acquire();
        try std.testing.expect(result.isAllowed());
    }
    const result = try limiter.acquire();
    try std.testing.expect(!result.isAllowed());
}

test "rate limiter unified" {
    const allocator = std.testing.allocator;
    var limiter = RateLimiter.init(allocator, .{ .max_requests = 10, .algorithm = .token_bucket });
    defer limiter.deinit();
    const result = try limiter.acquire();
    try std.testing.expect(result.isAllowed());
    const stats = limiter.getStats();
    try std.testing.expectEqual(RateLimitAlgorithm.token_bucket, stats.algorithm);
}

test {
    std.testing.refAllDecls(@This());
}
