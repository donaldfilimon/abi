//! Rate limiting for network requests (nanosecond-precision, single-client).
//!
//! For HTTP/API-level rate limiting with per-key tracking, bans, whitelist,
//! and auth integration, see `services/shared/security/rate_limit.zig`.
//!
//! This module provides lightweight, per-connection rate limiters using
//! token bucket, sliding window, and fixed window algorithms.

const std = @import("std");
const time = @import("../../services/shared/utils.zig");
const sync = @import("../../services/shared/sync.zig");

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

/// Token bucket rate limiter with optional burst capacity.
pub const TokenBucketLimiter = struct {
    config: RateLimiterConfig,
    tokens: std.atomic.Value(u64),
    last_refill: std.atomic.Value(i64),
    refill_rate_ns: u64,
    mutex: sync.Mutex,

    pub fn init(config: RateLimiterConfig) TokenBucketLimiter {
        const capacity = config.burst_capacity orelse config.max_requests;
        return .{
            .config = config,
            .tokens = std.atomic.Value(u64).init(capacity),
            .last_refill = std.atomic.Value(i64).init(time.nowNanoseconds()),
            .refill_rate_ns = config.window_ns / config.max_requests,
            .mutex = .{},
        };
    }

    pub fn acquire(self: *TokenBucketLimiter) AcquireResult {
        return self.acquireN(1);
    }

    pub fn acquireN(self: *TokenBucketLimiter, n: u32) AcquireResult {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.refill();

        const current = self.tokens.load(.monotonic);
        if (current >= n) {
            self.tokens.store(current - n, .monotonic);
            const capacity = self.config.burst_capacity orelse self.config.max_requests;
            return .{ .allowed = .{ .remaining = @intCast(current - n), .reset_after_ns = self.refill_rate_ns * n, .limit = capacity } };
        }

        const tokens_needed = n - @as(u32, @intCast(current));
        return .{ .denied = .{ .retry_after_ns = self.refill_rate_ns * tokens_needed, .limit = self.config.max_requests, .current = @intCast(self.config.max_requests - current) } };
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
        self.mutex.lock();
        defer self.mutex.unlock();
        self.refill();
        return @intCast(self.tokens.load(.monotonic));
    }

    fn refill(self: *TokenBucketLimiter) void {
        const now = time.nowNanoseconds();
        const last = self.last_refill.load(.monotonic);
        const elapsed_ns: u64 = @intCast(@max(0, now - last));
        if (elapsed_ns == 0) return;

        const tokens_to_add = elapsed_ns / self.refill_rate_ns;
        if (tokens_to_add > 0) {
            const capacity = self.config.burst_capacity orelse self.config.max_requests;
            const current = self.tokens.load(.monotonic);
            self.tokens.store(@min(current + tokens_to_add, capacity), .monotonic);
            self.last_refill.store(now, .monotonic);
        }
    }
};

/// Sliding window rate limiter using timestamp log.
pub const SlidingWindowLimiter = struct {
    config: RateLimiterConfig,
    timestamps: std.ArrayListUnmanaged(i64),
    allocator: std.mem.Allocator,
    mutex: sync.Mutex,

    pub fn init(allocator: std.mem.Allocator, config: RateLimiterConfig) SlidingWindowLimiter {
        return .{ .config = config, .timestamps = .{}, .allocator = allocator, .mutex = .{} };
    }

    pub fn deinit(self: *SlidingWindowLimiter) void {
        self.timestamps.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn acquire(self: *SlidingWindowLimiter) !AcquireResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.nowNanoseconds();
        const window_start = now - @as(i64, @intCast(self.config.window_ns));

        // Remove expired timestamps
        var i: usize = 0;
        while (i < self.timestamps.items.len) {
            if (self.timestamps.items[i] < window_start) {
                _ = self.timestamps.orderedRemove(i);
            } else {
                i += 1;
            }
        }

        if (self.timestamps.items.len < self.config.max_requests) {
            try self.timestamps.append(self.allocator, now);
            const remaining: u32 = @intCast(self.config.max_requests - self.timestamps.items.len);
            const oldest = if (self.timestamps.items.len > 0) self.timestamps.items[0] else now;
            const reset_after: u64 = @intCast(@max(0, oldest + @as(i64, @intCast(self.config.window_ns)) - now));
            return .{ .allowed = .{ .remaining = remaining, .reset_after_ns = reset_after, .limit = self.config.max_requests } };
        }

        const oldest = self.timestamps.items[0];
        const retry_after: u64 = @intCast(@max(0, oldest + @as(i64, @intCast(self.config.window_ns)) - now));
        return .{ .denied = .{ .retry_after_ns = retry_after, .limit = self.config.max_requests, .current = @intCast(self.timestamps.items.len) } };
    }

    pub fn currentCount(self: *SlidingWindowLimiter) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        const now = time.nowNanoseconds();
        const window_start = now - @as(i64, @intCast(self.config.window_ns));
        var count: u32 = 0;
        for (self.timestamps.items) |ts| {
            if (ts >= window_start) count += 1;
        }
        return count;
    }
};

/// Fixed window rate limiter with atomic counter.
pub const FixedWindowLimiter = struct {
    config: RateLimiterConfig,
    count: std.atomic.Value(u32),
    window_start: std.atomic.Value(i64),
    mutex: sync.Mutex,

    pub fn init(config: RateLimiterConfig) FixedWindowLimiter {
        return .{
            .config = config,
            .count = std.atomic.Value(u32).init(0),
            .window_start = std.atomic.Value(i64).init(time.nowNanoseconds()),
            .mutex = .{},
        };
    }

    pub fn acquire(self: *FixedWindowLimiter) AcquireResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.nowNanoseconds();
        const ws = self.window_start.load(.monotonic);
        const window_end = ws + @as(i64, @intCast(self.config.window_ns));

        if (now >= window_end) {
            self.window_start.store(now, .monotonic);
            self.count.store(1, .monotonic);
            return .{ .allowed = .{ .remaining = self.config.max_requests - 1, .reset_after_ns = self.config.window_ns, .limit = self.config.max_requests } };
        }

        const current = self.count.load(.monotonic);
        if (current < self.config.max_requests) {
            self.count.store(current + 1, .monotonic);
            const reset_after: u64 = @intCast(@max(0, window_end - now));
            return .{ .allowed = .{ .remaining = self.config.max_requests - current - 1, .reset_after_ns = reset_after, .limit = self.config.max_requests } };
        }

        return .{ .denied = .{ .retry_after_ns = @intCast(@max(0, window_end - now)), .limit = self.config.max_requests, .current = current } };
    }

    pub fn currentCount(self: *FixedWindowLimiter) u32 {
        return self.count.load(.monotonic);
    }

    pub fn reset(self: *FixedWindowLimiter) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.count.store(0, .monotonic);
        self.window_start.store(time.nowNanoseconds(), .monotonic);
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
                .leaky_bucket => .{ .leaky_bucket = TokenBucketLimiter.init(config) },
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

test "token bucket sub-second refill" {
    var limiter = TokenBucketLimiter.init(.{ .max_requests = 10, .window_ns = 100_000_000 });
    _ = limiter.acquireN(10);
    try std.testing.expectEqual(@as(u32, 0), limiter.availableTokens());

    const now = time.nowNanoseconds();
    const advance_ns: u64 = limiter.refill_rate_ns * 5;
    limiter.last_refill.store(now - @as(i64, @intCast(advance_ns)), .monotonic);
    limiter.refill();
    try std.testing.expect(limiter.availableTokens() >= 5);
}
