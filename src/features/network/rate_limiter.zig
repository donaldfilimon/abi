//! Rate limiting for network requests.
//!
//! Provides configurable rate limiting using various algorithms
//! including token bucket, sliding window, and fixed window.

const std = @import("std");
const time = @import("../../shared/utils/time.zig");

/// Rate limiting algorithm.
pub const RateLimitAlgorithm = enum {
    /// Token bucket algorithm.
    token_bucket,
    /// Sliding window log.
    sliding_window,
    /// Fixed window counter.
    fixed_window,
    /// Leaky bucket.
    leaky_bucket,
};

/// Rate limiter configuration.
pub const RateLimiterConfig = struct {
    /// Maximum requests per window.
    max_requests: u32 = 100,
    /// Window size in nanoseconds.
    window_ns: u64 = 1_000_000_000, // 1 second
    /// Algorithm to use.
    algorithm: RateLimitAlgorithm = .token_bucket,
    /// Burst capacity (for token bucket).
    burst_capacity: ?u32 = null,
    /// Enable queueing of requests.
    enable_queue: bool = false,
    /// Maximum queue size.
    max_queue_size: usize = 100,
};

/// Result of acquire attempt.
pub const AcquireResult = union(enum) {
    /// Request allowed.
    allowed: AllowedInfo,
    /// Request denied.
    denied: DeniedInfo,
    /// Request queued.
    queued: QueuedInfo,

    pub fn isAllowed(self: AcquireResult) bool {
        return self == .allowed;
    }
};

/// Info when request is allowed.
pub const AllowedInfo = struct {
    /// Remaining requests in window.
    remaining: u32,
    /// Time until window resets (ns).
    reset_after_ns: u64,
    /// Current limit.
    limit: u32,
};

/// Info when request is denied.
pub const DeniedInfo = struct {
    /// Time to wait before retrying (ns).
    retry_after_ns: u64,
    /// Current limit.
    limit: u32,
    /// Requests made in window.
    current: u32,
};

/// Info when request is queued.
pub const QueuedInfo = struct {
    /// Position in queue.
    position: usize,
    /// Estimated wait time (ns).
    estimated_wait_ns: u64,
};

/// Rate limiter using token bucket algorithm.
pub const TokenBucketLimiter = struct {
    config: RateLimiterConfig,
    tokens: std.atomic.Value(u64),
    last_refill: std.atomic.Value(i64),
    refill_rate_ns: u64,
    mutex: std.Thread.Mutex,

    /// Initialize token bucket limiter.
    pub fn init(config: RateLimiterConfig) TokenBucketLimiter {
        const capacity = config.burst_capacity orelse config.max_requests;
        const refill_rate = config.window_ns / config.max_requests;

        return .{
            .config = config,
            .tokens = std.atomic.Value(u64).init(capacity),
            .last_refill = std.atomic.Value(i64).init(time.nowSeconds()),
            .refill_rate_ns = refill_rate,
            .mutex = .{},
        };
    }

    /// Try to acquire a token.
    pub fn acquire(self: *TokenBucketLimiter) AcquireResult {
        return self.acquireN(1);
    }

    /// Try to acquire N tokens.
    pub fn acquireN(self: *TokenBucketLimiter, n: u32) AcquireResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.refill();

        const current = self.tokens.load(.monotonic);
        if (current >= n) {
            self.tokens.store(current - n, .monotonic);
            const capacity = self.config.burst_capacity orelse self.config.max_requests;
            return .{ .allowed = .{
                .remaining = @intCast(current - n),
                .reset_after_ns = self.refill_rate_ns * n,
                .limit = capacity,
            } };
        }

        const tokens_needed = n - @as(u32, @intCast(current));
        const wait_time = self.refill_rate_ns * tokens_needed;

        return .{ .denied = .{
            .retry_after_ns = wait_time,
            .limit = self.config.max_requests,
            .current = @intCast(self.config.max_requests - current),
        } };
    }

    /// Block until tokens available.
    pub fn acquireBlocking(self: *TokenBucketLimiter) void {
        while (true) {
            const result = self.acquire();
            switch (result) {
                .allowed => return,
                .denied => |info| {
                    time.sleepNs(info.retry_after_ns);
                },
                .queued => {},
            }
        }
    }

    /// Get current token count.
    pub fn availableTokens(self: *TokenBucketLimiter) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.refill();
        return @intCast(self.tokens.load(.monotonic));
    }

    fn refill(self: *TokenBucketLimiter) void {
        const now = time.nowSeconds();
        const last = self.last_refill.load(.monotonic);
        const elapsed: u64 = @intCast(@max(0, now - last));

        if (elapsed == 0) return;

        const tokens_to_add = elapsed * 1_000_000_000 / self.refill_rate_ns;
        if (tokens_to_add > 0) {
            const capacity = self.config.burst_capacity orelse self.config.max_requests;
            const current = self.tokens.load(.monotonic);
            const new_tokens = @min(current + tokens_to_add, capacity);
            self.tokens.store(new_tokens, .monotonic);
            self.last_refill.store(now, .monotonic);
        }
    }
};

/// Rate limiter using sliding window.
pub const SlidingWindowLimiter = struct {
    config: RateLimiterConfig,
    timestamps: std.ArrayListUnmanaged(i64),
    allocator: std.mem.Allocator,
    mutex: std.Thread.Mutex,

    /// Initialize sliding window limiter.
    pub fn init(allocator: std.mem.Allocator, config: RateLimiterConfig) SlidingWindowLimiter {
        return .{
            .config = config,
            .timestamps = .{},
            .allocator = allocator,
            .mutex = .{},
        };
    }

    /// Deinitialize.
    pub fn deinit(self: *SlidingWindowLimiter) void {
        self.timestamps.deinit(self.allocator);
        self.* = undefined;
    }

    /// Try to acquire permission.
    pub fn acquire(self: *SlidingWindowLimiter) !AcquireResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.nowSeconds();
        const window_start = now - @as(i64, @intCast(self.config.window_ns / 1_000_000_000));

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
            const oldest = if (self.timestamps.items.len > 0)
                self.timestamps.items[0]
            else
                now;
            const time_diff = @max(0, oldest + @as(i64, @intCast(self.config.window_ns / 1_000_000_000)) - now);
            const reset_after: u64 = @as(u64, @intCast(time_diff)) * 1_000_000_000;

            return .{ .allowed = .{
                .remaining = remaining,
                .reset_after_ns = reset_after,
                .limit = self.config.max_requests,
            } };
        }

        // Calculate retry time
        const oldest = self.timestamps.items[0];
        const retry_diff = @max(0, oldest + @as(i64, @intCast(self.config.window_ns / 1_000_000_000)) - now);
        const retry_after: u64 = @as(u64, @intCast(retry_diff)) * 1_000_000_000;

        return .{ .denied = .{
            .retry_after_ns = retry_after,
            .limit = self.config.max_requests,
            .current = @intCast(self.timestamps.items.len),
        } };
    }

    /// Get current request count in window.
    pub fn currentCount(self: *SlidingWindowLimiter) u32 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.nowSeconds();
        const window_start = now - @as(i64, @intCast(self.config.window_ns / 1_000_000_000));

        var count: u32 = 0;
        for (self.timestamps.items) |ts| {
            if (ts >= window_start) count += 1;
        }
        return count;
    }
};

/// Rate limiter using fixed window.
pub const FixedWindowLimiter = struct {
    config: RateLimiterConfig,
    count: std.atomic.Value(u32),
    window_start: std.atomic.Value(i64),
    mutex: std.Thread.Mutex,

    /// Initialize fixed window limiter.
    pub fn init(config: RateLimiterConfig) FixedWindowLimiter {
        return .{
            .config = config,
            .count = std.atomic.Value(u32).init(0),
            .window_start = std.atomic.Value(i64).init(time.nowSeconds()),
            .mutex = .{},
        };
    }

    /// Try to acquire permission.
    pub fn acquire(self: *FixedWindowLimiter) AcquireResult {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = time.nowSeconds();
        const window_start = self.window_start.load(.monotonic);
        const window_end = window_start + @as(i64, @intCast(self.config.window_ns / 1_000_000_000));

        // Check if we need to reset window
        if (now >= window_end) {
            self.window_start.store(now, .monotonic);
            self.count.store(1, .monotonic);

            return .{ .allowed = .{
                .remaining = self.config.max_requests - 1,
                .reset_after_ns = self.config.window_ns,
                .limit = self.config.max_requests,
            } };
        }

        const current = self.count.load(.monotonic);
        if (current < self.config.max_requests) {
            self.count.store(current + 1, .monotonic);

            const reset_diff = @max(0, window_end - now);
            const reset_after: u64 = @as(u64, @intCast(reset_diff)) * 1_000_000_000;

            return .{ .allowed = .{
                .remaining = self.config.max_requests - current - 1,
                .reset_after_ns = reset_after,
                .limit = self.config.max_requests,
            } };
        }

        const retry_window_diff = @max(0, window_end - now);
        const retry_after: u64 = @as(u64, @intCast(retry_window_diff)) * 1_000_000_000;

        return .{ .denied = .{
            .retry_after_ns = retry_after,
            .limit = self.config.max_requests,
            .current = current,
        } };
    }

    /// Get current request count.
    pub fn currentCount(self: *FixedWindowLimiter) u32 {
        return self.count.load(.monotonic);
    }

    /// Reset the limiter.
    pub fn reset(self: *FixedWindowLimiter) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.count.store(0, .monotonic);
        self.window_start.store(time.nowSeconds(), .monotonic);
    }
};

/// Unified rate limiter interface.
pub const RateLimiter = struct {
    allocator: std.mem.Allocator,
    config: RateLimiterConfig,
    impl: LimiterImpl,

    const LimiterImpl = union(RateLimitAlgorithm) {
        token_bucket: TokenBucketLimiter,
        sliding_window: SlidingWindowLimiter,
        fixed_window: FixedWindowLimiter,
        leaky_bucket: TokenBucketLimiter, // Use token bucket as leaky bucket
    };

    /// Initialize rate limiter.
    pub fn init(allocator: std.mem.Allocator, config: RateLimiterConfig) RateLimiter {
        const impl: LimiterImpl = switch (config.algorithm) {
            .token_bucket => .{ .token_bucket = TokenBucketLimiter.init(config) },
            .sliding_window => .{ .sliding_window = SlidingWindowLimiter.init(allocator, config) },
            .fixed_window => .{ .fixed_window = FixedWindowLimiter.init(config) },
            .leaky_bucket => .{ .leaky_bucket = TokenBucketLimiter.init(config) },
        };

        return .{
            .allocator = allocator,
            .config = config,
            .impl = impl,
        };
    }

    /// Deinitialize.
    pub fn deinit(self: *RateLimiter) void {
        switch (self.impl) {
            .sliding_window => |*sw| sw.deinit(),
            else => {},
        }
        self.* = undefined;
    }

    /// Try to acquire permission.
    pub fn acquire(self: *RateLimiter) !AcquireResult {
        return switch (self.impl) {
            .token_bucket => |*tb| tb.acquire(),
            .sliding_window => |*sw| try sw.acquire(),
            .fixed_window => |*fw| fw.acquire(),
            .leaky_bucket => |*lb| lb.acquire(),
        };
    }

    /// Get limiter statistics.
    pub fn getStats(self: *RateLimiter) LimiterStats {
        return switch (self.impl) {
            .token_bucket => |*tb| .{
                .algorithm = .token_bucket,
                .limit = self.config.max_requests,
                .remaining = tb.availableTokens(),
                .window_ns = self.config.window_ns,
            },
            .sliding_window => |*sw| .{
                .algorithm = .sliding_window,
                .limit = self.config.max_requests,
                .remaining = self.config.max_requests -| sw.currentCount(),
                .window_ns = self.config.window_ns,
            },
            .fixed_window => |*fw| .{
                .algorithm = .fixed_window,
                .limit = self.config.max_requests,
                .remaining = self.config.max_requests -| fw.currentCount(),
                .window_ns = self.config.window_ns,
            },
            .leaky_bucket => |*lb| .{
                .algorithm = .leaky_bucket,
                .limit = self.config.max_requests,
                .remaining = lb.availableTokens(),
                .window_ns = self.config.window_ns,
            },
        };
    }
};

/// Limiter statistics.
pub const LimiterStats = struct {
    algorithm: RateLimitAlgorithm,
    limit: u32,
    remaining: u32,
    window_ns: u64,
};

test "token bucket basic" {
    var limiter = TokenBucketLimiter.init(.{
        .max_requests = 10,
        .window_ns = 1_000_000_000,
    });

    // Should allow requests up to limit
    for (0..10) |_| {
        const result = limiter.acquire();
        try std.testing.expect(result.isAllowed());
    }

    // Should deny when exhausted
    const result = limiter.acquire();
    try std.testing.expect(!result.isAllowed());
}

test "fixed window basic" {
    var limiter = FixedWindowLimiter.init(.{
        .max_requests = 5,
        .window_ns = 1_000_000_000,
    });

    for (0..5) |_| {
        const result = limiter.acquire();
        try std.testing.expect(result.isAllowed());
    }

    const result = limiter.acquire();
    try std.testing.expect(!result.isAllowed());
}

test "sliding window basic" {
    const allocator = std.testing.allocator;
    var limiter = SlidingWindowLimiter.init(allocator, .{
        .max_requests = 5,
        .window_ns = 1_000_000_000,
    });
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
    var limiter = RateLimiter.init(allocator, .{
        .max_requests = 10,
        .algorithm = .token_bucket,
    });
    defer limiter.deinit();

    const result = try limiter.acquire();
    try std.testing.expect(result.isAllowed());

    const stats = limiter.getStats();
    try std.testing.expectEqual(RateLimitAlgorithm.token_bucket, stats.algorithm);
}
