const std = @import("std");
const types = @import("types.zig");

pub const RateLimitConfig = types.RateLimitConfig;
pub const RateLimitAlgorithm = types.RateLimitAlgorithm;
pub const RateLimitResult = types.RateLimitResult;

/// Token bucket rate limiter state.
const TokenBucket = struct {
    tokens: f64,
    capacity: f64,
    refill_rate: f64, // tokens per nanosecond
    last_refill_ns: u128,

    fn init(config: RateLimitConfig, now_ns: u128) TokenBucket {
        const cap = @as(f64, @floatFromInt(config.burst_size));
        const rate = @as(f64, @floatFromInt(config.requests_per_second)) /
            @as(f64, @floatFromInt(std.time.ns_per_s));
        return .{
            .tokens = cap,
            .capacity = cap,
            .refill_rate = rate,
            .last_refill_ns = now_ns,
        };
    }

    fn tryConsume(self: *TokenBucket, now_ns: u128) RateLimitResult {
        const elapsed_ns = now_ns - self.last_refill_ns;
        const added = @as(f64, @floatFromInt(elapsed_ns)) * self.refill_rate;
        self.tokens = @min(self.tokens + added, self.capacity);
        self.last_refill_ns = now_ns;

        if (self.tokens >= 1.0) {
            self.tokens -= 1.0;
            return .{
                .allowed = true,
                .remaining = @intFromFloat(@max(self.tokens, 0)),
                .reset_after_ms = 0,
            };
        }

        const deficit = 1.0 - self.tokens;
        const wait_ns = deficit / self.refill_rate;
        return .{
            .allowed = false,
            .remaining = 0,
            .reset_after_ms = @intFromFloat(wait_ns / @as(f64, std.time.ns_per_ms)),
        };
    }
};

/// Fixed window rate limiter state.
const FixedWindow = struct {
    count: u32,
    window_start_ns: u128,
    window_duration_ns: u128,
    max_requests: u32,

    fn init(config: RateLimitConfig, now_ns: u128) FixedWindow {
        return .{
            .count = 0,
            .window_start_ns = now_ns,
            .window_duration_ns = std.time.ns_per_s, // 1 second windows
            .max_requests = config.requests_per_second,
        };
    }

    fn tryConsume(self: *FixedWindow, now_ns: u128) RateLimitResult {
        if (now_ns >= self.window_start_ns + self.window_duration_ns) {
            self.count = 0;
            self.window_start_ns = now_ns;
        }

        if (self.count < self.max_requests) {
            self.count += 1;
            return .{
                .allowed = true,
                .remaining = self.max_requests - self.count,
                .reset_after_ms = 0,
            };
        }

        const remaining_ns = (self.window_start_ns + self.window_duration_ns) - now_ns;
        return .{
            .allowed = false,
            .remaining = 0,
            .reset_after_ms = @intCast(remaining_ns / std.time.ns_per_ms),
        };
    }
};

/// Sliding window rate limiter using histogram buckets.
const SlidingWindow = struct {
    buckets: [7]u32 = [_]u32{0} ** 7,
    bucket_start_ns: u128,
    bucket_width_ns: u128,
    max_requests: u32,
    window_ns: u128,

    fn init(config: RateLimitConfig, now_ns: u128) SlidingWindow {
        const window_ns: u128 = std.time.ns_per_s; // 1 second
        return .{
            .bucket_start_ns = now_ns,
            .bucket_width_ns = window_ns / 7,
            .max_requests = config.requests_per_second,
            .window_ns = window_ns,
        };
    }

    fn tryConsume(self: *SlidingWindow, now_ns: u128) RateLimitResult {
        self.advanceBuckets(now_ns);

        var total: u32 = 0;
        for (self.buckets) |b| total += b;

        if (total < self.max_requests) {
            const current_bucket = self.currentBucketIdx(now_ns);
            self.buckets[current_bucket] += 1;
            return .{
                .allowed = true,
                .remaining = self.max_requests - total - 1,
                .reset_after_ms = 0,
            };
        }

        return .{
            .allowed = false,
            .remaining = 0,
            .reset_after_ms = @intCast(self.bucket_width_ns / std.time.ns_per_ms),
        };
    }

    fn currentBucketIdx(self: *const SlidingWindow, now_ns: u128) usize {
        if (self.bucket_width_ns == 0) return 0;
        const elapsed = now_ns -| self.bucket_start_ns;
        return @intCast((elapsed / self.bucket_width_ns) % 7);
    }

    fn advanceBuckets(self: *SlidingWindow, now_ns: u128) void {
        if (self.bucket_width_ns == 0) return;
        const elapsed = now_ns -| self.bucket_start_ns;
        const buckets_to_advance = elapsed / self.bucket_width_ns;

        if (buckets_to_advance >= 7) {
            self.buckets = [_]u32{0} ** 7;
            self.bucket_start_ns = now_ns;
        } else if (buckets_to_advance > 0) {
            var i: usize = 0;
            while (i < buckets_to_advance) : (i += 1) {
                const old_idx = (self.currentBucketIdx(self.bucket_start_ns) + i) % 7;
                self.buckets[old_idx] = 0;
            }
            self.bucket_start_ns += buckets_to_advance * self.bucket_width_ns;
        }
    }
};

/// Rate limiter wrapping the 3 algorithm variants.
pub const RateLimiter = union(RateLimitAlgorithm) {
    token_bucket: TokenBucket,
    sliding_window: SlidingWindow,
    fixed_window: FixedWindow,

    pub fn init(config: RateLimitConfig, now_ns: u128) RateLimiter {
        return switch (config.algorithm) {
            .token_bucket => .{ .token_bucket = TokenBucket.init(config, now_ns) },
            .sliding_window => .{ .sliding_window = SlidingWindow.init(config, now_ns) },
            .fixed_window => .{ .fixed_window = FixedWindow.init(config, now_ns) },
        };
    }

    pub fn tryConsume(self: *RateLimiter, now_ns: u128) RateLimitResult {
        return switch (self.*) {
            .token_bucket => |*tb| tb.tryConsume(now_ns),
            .sliding_window => |*sw| sw.tryConsume(now_ns),
            .fixed_window => |*fw| fw.tryConsume(now_ns),
        };
    }
};
