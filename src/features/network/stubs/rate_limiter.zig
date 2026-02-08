const std = @import("std");

pub const RateLimiter = struct {
    pub fn init(_: std.mem.Allocator, _: RateLimiterConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const RateLimiterConfig = struct {
    algorithm: RateLimitAlgorithm = .token_bucket,
    max_tokens: u64 = 100,
    refill_rate: f64 = 10.0,
};

pub const RateLimitAlgorithm = enum { token_bucket, sliding_window, fixed_window };

pub const AcquireResult = struct {
    allowed: bool = false,
    remaining: u64 = 0,
    retry_after_ms: ?u64 = null,
};

pub const TokenBucketLimiter = struct {
    pub fn init(_: u64, _: f64) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};

pub const SlidingWindowLimiter = struct {
    pub fn init(_: u64, _: u64) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};

pub const FixedWindowLimiter = struct {
    pub fn init(_: u64, _: u64) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};

pub const LimiterStats = struct {
    total_requests: u64 = 0,
    allowed_requests: u64 = 0,
    rejected_requests: u64 = 0,
};
