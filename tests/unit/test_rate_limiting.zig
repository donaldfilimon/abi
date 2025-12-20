//! Rate Limiting Tests for WDBX HTTP Server
//!
//! Ensures rate limiting works correctly to prevent DoS attacks

const std = @import("std");
const testing = std.testing;
const abi = @import("abi");

const security = abi.utils.security;
const ai = abi.ai;

const MockClock = struct {
    now: i64,

    fn provider(self: *MockClock) security.TimeProvider {
        return .{ .context = self, .now_fn = nowFn };
    }

    fn nowFn(context: ?*anyopaque) i64 {
        const clock: *MockClock = @ptrCast(@alignCast(context.?));
        return clock.now;
    }
};

test "basic rate limiting" {
    var clock = MockClock{ .now = 0 };
    var limiter = security.RateLimiter.initWithTimeProvider(
        testing.allocator,
        .{
            .requests_per_minute = 10,
            .window_seconds = 60,
            .max_concurrent = 10,
        },
        clock.provider(),
    );
    defer limiter.deinit();

    const test_ip: u32 = 0x7F000001; // 127.0.0.1

    // First 10 requests should pass
    for (0..10) |_| {
        try limiter.checkLimit(test_ip);
    }

    // 11th request should fail
    try testing.expectError(security.ValidationError.RateLimitExceeded, limiter.checkLimit(test_ip));

    // After window expires, should allow again
    clock.now = 61;
    try limiter.checkLimit(test_ip);
}

test "rate limiting with multiple IPs" {
    var clock = MockClock{ .now = 0 };
    var limiter = security.RateLimiter.initWithTimeProvider(
        testing.allocator,
        .{
            .requests_per_minute = 5,
            .window_seconds = 60,
            .max_concurrent = 10,
        },
        clock.provider(),
    );
    defer limiter.deinit();

    // Test different IPs
    const ip1: u32 = 0x7F000001; // 127.0.0.1
    const ip2: u32 = 0x7F000002; // 127.0.0.2

    // Both IPs should get their own limits
    for (0..5) |_| {
        try limiter.checkLimit(ip1);
        try limiter.checkLimit(ip2);
    }

    // Both should now be rate limited
    try testing.expectError(security.ValidationError.RateLimitExceeded, limiter.checkLimit(ip1));
    try testing.expectError(security.ValidationError.RateLimitExceeded, limiter.checkLimit(ip2));
}

test "windowed rate limiting resets after window" {
    var clock = MockClock{ .now = 0 };
    var limiter = security.RateLimiter.initWithTimeProvider(
        testing.allocator,
        .{
            .requests_per_minute = 3,
            .window_seconds = 1,
            .max_concurrent = 10,
        },
        clock.provider(),
    );
    defer limiter.deinit();

    // Test requests at different times
    try limiter.checkLimit(0x7F000001); // t=0
    try limiter.checkLimit(0x7F000001); // t=0
    try limiter.checkLimit(0x7F000001); // t=0
    try testing.expectError(
        security.ValidationError.RateLimitExceeded,
        limiter.checkLimit(0x7F000001),
    );

    // After window expires, should allow again
    clock.now = 2;
    try limiter.checkLimit(0x7F000001);
}

test "token bucket rate limiting" {
    var bucket = ai.limiter.TokenBucket.init(10, 1000, 0);

    // Should allow initial burst
    try testing.expect(bucket.acquire(0, 5));
    try testing.expect(bucket.acquire(0, 5));
    try testing.expect(!bucket.acquire(0, 1)); // Should fail, no tokens left

    // After 5 seconds, should have 5 new tokens
    try testing.expect(bucket.acquire(5000, 5));
}
