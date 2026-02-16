//! Circuit breaker for streaming backends — wrapper around shared resilience module.
//!
//! Uses lock-free atomic operations for high-performance state management
//! in streaming pipelines.

const std = @import("std");
const platform_time = @import("../../../services/shared/time.zig");
const resilience = @import("../../../services/shared/resilience/circuit_breaker.zig");

// Re-export shared types
pub const CircuitState = resilience.CircuitState;
pub const CircuitBreakerConfig = resilience.Config;
pub const CircuitStats = resilience.Stats;

/// Lock-free circuit breaker for streaming backends.
pub const CircuitBreaker = struct {
    inner: resilience.AtomicCircuitBreaker,

    const Self = @This();

    pub fn init(config: CircuitBreakerConfig) Self {
        return .{ .inner = resilience.AtomicCircuitBreaker.init(config) };
    }

    pub fn canAttempt(self: *Self) bool {
        return self.inner.canAttempt();
    }
    pub fn recordSuccess(self: *Self) void {
        self.inner.recordSuccess();
    }
    pub fn recordFailure(self: *Self) void {
        self.inner.recordFailure();
    }
    pub fn getState(self: *const Self) CircuitState {
        return self.inner.getState();
    }
    pub fn reset(self: *Self) void {
        self.inner.reset();
    }
    pub fn getStats(self: *const Self) CircuitStats {
        return self.inner.getStats();
    }
    pub fn isAllowing(self: *const Self) bool {
        return self.inner.isAllowing();
    }
};

// ============================================================================
// Tests
// ============================================================================

const TestClock = struct {
    now_ms: i64 = 0,

    fn nowMs(ctx: ?*anyopaque) i64 {
        const clock: *TestClock = @ptrCast(@alignCast(ctx.?));
        return clock.now_ms;
    }
};

fn testTimeProvider(clock: *TestClock) platform_time.TimeProvider {
    return .{ .ctx = @ptrCast(clock), .nowMsFn = TestClock.nowMs };
}

test "CircuitBreaker initial state is closed" {
    var cb = CircuitBreaker.init(.{});
    try std.testing.expectEqual(CircuitState.closed, cb.getState());
    try std.testing.expect(cb.canAttempt());
    try std.testing.expect(cb.isAllowing());
}

test "CircuitBreaker transitions to open after failures" {
    var cb = CircuitBreaker.init(.{ .failure_threshold = 3 });

    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();
    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();
    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();

    try std.testing.expectEqual(CircuitState.open, cb.getState());
    try std.testing.expect(!cb.canAttempt());
}

test "CircuitBreaker success resets failure count" {
    var cb = CircuitBreaker.init(.{ .failure_threshold = 3 });

    _ = cb.canAttempt();
    cb.recordFailure();
    _ = cb.canAttempt();
    cb.recordFailure();
    _ = cb.canAttempt();
    cb.recordSuccess();

    _ = cb.canAttempt();
    cb.recordFailure();
    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.closed, cb.getState());

    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());
}

test "CircuitBreaker half-open allows limited requests" {
    var clock = TestClock{};
    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .half_open_max_requests = 2,
        .timeout_ms = 1,
        .time_provider = testTimeProvider(&clock),
    });

    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());

    clock.now_ms += 5;
    try std.testing.expect(cb.canAttempt());
    try std.testing.expectEqual(CircuitState.half_open, cb.getState());
    try std.testing.expect(cb.canAttempt());
    try std.testing.expect(!cb.canAttempt());
}

test "CircuitBreaker half-open to closed on successes" {
    var clock = TestClock{};
    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .success_threshold = 2,
        .half_open_max_requests = 5,
        .timeout_ms = 1,
        .time_provider = testTimeProvider(&clock),
    });

    _ = cb.canAttempt();
    cb.recordFailure();

    clock.now_ms += 5;
    try std.testing.expect(cb.canAttempt());
    cb.recordSuccess();
    _ = cb.canAttempt();
    cb.recordSuccess();

    try std.testing.expectEqual(CircuitState.closed, cb.getState());
}

test "CircuitBreaker half-open to open on failure" {
    var clock = TestClock{};
    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .timeout_ms = 1,
        .time_provider = testTimeProvider(&clock),
    });

    _ = cb.canAttempt();
    cb.recordFailure();

    clock.now_ms += 5;
    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());
}

test "CircuitBreaker stats tracking" {
    var cb = CircuitBreaker.init(.{ .failure_threshold = 2 });

    _ = cb.canAttempt();
    cb.recordSuccess();
    _ = cb.canAttempt();
    cb.recordFailure();
    _ = cb.canAttempt();
    cb.recordFailure();
    _ = cb.canAttempt(); // rejected

    const stats = cb.getStats();
    try std.testing.expectEqual(@as(u64, 3), stats.total_requests);
    try std.testing.expectEqual(@as(u64, 1), stats.successful_requests);
    try std.testing.expectEqual(@as(u64, 2), stats.failed_requests);
    try std.testing.expectEqual(@as(u64, 1), stats.rejected_requests);
    try std.testing.expectEqual(@as(u64, 1), stats.times_opened);
    try std.testing.expectEqual(CircuitState.open, stats.current_state);
}

test "CircuitBreaker reset" {
    var cb = CircuitBreaker.init(.{ .failure_threshold = 1 });

    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());

    cb.reset();
    try std.testing.expectEqual(CircuitState.closed, cb.getState());
    try std.testing.expect(cb.canAttempt());
}
