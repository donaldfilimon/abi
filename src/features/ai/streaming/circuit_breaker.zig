//! Circuit breaker for streaming backends.
//!
//! Implements the circuit breaker pattern to prevent cascading failures
//! when streaming backends become unhealthy. Uses lock-free atomic
//! operations for high-performance state management.
//!
//! ## State Machine
//!
//! ```
//!                    +------------------+
//!                    |     CLOSED       |
//!                    |  (normal flow)   |
//!                    +--------+---------+
//!                             |
//!                  failure_count >= failure_threshold
//!                             |
//!                             v
//!                    +------------------+
//!             +----->|      OPEN        |<-----+
//!             |      | (reject all)     |      |
//!             |      +--------+---------+      |
//!             |               |                |
//!             |    timeout_ms elapsed          |
//!             |               |                |
//!             |               v                |
//!             |      +------------------+      |
//!             |      |    HALF_OPEN     |      |
//!             |      | (test requests)  |------+
//!             |      +--------+---------+   any failure
//!             |               |
//!             |    success_count >= success_threshold
//!             |               |
//!             +---------------+
//!                   (reset)
//! ```
//!
//! ## States
//!
//! - **Closed**: Normal operation, requests pass through. Failures increment
//!   `failure_count`. A success resets the counter. When `failure_count`
//!   reaches `failure_threshold`, transitions to Open.
//!
//! - **Open**: Circuit tripped. All requests are rejected immediately with
//!   `error.CircuitBreakerOpen`. After `timeout_ms` elapses since the last
//!   failure, transitions to Half-Open to test recovery.
//!
//! - **Half-Open**: Recovery testing phase. Allows up to `half_open_max_requests`
//!   probe requests. If `success_threshold` consecutive successes occur,
//!   transitions back to Closed. Any single failure immediately returns to Open.
//!
//! ## Configuration Parameters
//!
//! | Parameter               | Default | Description                                       |
//! |-------------------------|---------|---------------------------------------------------|
//! | `failure_threshold`     | 5       | Consecutive failures to open circuit              |
//! | `success_threshold`     | 2       | Successes in half-open to close circuit           |
//! | `timeout_ms`            | 60000   | Time in open state before testing recovery        |
//! | `half_open_max_requests`| 3       | Max concurrent requests allowed in half-open      |
//!
//! ## Usage
//!
//! ```zig
//! var cb = CircuitBreaker.init(.{ .failure_threshold = 5 });
//!
//! if (cb.canAttempt()) {
//!     const result = backend.streamTokens(prompt, config) catch |err| {
//!         cb.recordFailure();
//!         return err;
//!     };
//!     cb.recordSuccess();
//!     return result;
//! } else {
//!     return error.CircuitBreakerOpen;
//! }
//! ```
//!
//! ## Error Handling
//!
//! When `canAttempt()` returns false, the circuit is open. Callers should:
//! - Return `error.CircuitBreakerOpen` to propagate the rejection
//! - Implement fallback logic (cached response, alternative backend)
//! - Log/metric the rejection for monitoring
//!
//! ## Thread Safety
//!
//! All operations are lock-free using atomic compare-and-swap. Safe for
//! concurrent access from multiple threads without external synchronization.

const std = @import("std");
const platform_time = @import("../../../services/shared/time.zig");

/// Circuit breaker state.
pub const CircuitState = enum(u8) {
    /// Normal operation - requests pass through.
    closed = 0,
    /// Too many failures - requests rejected immediately.
    open = 1,
    /// Testing recovery - limited requests allowed.
    half_open = 2,

    pub fn format(
        self: CircuitState,
        comptime _: []const u8,
        _: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        const name = switch (self) {
            .closed => "closed",
            .open => "open",
            .half_open => "half_open",
        };
        try writer.writeAll(name);
    }
};

/// Circuit breaker configuration.
pub const CircuitBreakerConfig = struct {
    /// Number of consecutive failures before opening circuit.
    failure_threshold: u32 = 5,

    /// Number of successes needed to close circuit from half-open.
    success_threshold: u32 = 2,

    /// Time to wait before transitioning from open to half-open (milliseconds).
    timeout_ms: u64 = 60_000,

    /// Maximum requests allowed in half-open state.
    half_open_max_requests: u32 = 3,

    /// Name for logging/metrics (optional).
    name: []const u8 = "default",

    /// Optional time provider for deterministic tests.
    time_provider: ?platform_time.TimeProvider = null,
};

/// Statistics about circuit breaker behavior.
pub const CircuitStats = struct {
    /// Total requests attempted.
    total_requests: u64 = 0,
    /// Successful requests.
    successful_requests: u64 = 0,
    /// Failed requests.
    failed_requests: u64 = 0,
    /// Requests rejected due to open circuit.
    rejected_requests: u64 = 0,
    /// Number of times circuit opened.
    times_opened: u64 = 0,
    /// Current state.
    current_state: CircuitState = .closed,
};

/// Lock-free circuit breaker implementation.
pub const CircuitBreaker = struct {
    config: CircuitBreakerConfig,
    time_provider: platform_time.TimeProvider,
    state: std.atomic.Value(u8),
    failure_count: std.atomic.Value(u32),
    success_count: std.atomic.Value(u32),
    half_open_requests: std.atomic.Value(u32),
    last_failure_time_ms: std.atomic.Value(i64),

    // Statistics (relaxed ordering ok for stats)
    stats_total: std.atomic.Value(u64),
    stats_success: std.atomic.Value(u64),
    stats_failed: std.atomic.Value(u64),
    stats_rejected: std.atomic.Value(u64),
    stats_opened: std.atomic.Value(u64),

    const Self = @This();

    /// Initialize a new circuit breaker.
    pub fn init(config: CircuitBreakerConfig) Self {
        const time_provider = config.time_provider orelse platform_time.TimeProvider{};
        return .{
            .config = config,
            .time_provider = time_provider,
            .state = std.atomic.Value(u8).init(@intFromEnum(CircuitState.closed)),
            .failure_count = std.atomic.Value(u32).init(0),
            .success_count = std.atomic.Value(u32).init(0),
            .half_open_requests = std.atomic.Value(u32).init(0),
            .last_failure_time_ms = std.atomic.Value(i64).init(0),
            .stats_total = std.atomic.Value(u64).init(0),
            .stats_success = std.atomic.Value(u64).init(0),
            .stats_failed = std.atomic.Value(u64).init(0),
            .stats_rejected = std.atomic.Value(u64).init(0),
            .stats_opened = std.atomic.Value(u64).init(0),
        };
    }

    /// Check if a request can be attempted.
    ///
    /// Returns true if the circuit allows the request, false if rejected.
    /// In half-open state, only a limited number of requests are allowed.
    pub fn canAttempt(self: *Self) bool {
        const current_state = self.getState();

        switch (current_state) {
            .closed => {
                _ = self.stats_total.fetchAdd(1, .monotonic);
                return true;
            },
            .open => {
                // Check if timeout expired - should transition to half-open
                const now = self.time_provider.nowMs();
                const last_failure = self.last_failure_time_ms.load(.monotonic);
                const timeout = @as(i64, @intCast(self.config.timeout_ms));

                if (now - last_failure >= timeout) {
                    // Try to transition to half-open
                    const open_val = @intFromEnum(CircuitState.open);
                    const half_open_val = @intFromEnum(CircuitState.half_open);

                    if (self.state.cmpxchgStrong(
                        open_val,
                        half_open_val,
                        .acq_rel,
                        .monotonic,
                    ) == null) {
                        // Successfully transitioned to half-open
                        self.success_count.store(0, .monotonic);
                        // Start at 1: the transition request itself counts
                        self.half_open_requests.store(1, .monotonic);
                        _ = self.stats_total.fetchAdd(1, .monotonic);
                        return true;
                    }
                    // Someone else transitioned, re-check
                    return self.canAttempt();
                }

                // Circuit is open and timeout not expired
                _ = self.stats_rejected.fetchAdd(1, .monotonic);
                return false;
            },
            .half_open => {
                // Allow limited requests in half-open state
                const current = self.half_open_requests.fetchAdd(1, .monotonic);
                if (current < self.config.half_open_max_requests) {
                    _ = self.stats_total.fetchAdd(1, .monotonic);
                    return true;
                }
                // Undo the increment and reject
                _ = self.half_open_requests.fetchSub(1, .monotonic);
                _ = self.stats_rejected.fetchAdd(1, .monotonic);
                return false;
            },
        }
    }

    /// Record a successful operation.
    ///
    /// In closed state: resets failure count.
    /// In half-open state: may transition to closed after enough successes.
    pub fn recordSuccess(self: *Self) void {
        _ = self.stats_success.fetchAdd(1, .monotonic);
        const current_state = self.getState();

        switch (current_state) {
            .closed => {
                // Reset failure count on success
                self.failure_count.store(0, .monotonic);
            },
            .half_open => {
                const successes = self.success_count.fetchAdd(1, .monotonic) + 1;
                if (successes >= self.config.success_threshold) {
                    // Transition to closed
                    const half_open_val = @intFromEnum(CircuitState.half_open);
                    const closed_val = @intFromEnum(CircuitState.closed);
                    _ = self.state.cmpxchgStrong(
                        half_open_val,
                        closed_val,
                        .acq_rel,
                        .monotonic,
                    );
                    self.failure_count.store(0, .monotonic);
                }
            },
            .open => {
                // Shouldn't happen - we rejected in canAttempt
            },
        }
    }

    /// Record a failed operation.
    ///
    /// In closed state: may transition to open after enough failures.
    /// In half-open state: immediately transitions back to open.
    pub fn recordFailure(self: *Self) void {
        _ = self.stats_failed.fetchAdd(1, .monotonic);
        const now = self.time_provider.nowMs();
        self.last_failure_time_ms.store(now, .monotonic);

        const current_state = self.getState();

        switch (current_state) {
            .closed => {
                const failures = self.failure_count.fetchAdd(1, .monotonic) + 1;
                if (failures >= self.config.failure_threshold) {
                    // Transition to open
                    const closed_val = @intFromEnum(CircuitState.closed);
                    const open_val = @intFromEnum(CircuitState.open);
                    if (self.state.cmpxchgStrong(
                        closed_val,
                        open_val,
                        .acq_rel,
                        .monotonic,
                    ) == null) {
                        _ = self.stats_opened.fetchAdd(1, .monotonic);
                    }
                }
            },
            .half_open => {
                // Any failure in half-open immediately opens circuit
                const half_open_val = @intFromEnum(CircuitState.half_open);
                const open_val = @intFromEnum(CircuitState.open);
                if (self.state.cmpxchgStrong(
                    half_open_val,
                    open_val,
                    .acq_rel,
                    .monotonic,
                ) == null) {
                    _ = self.stats_opened.fetchAdd(1, .monotonic);
                }
                self.success_count.store(0, .monotonic);
            },
            .open => {
                // Already open, just update failure time
            },
        }
    }

    /// Get the current circuit state.
    pub fn getState(self: *const Self) CircuitState {
        return @enumFromInt(self.state.load(.monotonic));
    }

    /// Manually reset the circuit breaker to closed state.
    pub fn reset(self: *Self) void {
        self.state.store(@intFromEnum(CircuitState.closed), .monotonic);
        self.failure_count.store(0, .monotonic);
        self.success_count.store(0, .monotonic);
        self.half_open_requests.store(0, .monotonic);
    }

    /// Get statistics about circuit breaker behavior.
    pub fn getStats(self: *const Self) CircuitStats {
        return .{
            .total_requests = self.stats_total.load(.monotonic),
            .successful_requests = self.stats_success.load(.monotonic),
            .failed_requests = self.stats_failed.load(.monotonic),
            .rejected_requests = self.stats_rejected.load(.monotonic),
            .times_opened = self.stats_opened.load(.monotonic),
            .current_state = self.getState(),
        };
    }

    /// Check if circuit is currently allowing requests.
    pub fn isAllowing(self: *const Self) bool {
        return self.getState() != .open;
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
    return .{
        .ctx = @ptrCast(clock),
        .nowMsFn = TestClock.nowMs,
    };
}

test "CircuitBreaker initial state is closed" {
    var cb = CircuitBreaker.init(.{});

    try std.testing.expectEqual(CircuitState.closed, cb.getState());
    try std.testing.expect(cb.canAttempt());
    try std.testing.expect(cb.isAllowing());
}

test "CircuitBreaker transitions to open after failures" {
    var cb = CircuitBreaker.init(.{ .failure_threshold = 3 });

    // Should start closed
    try std.testing.expectEqual(CircuitState.closed, cb.getState());

    // Record failures
    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.closed, cb.getState());

    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.closed, cb.getState());

    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();

    // Should now be open
    try std.testing.expectEqual(CircuitState.open, cb.getState());
    try std.testing.expect(!cb.canAttempt());
}

test "CircuitBreaker success resets failure count" {
    var cb = CircuitBreaker.init(.{ .failure_threshold = 3 });

    // Two failures
    _ = cb.canAttempt();
    cb.recordFailure();
    _ = cb.canAttempt();
    cb.recordFailure();

    // One success resets
    _ = cb.canAttempt();
    cb.recordSuccess();

    // Two more failures shouldn't trip
    _ = cb.canAttempt();
    cb.recordFailure();
    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.closed, cb.getState());

    // Third failure trips
    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());
}

test "CircuitBreaker half-open allows limited requests" {
    var clock = TestClock{};
    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .half_open_max_requests = 2,
        .timeout_ms = 1, // 1ms timeout for test
        .time_provider = testTimeProvider(&clock),
    });

    // Trip to open
    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());

    // Wait for timeout
    clock.now_ms += 5;

    // Should transition to half-open
    try std.testing.expect(cb.canAttempt());
    try std.testing.expectEqual(CircuitState.half_open, cb.getState());

    // Second request allowed
    try std.testing.expect(cb.canAttempt());

    // Third request rejected (max is 2)
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

    // Trip to open
    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());

    // Wait for timeout
    clock.now_ms += 5;

    // Transition to half-open
    try std.testing.expect(cb.canAttempt());
    try std.testing.expectEqual(CircuitState.half_open, cb.getState());
    cb.recordSuccess();

    _ = cb.canAttempt();
    cb.recordSuccess();

    // Should now be closed
    try std.testing.expectEqual(CircuitState.closed, cb.getState());
}

test "CircuitBreaker half-open to open on failure" {
    var clock = TestClock{};
    var cb = CircuitBreaker.init(.{
        .failure_threshold = 1,
        .timeout_ms = 1,
        .time_provider = testTimeProvider(&clock),
    });

    // Trip to open
    _ = cb.canAttempt();
    cb.recordFailure();

    // Wait for timeout
    clock.now_ms += 5;

    // Transition to half-open
    try std.testing.expect(cb.canAttempt());
    try std.testing.expectEqual(CircuitState.half_open, cb.getState());

    // Failure immediately opens
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

    // Circuit should be open, next attempt rejected
    _ = cb.canAttempt();

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

    // Trip to open
    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());

    // Reset
    cb.reset();
    try std.testing.expectEqual(CircuitState.closed, cb.getState());
    try std.testing.expect(cb.canAttempt());
}
