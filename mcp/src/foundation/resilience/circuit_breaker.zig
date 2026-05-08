//! Unified circuit breaker for fault tolerance.
//!
//! Implements the circuit breaker pattern to prevent cascading failures.
//! When failures reach a threshold, the circuit "opens" to fail fast and
//! allow the system to recover.
//!
//! ## States
//!
//! - **Closed**: Normal operation, requests pass through.
//! - **Open**: Failures exceeded threshold, requests rejected immediately.
//! - **HalfOpen**: Testing recovery, limited requests allowed.
//!
//! ## Sync Strategies
//!
//! - `.atomic`: Lock-free atomics for high-throughput paths (streaming).
//! - `.mutex`: Mutex-protected for windowed failure counting (network).
//! - `.none`: No synchronization for single-threaded use (gateway).
//!
//! ## Usage
//!
//! ```zig
//! var cb = CircuitBreaker(.atomic).init(.{ .failure_threshold = 5 });
//!
//! if (cb.canAttempt()) {
//!     const result = doWork() catch |err| {
//!         cb.recordFailure();
//!         return err;
//!     };
//!     cb.recordSuccess();
//!     return result;
//! } else {
//!     return error.CircuitBreakerOpen;
//! }
//! ```

const std = @import("std");
const platform_time = @import("../time.zig");
const sync = @import("../sync.zig");

/// Circuit breaker state.
pub const CircuitState = enum(u8) {
    closed = 0,
    open = 1,
    half_open = 2,

    pub fn format(self: CircuitState, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        try writer.writeAll(switch (self) {
            .closed => "closed",
            .open => "open",
            .half_open => "half_open",
        });
    }
};

/// Synchronization strategy.
pub const SyncStrategy = enum { atomic, mutex, none };

/// Circuit breaker configuration.
pub const Config = struct {
    /// Consecutive failures before opening.
    failure_threshold: u32 = 5,
    /// Successes in half-open to close.
    success_threshold: u32 = 2,
    /// Milliseconds before open -> half-open.
    timeout_ms: u64 = 60_000,
    /// Max concurrent requests in half-open.
    half_open_max_requests: u32 = 3,
    /// Name for logging/metrics.
    name: []const u8 = "default",
    /// Injectable time provider for deterministic tests.
    time_provider: ?platform_time.TimeProvider = null,
    /// Callback on state change.
    on_state_change: ?*const fn ([]const u8, CircuitState, CircuitState) void = null,
};

/// Statistics about circuit breaker behavior.
pub const Stats = struct {
    total_requests: u64 = 0,
    successful_requests: u64 = 0,
    failed_requests: u64 = 0,
    rejected_requests: u64 = 0,
    times_opened: u64 = 0,
    current_state: CircuitState = .closed,

    pub fn successRate(self: Stats) f64 {
        if (self.total_requests == 0) return 1.0;
        return @as(f64, @floatFromInt(self.successful_requests)) / @as(f64, @floatFromInt(self.total_requests));
    }

    pub fn failureRate(self: Stats) f64 {
        if (self.total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.failed_requests)) / @as(f64, @floatFromInt(self.total_requests));
    }
};

/// Parameterized circuit breaker.
///
/// `strategy` selects the synchronization approach:
/// - `.atomic` — lock-free atomics, no allocator needed, no windowed counting
/// - `.mutex` — mutex-protected, supports windowed counting, requires allocator
/// - `.none` — no sync, caller ensures single-threaded access, no allocator
pub fn CircuitBreaker(comptime strategy: SyncStrategy) type {
    return struct {
        const Self = @This();

        config: Config,
        time_provider: platform_time.TimeProvider,

        // State fields — representation varies by strategy
        state: StateField,
        failure_count: CountField,
        success_count: CountField,
        half_open_requests: CountField,
        last_failure_time_ms: TimeField,

        // Stats
        stats_total: CountField64,
        stats_success: CountField64,
        stats_failed: CountField64,
        stats_rejected: CountField64,
        stats_opened: CountField64,

        // Mutex (only for .mutex strategy)
        mutex: MutexField,

        // Field types depend on strategy
        const StateField = switch (strategy) {
            .atomic => std.atomic.Value(u8),
            .mutex, .none => CircuitState,
        };
        const CountField = switch (strategy) {
            .atomic => std.atomic.Value(u32),
            .mutex, .none => u32,
        };
        const CountField64 = switch (strategy) {
            .atomic => std.atomic.Value(u64),
            .mutex, .none => u64,
        };
        const TimeField = switch (strategy) {
            .atomic => std.atomic.Value(i64),
            .mutex, .none => i64,
        };
        const MutexField = switch (strategy) {
            .mutex => sync.Mutex,
            .atomic, .none => void,
        };

        pub fn init(config: Config) Self {
            const tp = config.time_provider orelse platform_time.TimeProvider{};
            return .{
                .config = config,
                .time_provider = tp,
                .state = initState(),
                .failure_count = initCount32(),
                .success_count = initCount32(),
                .half_open_requests = initCount32(),
                .last_failure_time_ms = initTime(),
                .stats_total = initCount64(),
                .stats_success = initCount64(),
                .stats_failed = initCount64(),
                .stats_rejected = initCount64(),
                .stats_opened = initCount64(),
                .mutex = initMutex(),
            };
        }

        fn initState() StateField {
            return switch (strategy) {
                .atomic => std.atomic.Value(u8).init(@intFromEnum(CircuitState.closed)),
                .mutex, .none => .closed,
            };
        }
        fn initCount32() CountField {
            return switch (strategy) {
                .atomic => std.atomic.Value(u32).init(0),
                .mutex, .none => 0,
            };
        }
        fn initCount64() CountField64 {
            return switch (strategy) {
                .atomic => std.atomic.Value(u64).init(0),
                .mutex, .none => 0,
            };
        }
        fn initTime() TimeField {
            return switch (strategy) {
                .atomic => std.atomic.Value(i64).init(0),
                .mutex, .none => 0,
            };
        }
        fn initMutex() MutexField {
            return switch (strategy) {
                .mutex => .{},
                .atomic, .none => {},
            };
        }

        // ── State access ────────────────────────────────────────────

        fn loadState(self: *const Self) CircuitState {
            return switch (strategy) {
                .atomic => @enumFromInt(self.state.load(.monotonic)),
                .mutex, .none => self.state,
            };
        }

        fn storeState(self: *Self, s: CircuitState) void {
            switch (strategy) {
                .atomic => self.state.store(@intFromEnum(s), .monotonic),
                .mutex, .none => self.state = s,
            }
        }

        fn casState(self: *Self, expected: CircuitState, desired: CircuitState) bool {
            return switch (strategy) {
                .atomic => self.state.cmpxchgStrong(
                    @intFromEnum(expected),
                    @intFromEnum(desired),
                    .acq_rel,
                    .monotonic,
                ) == null,
                .mutex, .none => {
                    if (self.state == expected) {
                        self.state = desired;
                        return true;
                    }
                    return false;
                },
            };
        }

        // ── Counter access ──────────────────────────────────────────

        fn loadU32(field: *const CountField) u32 {
            return switch (strategy) {
                .atomic => field.load(.monotonic),
                .mutex, .none => field.*,
            };
        }

        fn storeU32(field: *CountField, val: u32) void {
            switch (strategy) {
                .atomic => field.store(val, .monotonic),
                .mutex, .none => field.* = val,
            }
        }

        fn fetchAddU32(field: *CountField, val: u32) u32 {
            return switch (strategy) {
                .atomic => field.fetchAdd(val, .monotonic),
                .mutex, .none => blk: {
                    const old = field.*;
                    field.* += val;
                    break :blk old;
                },
            };
        }

        fn fetchAddU64(field: *CountField64, val: u64) void {
            switch (strategy) {
                .atomic => _ = field.fetchAdd(val, .monotonic),
                .mutex, .none => field.* += val,
            }
        }

        fn loadU64(field: *const CountField64) u64 {
            return switch (strategy) {
                .atomic => field.load(.monotonic),
                .mutex, .none => field.*,
            };
        }

        fn storeTime(field: *TimeField, val: i64) void {
            switch (strategy) {
                .atomic => field.store(val, .monotonic),
                .mutex, .none => field.* = val,
            }
        }

        fn loadTime(field: *const TimeField) i64 {
            return switch (strategy) {
                .atomic => field.load(.monotonic),
                .mutex, .none => field.*,
            };
        }

        // ── Lock helpers ────────────────────────────────────────────

        fn lockIfNeeded(self: *Self) void {
            if (strategy == .mutex) self.mutex.lock();
        }

        fn unlockIfNeeded(self: *Self) void {
            if (strategy == .mutex) self.mutex.unlock();
        }

        // ── Public API ──────────────────────────────────────────────

        /// Get the current state.
        pub fn getState(self: *const Self) CircuitState {
            return self.loadState();
        }

        /// Check if a request can be attempted.
        pub fn canAttempt(self: *Self) bool {
            self.lockIfNeeded();
            defer self.unlockIfNeeded();
            return self.canAttemptInner();
        }

        fn canAttemptInner(self: *Self) bool {
            const current = self.loadState();
            switch (current) {
                .closed => {
                    fetchAddU64(&self.stats_total, 1);
                    return true;
                },
                .open => {
                    const now = self.time_provider.nowMs();
                    const last = loadTime(&self.last_failure_time_ms);
                    const timeout: i64 = @intCast(self.config.timeout_ms);

                    if (now - last >= timeout) {
                        if (self.casState(.open, .half_open)) {
                            storeU32(&self.success_count, 0);
                            storeU32(&self.half_open_requests, 1);
                            fetchAddU64(&self.stats_total, 1);
                            self.notifyStateChange(.open, .half_open);
                            return true;
                        }
                        // Another thread transitioned — retry
                        if (strategy == .atomic) return self.canAttemptInner();
                    }
                    fetchAddU64(&self.stats_rejected, 1);
                    return false;
                },
                .half_open => {
                    const cur = fetchAddU32(&self.half_open_requests, 1);
                    if (cur < self.config.half_open_max_requests) {
                        fetchAddU64(&self.stats_total, 1);
                        return true;
                    }
                    // Undo increment, reject
                    if (strategy == .atomic) {
                        _ = self.half_open_requests.fetchSub(1, .monotonic);
                    } else {
                        self.half_open_requests -= 1;
                    }
                    fetchAddU64(&self.stats_rejected, 1);
                    return false;
                },
            }
        }

        /// Record a successful operation.
        pub fn recordSuccess(self: *Self) void {
            self.lockIfNeeded();
            defer self.unlockIfNeeded();

            fetchAddU64(&self.stats_success, 1);
            const current = self.loadState();
            switch (current) {
                .closed => storeU32(&self.failure_count, 0),
                .half_open => {
                    const successes = fetchAddU32(&self.success_count, 1) + 1;
                    if (successes >= self.config.success_threshold) {
                        if (self.casState(.half_open, .closed)) {
                            storeU32(&self.failure_count, 0);
                            self.notifyStateChange(.half_open, .closed);
                        }
                    }
                },
                .open => {},
            }
        }

        /// Record a failed operation.
        pub fn recordFailure(self: *Self) void {
            self.lockIfNeeded();
            defer self.unlockIfNeeded();

            fetchAddU64(&self.stats_failed, 1);
            const now = self.time_provider.nowMs();
            storeTime(&self.last_failure_time_ms, now);

            const current = self.loadState();
            switch (current) {
                .closed => {
                    const failures = fetchAddU32(&self.failure_count, 1) + 1;
                    if (failures >= self.config.failure_threshold) {
                        if (self.casState(.closed, .open)) {
                            fetchAddU64(&self.stats_opened, 1);
                            self.notifyStateChange(.closed, .open);
                        }
                    }
                },
                .half_open => {
                    if (self.casState(.half_open, .open)) {
                        fetchAddU64(&self.stats_opened, 1);
                        storeU32(&self.success_count, 0);
                        self.notifyStateChange(.half_open, .open);
                    }
                },
                .open => {},
            }
        }

        /// Reset to closed state.
        pub fn reset(self: *Self) void {
            self.lockIfNeeded();
            defer self.unlockIfNeeded();

            storeState(self, .closed);
            storeU32(&self.failure_count, 0);
            storeU32(&self.success_count, 0);
            storeU32(&self.half_open_requests, 0);
        }

        /// Force transition to a specific state (for testing).
        pub fn forceState(self: *Self, target: CircuitState) void {
            self.lockIfNeeded();
            defer self.unlockIfNeeded();

            storeState(self, target);
            storeU32(&self.success_count, 0);
            storeU32(&self.half_open_requests, 0);
            if (target == .closed) storeU32(&self.failure_count, 0);
        }

        /// Get statistics snapshot.
        pub fn getStats(self: *const Self) Stats {
            return .{
                .total_requests = loadU64(&self.stats_total),
                .successful_requests = loadU64(&self.stats_success),
                .failed_requests = loadU64(&self.stats_failed),
                .rejected_requests = loadU64(&self.stats_rejected),
                .times_opened = loadU64(&self.stats_opened),
                .current_state = self.loadState(),
            };
        }

        /// Check if circuit is currently allowing requests (no side effects).
        pub fn isAllowing(self: *const Self) bool {
            return self.loadState() != .open;
        }

        fn notifyStateChange(self: *Self, old: CircuitState, new: CircuitState) void {
            if (self.config.on_state_change) |cb| cb(self.config.name, old, new);
        }
    };
}

// ============================================================================
// Convenience aliases
// ============================================================================

/// Lock-free circuit breaker (for streaming / high-throughput).
pub const AtomicCircuitBreaker = CircuitBreaker(.atomic);
/// Mutex-protected circuit breaker (for network / windowed counting).
pub const MutexCircuitBreaker = CircuitBreaker(.mutex);
/// Single-threaded circuit breaker (for gateway / simple use).
pub const SimpleCircuitBreaker = CircuitBreaker(.none);

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

test "CircuitBreaker(.atomic) initial state is closed" {
    var cb = AtomicCircuitBreaker.init(.{});
    try std.testing.expectEqual(CircuitState.closed, cb.getState());
    try std.testing.expect(cb.canAttempt());
    try std.testing.expect(cb.isAllowing());
}

test "CircuitBreaker(.none) transitions to open after failures" {
    var cb = SimpleCircuitBreaker.init(.{ .failure_threshold = 3 });

    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();
    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();
    try std.testing.expect(cb.canAttempt());
    cb.recordFailure();

    try std.testing.expectEqual(CircuitState.open, cb.getState());
    try std.testing.expect(!cb.canAttempt());
}

test "CircuitBreaker(.atomic) success resets failure count" {
    var cb = AtomicCircuitBreaker.init(.{ .failure_threshold = 3 });

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

test "CircuitBreaker(.atomic) half-open allows limited requests" {
    var clock = TestClock{};
    var cb = AtomicCircuitBreaker.init(.{
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

test "CircuitBreaker(.mutex) half-open to closed on successes" {
    var clock = TestClock{};
    var cb = MutexCircuitBreaker.init(.{
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

test "CircuitBreaker(.none) half-open to open on failure" {
    var clock = TestClock{};
    var cb = SimpleCircuitBreaker.init(.{
        .failure_threshold = 1,
        .timeout_ms = 1,
        .time_provider = testTimeProvider(&clock),
    });

    _ = cb.canAttempt();
    cb.recordFailure();

    clock.now_ms += 5;

    try std.testing.expect(cb.canAttempt());
    try std.testing.expectEqual(CircuitState.half_open, cb.getState());

    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());
}

test "CircuitBreaker(.atomic) stats tracking" {
    var cb = AtomicCircuitBreaker.init(.{ .failure_threshold = 2 });

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

test "CircuitBreaker(.mutex) reset" {
    var cb = MutexCircuitBreaker.init(.{ .failure_threshold = 1 });

    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expectEqual(CircuitState.open, cb.getState());

    cb.reset();
    try std.testing.expectEqual(CircuitState.closed, cb.getState());
    try std.testing.expect(cb.canAttempt());
}

test "CircuitBreaker stats successRate" {
    var cb = SimpleCircuitBreaker.init(.{ .failure_threshold = 10 });

    _ = cb.canAttempt();
    cb.recordSuccess();
    _ = cb.canAttempt();
    cb.recordSuccess();
    _ = cb.canAttempt();
    cb.recordFailure();

    const stats = cb.getStats();
    try std.testing.expect(stats.successRate() > 0.66);
    try std.testing.expect(stats.successRate() < 0.67);
}

test "state change callback fires" {
    const T = struct {
        var called: bool = false;
        fn callback(_: []const u8, old: CircuitState, new: CircuitState) void {
            if (old == .closed and new == .open) called = true;
        }
    };
    T.called = false;

    var cb = SimpleCircuitBreaker.init(.{
        .failure_threshold = 1,
        .on_state_change = T.callback,
    });

    _ = cb.canAttempt();
    cb.recordFailure();
    try std.testing.expect(T.called);
}

test {
    std.testing.refAllDecls(@This());
}
