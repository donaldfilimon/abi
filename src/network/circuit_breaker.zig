//! Circuit breaker pattern for network resilience.
//!
//! Implements the circuit breaker pattern to prevent cascading failures
//! in distributed systems. When failures reach a threshold, the circuit
//! "opens" to fail fast and allow the system to recover.
//!
//! States:
//! - Closed: Normal operation, requests pass through
//! - Open: Failures exceeded threshold, requests fail immediately
//! - HalfOpen: Testing if service recovered, limited requests allowed
//!
//! Features:
//! - Thread-safe operations with mutex protection
//! - Windowed failure counting (sliding window)
//! - State change callbacks for monitoring
//! - Comprehensive statistics tracking
//! - Registry for managing multiple circuit breakers

const std = @import("std");
const time = @import("../shared/utils.zig");

/// Error set for network operations that can be wrapped by the circuit breaker
pub const NetworkOperationError = error{
    /// Network connection failed
    ConnectionFailed,
    /// Request timed out
    Timeout,
    /// Server returned an error response
    ServerError,
    /// DNS resolution failed
    DnsResolutionFailed,
    /// Connection was reset by peer
    ConnectionReset,
    /// Host is unreachable
    HostUnreachable,
    /// Network is unreachable
    NetworkUnreachable,
    /// Connection refused
    ConnectionRefused,
    /// SSL/TLS handshake failed
    TlsHandshakeFailed,
    /// Circuit breaker is open
    CircuitOpen,
    /// Too many requests in half-open state
    TooManyRequests,
} || std.mem.Allocator.Error;

pub const CircuitState = enum {
    closed,
    open,
    half_open,

    pub fn format(self: CircuitState, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        const name = switch (self) {
            .closed => "closed",
            .open => "open",
            .half_open => "half_open",
        };
        try writer.writeAll(name);
    }
};

pub const CircuitConfig = struct {
    /// Number of failures before opening circuit.
    failure_threshold: u32 = 5,
    /// Number of successes needed to close circuit from half-open.
    success_threshold: u32 = 2,
    /// Time to wait before transitioning from open to half-open (ms).
    timeout_ms: u64 = 60000,
    /// Maximum requests allowed in half-open state.
    half_open_max_calls: u32 = 3,
    /// Time window for counting failures (ms). 0 = no window (count all).
    failure_window_ms: u64 = 0,
    /// Enable automatic half-open transition after timeout.
    auto_half_open: bool = true,
    /// Callback for state change events.
    on_state_change: ?*const fn ([]const u8, CircuitState, CircuitState) void = null,
};

pub const CircuitResult = union(enum) {
    success: []const u8,
    err: struct {
        code: CircuitError,
        message: []const u8,
    },
    rejected: struct {
        message: []const u8,
        fallback: ?[]const u8,
    },
};

pub const CircuitError = enum {
    circuit_open,
    timeout,
    rejected,
};

/// Failure record for windowed counting.
const FailureRecord = struct {
    timestamp_ms: i64,
    error_code: u32,
};

/// Comprehensive circuit breaker statistics.
pub const CircuitStats = struct {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    rejected_requests: u64,
    state_transitions: u32,
    consecutive_successes: u32,
    consecutive_failures: u32,
    last_failure_time_ms: i64,
    last_success_time_ms: i64,
    time_in_current_state_ms: u64,

    pub fn successRate(self: CircuitStats) f64 {
        if (self.total_requests == 0) return 1.0;
        return @as(f64, @floatFromInt(self.successful_requests)) /
            @as(f64, @floatFromInt(self.total_requests));
    }

    pub fn failureRate(self: CircuitStats) f64 {
        if (self.total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.failed_requests)) /
            @as(f64, @floatFromInt(self.total_requests));
    }

    pub fn rejectionRate(self: CircuitStats) f64 {
        const total = self.total_requests + self.rejected_requests;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.rejected_requests)) /
            @as(f64, @floatFromInt(total));
    }
};

pub const CircuitBreaker = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    config: CircuitConfig,
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    half_open_calls: u32,
    last_failure_time_ms: i64,
    open_until_ms: i64,
    state_changed_at_ms: i64,
    failure_records: std.ArrayListUnmanaged(FailureRecord),
    stats: CircuitStats,
    mutex: std.Thread.Mutex,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: CircuitConfig) CircuitBreaker {
        return initWithName(allocator, "default", config);
    }

    pub fn initWithName(allocator: std.mem.Allocator, name: []const u8, config: CircuitConfig) CircuitBreaker {
        return .{
            .allocator = allocator,
            .name = name,
            .config = config,
            .state = .closed,
            .failure_count = 0,
            .success_count = 0,
            .half_open_calls = 0,
            .last_failure_time_ms = 0,
            .open_until_ms = 0,
            .state_changed_at_ms = time.nowMilliseconds(),
            .failure_records = .{},
            .stats = .{
                .total_requests = 0,
                .successful_requests = 0,
                .failed_requests = 0,
                .rejected_requests = 0,
                .state_transitions = 0,
                .consecutive_successes = 0,
                .consecutive_failures = 0,
                .last_failure_time_ms = 0,
                .last_success_time_ms = 0,
                .time_in_current_state_ms = 0,
            },
            .mutex = .{},
        };
    }

    pub fn deinit(self: *CircuitBreaker) void {
        self.failure_records.deinit(self.allocator);
        self.* = undefined;
    }

    /// Thread-safe state getter with auto-transition check.
    pub fn getState(self: *Self) CircuitState {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.checkAutoTransition();
        return self.state;
    }

    /// Thread-safe reset.
    pub fn reset(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.state = .closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.half_open_calls = 0;
        self.state_changed_at_ms = time.nowMilliseconds();
        self.failure_records.clearRetainingCapacity();
        self.stats = .{
            .total_requests = 0,
            .successful_requests = 0,
            .failed_requests = 0,
            .rejected_requests = 0,
            .state_transitions = 0,
            .consecutive_successes = 0,
            .consecutive_failures = 0,
            .last_failure_time_ms = 0,
            .last_success_time_ms = 0,
            .time_in_current_state_ms = 0,
        };
    }

    /// Get comprehensive statistics.
    pub fn getStats(self: *Self) CircuitStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        var stats = self.stats;
        const now_ms = time.nowMilliseconds();
        stats.time_in_current_state_ms = @intCast(@max(0, now_ms - self.state_changed_at_ms));
        return stats;
    }

    /// Force circuit to open state.
    pub fn forceOpen(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.transitionTo(.open);
        self.open_until_ms = time.nowMilliseconds() + @as(i64, @intCast(self.config.timeout_ms));
    }

    /// Force circuit to closed state.
    pub fn forceClosed(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.transitionTo(.closed);
    }

    /// Function pointer type for operations wrapped by the circuit breaker.
    pub const OperationFn = *const fn () NetworkOperationError![]const u8;

    pub fn execute(self: *CircuitBreaker, operation: OperationFn) CircuitResult {
        if (!self.canExecute()) {
            const metrics = self.getMetrics();
            std.log.debug("Circuit breaker rejected request: state={t}, failures={d}, time_to_reset={d}ms", .{
                self.state,
                self.failure_count,
                metrics.time_to_next_state_ms,
            });
            return .{ .rejected = .{ .message = "Circuit open", .fallback = null } };
        }

        const payload = operation() catch |err| {
            self.onFailure();
            const code: CircuitError = switch (err) {
                error.Timeout => .timeout,
                else => .rejected,
            };
            std.log.debug("Circuit breaker operation failed: error={t}, failure_count={d}/{d}", .{
                err,
                self.failure_count,
                self.config.failure_threshold,
            });
            return .{ .err = .{ .code = code, .message = @errorName(err) } };
        };

        self.onSuccess();
        return .{ .success = payload };
    }

    pub fn onSuccess(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.onSuccessUnlocked();
    }

    fn onSuccessUnlocked(self: *Self) void {
        const now_ms = time.nowMilliseconds();

        self.stats.total_requests += 1;
        self.stats.successful_requests += 1;
        self.stats.last_success_time_ms = now_ms;
        self.stats.consecutive_successes += 1;
        self.stats.consecutive_failures = 0;

        switch (self.state) {
            .closed => {
                self.success_count += 1;
                if (self.success_count >= self.config.success_threshold) {
                    self.failure_count = 0;
                }
            },
            .half_open => {
                self.success_count += 1;
                self.half_open_calls += 1;
                if (self.success_count >= self.config.success_threshold) {
                    self.transitionTo(.closed);
                    self.failure_count = 0;
                    self.success_count = 0;
                    self.half_open_calls = 0;
                }
            },
            .open => {},
        }
    }

    pub fn onFailure(self: *Self) void {
        self.onFailureWithCode(0);
    }

    pub fn onFailureWithCode(self: *Self, error_code: u32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.onFailureUnlocked(error_code);
    }

    fn onFailureUnlocked(self: *Self, error_code: u32) void {
        const now_ms = time.nowMilliseconds();
        self.last_failure_time_ms = now_ms;
        self.failure_count += 1;

        self.stats.total_requests += 1;
        self.stats.failed_requests += 1;
        self.stats.last_failure_time_ms = now_ms;
        self.stats.consecutive_failures += 1;
        self.stats.consecutive_successes = 0;

        // Record failure for windowed counting
        self.failure_records.append(self.allocator, .{
            .timestamp_ms = now_ms,
            .error_code = error_code,
        }) catch {};

        // Clean old records outside the window
        self.cleanOldFailures(now_ms);

        switch (self.state) {
            .closed => {
                const recent_failures = self.countRecentFailures();
                if (recent_failures >= self.config.failure_threshold) {
                    self.transitionTo(.open);
                    self.open_until_ms = now_ms + @as(i64, @intCast(self.config.timeout_ms));
                }
            },
            .half_open => {
                // Any failure in half-open returns to open
                self.transitionTo(.open);
                self.open_until_ms = now_ms + @as(i64, @intCast(self.config.timeout_ms));
            },
            .open => {},
        }
    }

    fn transitionTo(self: *Self, new_state: CircuitState) void {
        if (self.state == new_state) return;

        const old_state = self.state;
        self.state = new_state;
        self.state_changed_at_ms = time.nowMilliseconds();
        self.stats.state_transitions += 1;

        // Reset state-specific counters
        if (new_state == .half_open) {
            self.half_open_calls = 0;
            self.stats.consecutive_successes = 0;
        } else if (new_state == .closed) {
            self.failure_records.clearRetainingCapacity();
            self.stats.consecutive_failures = 0;
        }

        // Notify callback
        if (self.config.on_state_change) |callback| {
            callback(self.name, old_state, new_state);
        }
    }

    fn checkAutoTransition(self: *Self) void {
        if (!self.config.auto_half_open) return;

        if (self.state == .open) {
            const now_ms = time.nowMilliseconds();
            if (now_ms >= self.open_until_ms) {
                self.transitionTo(.half_open);
            }
        }
    }

    fn countRecentFailures(self: *Self) u32 {
        if (self.config.failure_window_ms == 0) {
            return @intCast(self.failure_records.items.len);
        }

        const now_ms = time.nowMilliseconds();
        const window_start = now_ms - @as(i64, @intCast(self.config.failure_window_ms));

        var count: u32 = 0;
        for (self.failure_records.items) |record| {
            if (record.timestamp_ms >= window_start) {
                count += 1;
            }
        }
        return count;
    }

    fn cleanOldFailures(self: *Self, now_ms: i64) void {
        if (self.config.failure_window_ms == 0) return;

        const window_start = now_ms - @as(i64, @intCast(self.config.failure_window_ms));

        // Find first record within the window (records are in chronological order)
        var first_valid: usize = 0;
        for (self.failure_records.items, 0..) |record, i| {
            if (record.timestamp_ms >= window_start) {
                first_valid = i;
                break;
            }
            first_valid = i + 1;
        }

        // Remove all old records at once by shifting
        if (first_valid > 0 and first_valid <= self.failure_records.items.len) {
            const remaining = self.failure_records.items.len - first_valid;
            if (remaining > 0) {
                std.mem.copyForwards(
                    FailureRecord,
                    self.failure_records.items[0..remaining],
                    self.failure_records.items[first_valid..],
                );
            }
            self.failure_records.shrinkRetainingCapacity(remaining);
        }
    }

    /// Thread-safe check if request can execute.
    pub fn canExecute(self: *Self) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.checkAutoTransition();
        return self.canExecuteUnlocked();
    }

    fn canExecuteUnlocked(self: *Self) bool {
        switch (self.state) {
            .closed => return true,
            .open => return false,
            .half_open => return self.half_open_calls < self.config.half_open_max_calls,
        }
    }

    /// Thread-safe request allowance check.
    pub fn allowRequest(self: *Self) NetworkOperationError!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.checkAutoTransition();

        switch (self.state) {
            .closed => return,
            .open => {
                self.stats.rejected_requests += 1;
                return NetworkOperationError.CircuitOpen;
            },
            .half_open => {
                if (self.half_open_calls >= self.config.half_open_max_calls) {
                    self.stats.rejected_requests += 1;
                    return NetworkOperationError.TooManyRequests;
                }
                self.half_open_calls += 1;
                return;
            },
        }
    }

    pub fn recordSuccess(self: *Self) void {
        self.onSuccess();
    }

    pub fn recordFailure(self: *Self) void {
        self.onFailure();
    }

    pub fn recordFailureWithCode(self: *Self, error_code: u32) void {
        self.onFailureWithCode(error_code);
    }

    pub fn getMetrics(self: *Self) CircuitMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now_ms = time.nowMilliseconds();
        const time_to_next_state_ms = if (self.state == .open)
            @max(0, self.open_until_ms - now_ms)
        else
            0;

        return .{
            .state = self.state,
            .failure_count = self.failure_count,
            .success_count = self.success_count,
            .last_failure_time_ms = self.last_failure_time_ms,
            .open_until_ms = self.open_until_ms,
            .time_to_next_state_ms = time_to_next_state_ms,
        };
    }
};

pub const CircuitMetrics = struct {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time_ms: i64,
    open_until_ms: i64,
    time_to_next_state_ms: i64,
};

/// Aggregate statistics for a circuit breaker registry.
pub const AggregateStats = struct {
    total_breakers: u32,
    open_breakers: u32,
    half_open_breakers: u32,
    closed_breakers: u32,
    total_requests: u64,
    total_failures: u64,
    total_rejections: u64,
};

pub const CircuitRegistry = struct {
    allocator: std.mem.Allocator,
    breakers: std.StringArrayHashMapUnmanaged(CircuitBreaker),
    default_config: CircuitConfig,
    mutex: std.Thread.Mutex,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) CircuitRegistry {
        return initWithConfig(allocator, .{});
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, default_config: CircuitConfig) CircuitRegistry {
        return .{
            .allocator = allocator,
            .breakers = std.StringArrayHashMapUnmanaged(CircuitBreaker).empty,
            .default_config = default_config,
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Self) void {
        var it = self.breakers.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.breakers.deinit(self.allocator);
        self.* = undefined;
    }

    /// Register a new circuit breaker with custom config.
    pub fn register(self: *Self, name: []const u8, config: CircuitConfig) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        try self.breakers.put(self.allocator, name_copy, CircuitBreaker.initWithName(self.allocator, name_copy, config));
    }

    /// Get or create a circuit breaker with default config.
    pub fn getOrCreate(self: *Self, name: []const u8) !*CircuitBreaker {
        return self.getOrCreateWithConfig(name, self.default_config);
    }

    /// Get or create with custom config.
    pub fn getOrCreateWithConfig(self: *Self, name: []const u8, config: CircuitConfig) !*CircuitBreaker {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.breakers.getPtr(name)) |breaker| {
            return breaker;
        }

        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);

        try self.breakers.put(self.allocator, name_copy, CircuitBreaker.initWithName(self.allocator, name_copy, config));
        return self.breakers.getPtr(name).?;
    }

    pub fn getBreaker(self: *Self, name: []const u8) ?*CircuitBreaker {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.breakers.getPtr(name);
    }

    pub fn removeBreaker(self: *Self, name: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.breakers.fetchSwapRemove(name)) |entry| {
            entry.value.deinit();
            self.allocator.free(entry.key);
            return true;
        }
        return false;
    }

    pub fn getAllMetrics(self: *Self) ![]CircuitMetricEntry {
        self.mutex.lock();
        defer self.mutex.unlock();

        const entries = try self.allocator.alloc(CircuitMetricEntry, self.breakers.count());
        var i: usize = 0;

        var it = self.breakers.iterator();
        while (it.next()) |entry| {
            entries[i] = CircuitMetricEntry{
                .name = entry.key_ptr.*,
                .metrics = entry.value_ptr.getMetrics(),
            };
            i += 1;
        }

        return entries;
    }

    /// Get aggregate statistics across all circuit breakers.
    pub fn getAggregateStats(self: *Self) AggregateStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        var result = AggregateStats{
            .total_breakers = 0,
            .open_breakers = 0,
            .half_open_breakers = 0,
            .closed_breakers = 0,
            .total_requests = 0,
            .total_failures = 0,
            .total_rejections = 0,
        };

        var it = self.breakers.iterator();
        while (it.next()) |entry| {
            result.total_breakers += 1;
            const stats = entry.value_ptr.getStats();
            result.total_requests += stats.total_requests;
            result.total_failures += stats.failed_requests;
            result.total_rejections += stats.rejected_requests;

            switch (entry.value_ptr.getState()) {
                .open => result.open_breakers += 1,
                .half_open => result.half_open_breakers += 1,
                .closed => result.closed_breakers += 1,
            }
        }

        return result;
    }

    /// Reset all circuit breakers to closed state.
    pub fn resetAll(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.breakers.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.reset();
        }
    }

    /// List all breaker names.
    pub fn listBreakers(self: *Self) ![][]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        var names = std.ArrayListUnmanaged([]const u8){};
        var it = self.breakers.keyIterator();
        while (it.next()) |key| {
            try names.append(self.allocator, key.*);
        }
        return names.toOwnedSlice(self.allocator);
    }
};

pub const CircuitMetricEntry = struct {
    name: []const u8,
    metrics: CircuitMetrics,
};

test "circuit breaker init" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{});
    defer breaker.deinit();

    try std.testing.expectEqual(CircuitState.closed, breaker.getState());
    try std.testing.expectEqual(@as(u32, 0), breaker.failure_count);
}

test "circuit breaker failure threshold" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{
        .failure_threshold = 3,
        .success_threshold = 2,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    breaker.recordFailure();
    breaker.recordFailure();
    breaker.recordFailure();

    try std.testing.expectEqual(CircuitState.open, breaker.getState());
}

test "circuit breaker reset" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{
        .failure_threshold = 3,
        .success_threshold = 2,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    breaker.recordFailure();
    breaker.recordFailure();
    breaker.recordFailure();
    try std.testing.expectEqual(CircuitState.open, breaker.getState());

    breaker.reset();
    try std.testing.expectEqual(CircuitState.closed, breaker.getState());
    try std.testing.expectEqual(@as(u32, 0), breaker.failure_count);
}

test "circuit breaker can execute" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{
        .failure_threshold = 3,
        .success_threshold = 2,
        .timeout_ms = 60000,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    try std.testing.expect(breaker.canExecute());

    breaker.recordFailure();
    breaker.recordFailure();
    breaker.recordFailure();

    try std.testing.expect(!breaker.canExecute());
}

test "circuit breaker stats tracking" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{
        .failure_threshold = 10,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    // Record some activity
    breaker.recordSuccess();
    breaker.recordSuccess();
    breaker.recordFailure();

    const stats = breaker.getStats();
    try std.testing.expectEqual(@as(u64, 3), stats.total_requests);
    try std.testing.expectEqual(@as(u64, 2), stats.successful_requests);
    try std.testing.expectEqual(@as(u64, 1), stats.failed_requests);

    // Success rate should be ~66.7%
    try std.testing.expect(stats.successRate() > 0.66);
    try std.testing.expect(stats.successRate() < 0.67);
}

test "circuit breaker allowRequest rejection" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{
        .failure_threshold = 2,
        .auto_half_open = false,
    });
    defer breaker.deinit();

    // Should work when closed
    try breaker.allowRequest();
    breaker.recordSuccess();

    // Open the circuit
    breaker.recordFailure();
    breaker.recordFailure();

    // Should fail when open
    try std.testing.expectError(NetworkOperationError.CircuitOpen, breaker.allowRequest());

    const stats = breaker.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.rejected_requests);
}

test "circuit breaker registry" {
    const allocator = std.testing.allocator;
    var registry = CircuitRegistry.init(allocator);
    defer registry.deinit();

    const config = CircuitConfig{
        .failure_threshold = 3,
        .success_threshold = 2,
    };

    try registry.register("service1", config);
    try registry.register("service2", config);

    const breaker1 = registry.getBreaker("service1");
    try std.testing.expect(breaker1 != null);

    try std.testing.expectEqual(@as(usize, 2), registry.breakers.count());
}

test "circuit breaker registry getOrCreate" {
    const allocator = std.testing.allocator;
    var registry = CircuitRegistry.init(allocator);
    defer registry.deinit();

    const breaker1 = try registry.getOrCreate("service-a");
    const breaker2 = try registry.getOrCreate("service-b");

    try std.testing.expect(breaker1 != breaker2);

    // Same name should return same breaker
    const breaker1_again = try registry.getOrCreate("service-a");
    try std.testing.expect(breaker1 == breaker1_again);

    const stats = registry.getAggregateStats();
    try std.testing.expectEqual(@as(u32, 2), stats.total_breakers);
    try std.testing.expectEqual(@as(u32, 2), stats.closed_breakers);
}
