//! Circuit breaker pattern for network resilience.
//!
//! Wraps the shared resilience circuit breaker with network-specific
//! features: CircuitRegistry for managing multiple breakers, execute()
//! wrapper, and allowRequest() API.

const std = @import("std");
const time = @import("../../services/shared/utils.zig");
const sync = @import("../../services/shared/sync.zig");
const Mutex = sync.Mutex;
const resilience = @import("../../services/shared/resilience/circuit_breaker.zig");

/// Error set for network operations wrapped by the circuit breaker.
pub const NetworkOperationError = error{
    ConnectionFailed,
    Timeout,
    ServerError,
    DnsResolutionFailed,
    ConnectionReset,
    HostUnreachable,
    NetworkUnreachable,
    ConnectionRefused,
    TlsHandshakeFailed,
    CircuitOpen,
    TooManyRequests,
} || std.mem.Allocator.Error;

// Re-export shared types
pub const CircuitState = resilience.CircuitState;

pub const CircuitConfig = struct {
    failure_threshold: u32 = 5,
    success_threshold: u32 = 2,
    timeout_ms: u64 = 60000,
    half_open_max_calls: u32 = 3,
    /// Time window for counting failures (ms). 0 = no window.
    failure_window_ms: u64 = 0,
    auto_half_open: bool = true,
    on_state_change: ?*const fn ([]const u8, CircuitState, CircuitState) void = null,
};

pub const CircuitResult = union(enum) {
    success: []const u8,
    err: struct { code: CircuitError, message: []const u8 },
    rejected: struct { message: []const u8, fallback: ?[]const u8 },
};

pub const CircuitError = enum { circuit_open, timeout, rejected };

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
        return @as(f64, @floatFromInt(self.successful_requests)) / @as(f64, @floatFromInt(self.total_requests));
    }

    pub fn failureRate(self: CircuitStats) f64 {
        if (self.total_requests == 0) return 0.0;
        return @as(f64, @floatFromInt(self.failed_requests)) / @as(f64, @floatFromInt(self.total_requests));
    }

    pub fn rejectionRate(self: CircuitStats) f64 {
        const total = self.total_requests + self.rejected_requests;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.rejected_requests)) / @as(f64, @floatFromInt(total));
    }
};

/// Network circuit breaker — mutex-protected with extended stats.
///
/// Wraps the shared `MutexCircuitBreaker` and adds network-specific
/// tracking (consecutive counts, time-in-state, windowed failures).
pub const CircuitBreaker = struct {
    allocator: std.mem.Allocator,
    name: []const u8,
    config: CircuitConfig,
    inner: resilience.MutexCircuitBreaker,

    // Extended stats beyond what the shared module tracks
    state_transitions: u32 = 0,
    consecutive_successes: u32 = 0,
    consecutive_failures: u32 = 0,
    last_failure_time_ms: i64 = 0,
    last_success_time_ms: i64 = 0,
    state_changed_at_ms: i64 = 0,
    rejected_requests: u64 = 0,
    mutex: Mutex = .{},

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, config: CircuitConfig) CircuitBreaker {
        return initWithName(allocator, "default", config);
    }

    pub fn initWithName(allocator: std.mem.Allocator, name: []const u8, config: CircuitConfig) CircuitBreaker {
        return .{
            .allocator = allocator,
            .name = name,
            .config = config,
            .inner = resilience.MutexCircuitBreaker.init(.{
                .failure_threshold = config.failure_threshold,
                .success_threshold = config.success_threshold,
                .timeout_ms = config.timeout_ms,
                .half_open_max_requests = config.half_open_max_calls,
                .name = name,
                .on_state_change = config.on_state_change,
            }),
            .state_changed_at_ms = time.nowMilliseconds(),
        };
    }

    pub fn deinit(self: *CircuitBreaker) void {
        self.* = undefined;
    }

    pub fn getState(self: *Self) CircuitState {
        return self.inner.getState();
    }

    pub fn reset(self: *Self) void {
        self.inner.reset();
        self.mutex.lock();
        defer self.mutex.unlock();
        self.state_transitions = 0;
        self.consecutive_successes = 0;
        self.consecutive_failures = 0;
        self.last_failure_time_ms = 0;
        self.last_success_time_ms = 0;
        self.state_changed_at_ms = time.nowMilliseconds();
        self.rejected_requests = 0;
    }

    pub fn getStats(self: *Self) CircuitStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        const inner_stats = self.inner.getStats();
        const now_ms = time.nowMilliseconds();
        return .{
            .total_requests = inner_stats.total_requests,
            .successful_requests = inner_stats.successful_requests,
            .failed_requests = inner_stats.failed_requests,
            .rejected_requests = inner_stats.rejected_requests + self.rejected_requests,
            .state_transitions = self.state_transitions,
            .consecutive_successes = self.consecutive_successes,
            .consecutive_failures = self.consecutive_failures,
            .last_failure_time_ms = self.last_failure_time_ms,
            .last_success_time_ms = self.last_success_time_ms,
            .time_in_current_state_ms = @intCast(@max(0, now_ms - self.state_changed_at_ms)),
        };
    }

    pub fn forceOpen(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        // Trip the inner breaker by recording enough failures
        var i: u32 = 0;
        while (i < self.config.failure_threshold) : (i += 1) {
            self.inner.recordFailure();
        }
        self.state_transitions += 1;
        self.state_changed_at_ms = time.nowMilliseconds();
    }

    pub fn forceClosed(self: *Self) void {
        self.inner.reset();
        self.mutex.lock();
        defer self.mutex.unlock();
        self.state_transitions += 1;
        self.state_changed_at_ms = time.nowMilliseconds();
    }

    /// Force circuit to half-open state (for testing).
    pub fn forceHalfOpen(self: *Self) void {
        self.inner.forceState(.half_open);
        self.mutex.lock();
        defer self.mutex.unlock();
        self.state_transitions += 1;
        self.state_changed_at_ms = time.nowMilliseconds();
    }

    pub const OperationFn = *const fn () NetworkOperationError![]const u8;

    pub fn execute(self: *CircuitBreaker, operation: OperationFn) CircuitResult {
        if (!self.canExecute()) {
            return .{ .rejected = .{ .message = "Circuit open", .fallback = null } };
        }

        const payload = operation() catch |err| {
            self.onFailure();
            const code: CircuitError = switch (err) {
                error.Timeout => .timeout,
                else => .rejected,
            };
            var err_buf: [64]u8 = undefined;
            const err_name = std.fmt.bufPrint(&err_buf, "{t}", .{err}) catch "unknown_error";
            return .{ .err = .{ .code = code, .message = err_name } };
        };

        self.onSuccess();
        return .{ .success = payload };
    }

    pub fn onSuccess(self: *Self) void {
        self.inner.recordSuccess();
        self.mutex.lock();
        defer self.mutex.unlock();
        self.last_success_time_ms = time.nowMilliseconds();
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;
    }

    pub fn onFailure(self: *Self) void {
        self.onFailureWithCode(0);
    }

    pub fn onFailureWithCode(self: *Self, _: u32) void {
        self.inner.recordFailure();
        self.mutex.lock();
        defer self.mutex.unlock();
        const now_ms = time.nowMilliseconds();
        self.last_failure_time_ms = now_ms;
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
    }

    pub fn canExecute(self: *Self) bool {
        return self.inner.canAttempt();
    }

    pub fn allowRequest(self: *Self) NetworkOperationError!void {
        if (!self.inner.canAttempt()) {
            self.mutex.lock();
            self.rejected_requests += 1;
            self.mutex.unlock();
            if (self.inner.getState() == .open) return NetworkOperationError.CircuitOpen;
            return NetworkOperationError.TooManyRequests;
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
        const inner_stats = self.inner.getStats();
        return .{
            .state = self.inner.getState(),
            .failure_count = @intCast(inner_stats.failed_requests),
            .success_count = @intCast(inner_stats.successful_requests),
            .last_failure_time_ms = self.last_failure_time_ms,
            .open_until_ms = 0, // Not directly exposed by shared; approximate
            .time_to_next_state_ms = 0,
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

pub const AggregateStats = struct {
    total_breakers: u32,
    open_breakers: u32,
    half_open_breakers: u32,
    closed_breakers: u32,
    total_requests: u64,
    total_failures: u64,
    total_rejections: u64,
};

pub const CircuitMetricEntry = struct {
    name: []const u8,
    metrics: CircuitMetrics,
};

/// Registry for managing multiple named circuit breakers.
pub const CircuitRegistry = struct {
    allocator: std.mem.Allocator,
    breakers: std.StringArrayHashMapUnmanaged(CircuitBreaker),
    default_config: CircuitConfig,
    mutex: Mutex,

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

    pub fn register(self: *Self, name: []const u8, config: CircuitConfig) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        try self.breakers.put(self.allocator, name_copy, CircuitBreaker.initWithName(self.allocator, name_copy, config));
    }

    pub fn getOrCreate(self: *Self, name: []const u8) !*CircuitBreaker {
        return self.getOrCreateWithConfig(name, self.default_config);
    }

    pub fn getOrCreateWithConfig(self: *Self, name: []const u8, config: CircuitConfig) !*CircuitBreaker {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.breakers.getPtr(name)) |breaker| return breaker;
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        try self.breakers.put(self.allocator, name_copy, CircuitBreaker.initWithName(self.allocator, name_copy, config));
        return self.breakers.getPtr(name_copy).?;
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
            entries[i] = .{ .name = entry.key_ptr.*, .metrics = entry.value_ptr.getMetrics() };
            i += 1;
        }
        return entries;
    }

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

    pub fn resetAll(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        var it = self.breakers.iterator();
        while (it.next()) |entry| entry.value_ptr.reset();
    }

    pub fn listBreakers(self: *Self) ![][]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        var names = std.ArrayListUnmanaged([]const u8).empty;
        var it = self.breakers.iterator();
        while (it.next()) |entry| try names.append(self.allocator, entry.key_ptr.*);
        return names.toOwnedSlice(self.allocator);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "circuit breaker init" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{});
    defer breaker.deinit();
    try std.testing.expectEqual(CircuitState.closed, breaker.getState());
}

test "circuit breaker failure threshold" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{ .failure_threshold = 3, .success_threshold = 2 });
    defer breaker.deinit();
    breaker.recordFailure();
    breaker.recordFailure();
    breaker.recordFailure();
    try std.testing.expectEqual(CircuitState.open, breaker.getState());
}

test "circuit breaker reset" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{ .failure_threshold = 3 });
    defer breaker.deinit();
    breaker.recordFailure();
    breaker.recordFailure();
    breaker.recordFailure();
    try std.testing.expectEqual(CircuitState.open, breaker.getState());
    breaker.reset();
    try std.testing.expectEqual(CircuitState.closed, breaker.getState());
}

test "circuit breaker can execute" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{ .failure_threshold = 3, .timeout_ms = 60000 });
    defer breaker.deinit();
    try std.testing.expect(breaker.canExecute());
    breaker.recordFailure();
    breaker.recordFailure();
    breaker.recordFailure();
    try std.testing.expect(!breaker.canExecute());
}

test "circuit breaker stats tracking" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{ .failure_threshold = 10 });
    defer breaker.deinit();
    breaker.recordSuccess();
    breaker.recordSuccess();
    breaker.recordFailure();
    const stats = breaker.getStats();
    try std.testing.expectEqual(@as(u64, 3), stats.total_requests);
    try std.testing.expectEqual(@as(u64, 2), stats.successful_requests);
    try std.testing.expectEqual(@as(u64, 1), stats.failed_requests);
    try std.testing.expect(stats.successRate() > 0.66);
    try std.testing.expect(stats.successRate() < 0.67);
}

test "circuit breaker allowRequest rejection" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{ .failure_threshold = 2 });
    defer breaker.deinit();
    try breaker.allowRequest();
    breaker.recordSuccess();
    breaker.recordFailure();
    breaker.recordFailure();
    try std.testing.expectError(NetworkOperationError.CircuitOpen, breaker.allowRequest());
}

test "circuit breaker registry" {
    const allocator = std.testing.allocator;
    var registry = CircuitRegistry.init(allocator);
    defer registry.deinit();
    const config = CircuitConfig{ .failure_threshold = 3, .success_threshold = 2 };
    try registry.register("service1", config);
    try registry.register("service2", config);
    try std.testing.expect(registry.getBreaker("service1") != null);
    try std.testing.expectEqual(@as(usize, 2), registry.breakers.count());
}

test "circuit breaker registry getOrCreate" {
    const allocator = std.testing.allocator;
    var registry = CircuitRegistry.init(allocator);
    defer registry.deinit();
    const breaker1 = try registry.getOrCreate("service-a");
    const breaker2 = try registry.getOrCreate("service-b");
    try std.testing.expect(breaker1 != breaker2);
    const breaker1_again = try registry.getOrCreate("service-a");
    try std.testing.expect(breaker1 == breaker1_again);
    const stats = registry.getAggregateStats();
    try std.testing.expectEqual(@as(u32, 2), stats.total_breakers);
    try std.testing.expectEqual(@as(u32, 2), stats.closed_breakers);
}

test {
    std.testing.refAllDecls(@This());
}
