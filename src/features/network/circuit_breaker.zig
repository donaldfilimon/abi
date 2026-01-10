//! Circuit breaker pattern for network resilience.
const std = @import("std");

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
} || std.mem.Allocator.Error;

pub const CircuitState = enum {
    closed,
    open,
    half_open,
};

pub const CircuitConfig = struct {
    failure_threshold: u32 = 5,
    success_threshold: u32 = 2,
    timeout_ms: u64 = 60000,
    half_open_max_calls: u32 = 3,
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

pub const CircuitBreaker = struct {
    allocator: std.mem.Allocator,
    config: CircuitConfig,
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    half_open_calls: u32,
    last_failure_time_ms: i64,
    open_until_ms: i64,

    pub fn init(allocator: std.mem.Allocator, config: CircuitConfig) CircuitBreaker {
        return .{
            .allocator = allocator,
            .config = config,
            .state = .closed,
            .failure_count = 0,
            .success_count = 0,
            .half_open_calls = 0,
            .last_failure_time_ms = 0,
            .open_until_ms = 0,
        };
    }

    pub fn deinit(self: *CircuitBreaker) void {
        self.* = undefined;
    }

    pub fn getState(self: *const CircuitBreaker) CircuitState {
        return self.state;
    }

    pub fn reset(self: *CircuitBreaker) void {
        self.state = .closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.half_open_calls = 0;
    }

    pub fn execute(self: *CircuitBreaker, operation: fn () anyerror![]const u8) CircuitResult {
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
                error.Timeout, error.TimedOut => .timeout,
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

    pub fn onSuccess(self: *CircuitBreaker) void {
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
                    self.state = .closed;
                    self.failure_count = 0;
                    self.success_count = 0;
                    self.half_open_calls = 0;
                }
            },
            .open => {},
        }
    }

    pub fn onFailure(self: *CircuitBreaker) void {
        const now_ms = std.time.milliTimestamp();
        self.last_failure_time_ms = now_ms;
        self.failure_count += 1;

        switch (self.state) {
            .closed => {
                if (self.failure_count >= self.config.failure_threshold) {
                    self.state = .open;
                    self.open_until_ms = now_ms + @as(i64, @intCast(self.config.timeout_ms));
                }
            },
            .half_open => {
                self.state = .open;
                self.open_until_ms = now_ms + @as(i64, @intCast(self.config.timeout_ms));
            },
            .open => {},
        }
    }

    pub fn canExecute(self: *const CircuitBreaker) bool {
        const now_ms = std.time.milliTimestamp();

        switch (self.state) {
            .closed => return true,
            .open => {
                // Check if timeout has elapsed
                if (now_ms >= self.open_until_ms) {
                    return true;
                }
                return false;
            },
            .half_open => return self.half_open_calls < self.config.half_open_max_calls,
        }
    }

    pub fn allowRequest(self: *CircuitBreaker) !void {
        if (!self.canExecute()) {
            return error.CircuitOpen;
        }

        if (self.state == .half_open) {
            self.half_open_calls += 1;
        }
    }

    pub fn recordSuccess(self: *CircuitBreaker) void {
        self.onSuccess();
    }

    pub fn recordFailure(self: *CircuitBreaker) void {
        self.onFailure();
    }

    pub fn getMetrics(self: *const CircuitBreaker) CircuitMetrics {
        const now_ms = std.time.milliTimestamp();
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

pub const CircuitRegistry = struct {
    allocator: std.mem.Allocator,
    breakers: std.StringArrayHashMapUnmanaged(CircuitBreaker),

    pub fn init(allocator: std.mem.Allocator) CircuitRegistry {
        return .{
            .allocator = allocator,
            .breakers = std.StringArrayHashMapUnmanaged(CircuitBreaker).empty,
        };
    }

    pub fn deinit(self: *CircuitRegistry) void {
        var it = self.breakers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.breakers.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn register(self: *CircuitRegistry, name: []const u8, config: CircuitConfig) !void {
        const name_copy = try self.allocator.dupe(u8, name);
        try self.breakers.put(self.allocator, name_copy, CircuitBreaker.init(self.allocator, config));
    }

    pub fn getBreaker(self: *CircuitRegistry, name: []const u8) ?*CircuitBreaker {
        return self.breakers.getPtr(name);
    }

    pub fn removeBreaker(self: *CircuitRegistry, name: []const u8) bool {
        if (self.breakers.remove(name)) |entry| {
            self.allocator.free(entry.key_ptr.*);
            return true;
        }
        return false;
    }

    pub fn getAllMetrics(self: *CircuitRegistry) ![]CircuitMetricEntry {
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
    });
    defer breaker.deinit();

    try std.testing.expect(breaker.canExecute());

    breaker.recordFailure();
    breaker.recordFailure();
    breaker.recordFailure();

    try std.testing.expect(!breaker.canExecute());
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
