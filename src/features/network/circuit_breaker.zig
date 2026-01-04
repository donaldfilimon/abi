//! Circuit breaker pattern for network resilience.
const std = @import("std");

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
    last_failure_time: i64,
    open_until: i64,

    pub fn init(allocator: std.mem.Allocator, config: CircuitConfig) CircuitBreaker {
        return .{
            .allocator = allocator,
            .config = config,
            .state = .closed,
            .failure_count = 0,
            .success_count = 0,
            .half_open_calls = 0,
            .last_failure_time = 0,
            .open_until = 0,
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
        self.last_failure_time = 0;
        self.open_until = 0;
    }

    pub fn execute(self: *CircuitBreaker, operation: fn() anyerror![]const u8) !CircuitResult) !CircuitResult {
        const result = try operation();

        if (result) |r| {
            return switch (r) {
                .success => |data| {
                    self.onSuccess();
                    return .{ .success = data };
                },
                .err => |e| {
                    self.onFailure();
                    return .{ .err = e };
                },
            };
        }

        return .{ .rejected = .{ .message = "Circuit open" } };
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
        self.last_failure_time = std.time.timestamp();
        self.failure_count += 1;

        switch (self.state) {
            .closed => {
                if (self.failure_count >= self.config.failure_threshold) {
                    self.state = .open;
                    self.open_until = self.last_failure_time + self.config.timeout_ms;
                }
            },
            .half_open => {
                self.state = .open;
                self.open_until = self.last_failure_time + self.config.timeout_ms;
            },
            .open => {},
        }
    }

    pub fn canExecute(self: *const CircuitBreaker) bool {
        const now = std.time.timestamp();

        switch (self.state) {
            .closed => return true,
            .open => return now >= self.open_until,
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
        const now = std.time.timestamp();
        const time_to_next_state = if (self.state == .open)
            @as(i64, @intCast(@max(0, self.open_until - now)))
        else
            0;

        return .{
            .state = self.state,
            .failure_count = self.failure_count,
            .success_count = self.success_count,
            .last_failure_time = self.last_failure_time,
            .open_until = self.open_until,
            .time_to_next_state = time_to_next_state,
        };
    }
};

pub const CircuitMetrics = struct {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: i64,
    open_until: i64,
    time_to_next_state: i64,
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
        const breaker = try self.allocator.create(CircuitBreaker);
        errdefer self.allocator.destroy(breaker);

        breaker.* = CircuitBreaker.init(self.allocator, config);

        const name_copy = try self.allocator.dupe(u8, name);
        try self.breakers.put(self.allocator, name_copy, breaker.*);
    }

    pub fn getBreaker(self: *CircuitRegistry, name: []const u8) ?*CircuitBreaker {
        return self.breakers.getPtr(name);
    }

    pub fn removeBreaker(self: *CircuitRegistry, name: []const u8) bool {
        if (self.breakers.remove(name)) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr);
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

    try breaker.recordFailure();
    try breaker.recordFailure();
    try breaker.recordFailure();

    try std.testing.expectEqual(CircuitState.open, breaker.getState());
}

test "circuit breaker reset" {
    const allocator = std.testing.allocator;
    var breaker = CircuitBreaker.init(allocator, .{
        .failure_threshold = 3,
        .success_threshold = 2,
    });
    defer breaker.deinit();

    try breaker.recordFailure();
    try breaker.recordFailure();
    try breaker.recordFailure();
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

    try breaker.recordFailure();
    try breaker.recordFailure();
    try breaker.recordFailure();

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
