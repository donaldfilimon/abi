const std = @import("std");

pub const CircuitBreaker = struct {
    pub fn init(_: std.mem.Allocator, _: CircuitConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const CircuitConfig = struct {
    failure_threshold: u32 = 5,
    recovery_timeout_ms: u64 = 30_000,
    half_open_max_requests: u32 = 1,
};

pub const CircuitState = enum { closed, open, half_open };

pub const CircuitRegistry = struct {
    pub fn init(_: std.mem.Allocator) @This() {
        return .{};
    }
    pub fn deinit(_: *@This()) void {}
};

pub const CircuitStats = struct {
    state: CircuitState = .closed,
    failure_count: u32 = 0,
    success_count: u32 = 0,
    total_requests: u64 = 0,
};

pub const CircuitMetrics = struct {
    total_circuits: usize = 0,
    open_circuits: usize = 0,
};

pub const CircuitMetricEntry = struct {
    name: []const u8 = "",
    stats: CircuitStats = .{},
};

pub const NetworkOperationError = error{
    NetworkDisabled,
    CircuitOpen,
    OperationFailed,
};

pub const AggregateStats = struct {
    total_requests: u64 = 0,
    total_failures: u64 = 0,
    total_successes: u64 = 0,
};

test {
    std.testing.refAllDecls(@This());
}
