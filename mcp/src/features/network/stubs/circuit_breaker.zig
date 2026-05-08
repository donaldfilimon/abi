const std = @import("std");

pub const CircuitBreaker = struct {
    pub fn init(_: std.mem.Allocator, _: CircuitConfig) !@This() {
        return error.NetworkDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const CircuitConfig = struct {
    failure_threshold: u32 = 5,
    success_threshold: u32 = 2,
    recovery_timeout_ms: u64 = 30_000,
    timeout_ms: u64 = 60000,
    half_open_max_requests: u32 = 1,
    half_open_max_calls: u32 = 3,
    failure_window_ms: u64 = 0,
    auto_half_open: bool = true,
    on_state_change: ?*const fn ([]const u8, CircuitState, CircuitState) void = null,
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
