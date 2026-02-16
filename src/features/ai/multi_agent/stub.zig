//! Multi-Agent stub — disabled at compile time.

const std = @import("std");
const retry = @import("../../../services/shared/utils/retry.zig");

pub const aggregation = @import("aggregation.zig");
pub const messaging = @import("messaging.zig");

pub const Error = error{ AgentDisabled, NoAgents, MaxAgentsReached, AgentNotFound, ExecutionFailed, AggregationFailed, Timeout };

pub const AgentResult = struct { agent_index: usize, response: []u8, success: bool, duration_ns: u64, timed_out: bool = false };

pub const AgentHealth = struct {
    consecutive_failures: u32 = 0,
    failure_threshold: u32 = 5,
    total_successes: u64 = 0,
    total_failures: u64 = 0,
    is_open: bool = false,
    pub fn recordSuccess(self: *AgentHealth) void {
        self.consecutive_failures = 0;
        self.total_successes += 1;
        self.is_open = false;
    }
    pub fn recordFailure(self: *AgentHealth) void {
        self.consecutive_failures += 1;
        self.total_failures += 1;
        if (self.consecutive_failures >= self.failure_threshold) self.is_open = true;
    }
    pub fn canAttempt(self: *const AgentHealth) bool {
        return !self.is_open;
    }
    pub fn successRate(self: *const AgentHealth) f64 {
        const total = self.total_successes + self.total_failures;
        if (total == 0) return 1.0;
        return @as(f64, @floatFromInt(self.total_successes)) / @as(f64, @floatFromInt(total));
    }
    pub fn reset(self: *AgentHealth) void {
        self.consecutive_failures = 0;
        self.is_open = false;
    }
};

pub const ExecutionStrategy = enum {
    sequential,
    parallel,
    pipeline,
    adaptive,
    pub fn toString(self: ExecutionStrategy) []const u8 {
        return @tagName(self);
    }
};

pub const AggregationStrategy = enum {
    concatenate,
    vote,
    select_best,
    merge,
    first_success,
    pub fn toString(self: AggregationStrategy) []const u8 {
        return @tagName(self);
    }
};

pub const CoordinatorConfig = struct {
    execution_strategy: ExecutionStrategy = .sequential,
    aggregation_strategy: AggregationStrategy = .concatenate,
    max_agents: u32 = 100,
    agent_timeout_ms: u64 = 30_000,
    enable_parallel: bool = true,
    enable_events: bool = false,
    max_threads: u32 = 0,
    retry_config: retry.RetryConfig = .{},
    circuit_breaker_threshold: u32 = 5,
    pub fn defaults() CoordinatorConfig {
        return .{};
    }
};

pub const Coordinator = struct {
    allocator: std.mem.Allocator = undefined,
    config: CoordinatorConfig = .{},
    agents: std.ArrayListUnmanaged(*anyopaque) = .{},
    pub fn init(allocator: std.mem.Allocator) Coordinator {
        return .{ .allocator = allocator };
    }
    pub fn initWithConfig(allocator: std.mem.Allocator, config: CoordinatorConfig) Coordinator {
        return .{ .allocator = allocator, .config = config };
    }
    pub fn deinit(self: *Coordinator) void {
        self.agents.deinit(self.allocator);
    }
    pub fn register(_: *Coordinator, _: *anyopaque) Error!void {
        return error.AgentDisabled;
    }
    pub fn getAgentHealth(_: *const Coordinator, _: usize) ?AgentHealth {
        return null;
    }
    pub fn sendMessage(_: *Coordinator, _: messaging.AgentMessage) Error!void {
        return error.AgentDisabled;
    }
    pub fn pendingMessages(_: *const Coordinator, _: usize) ?usize {
        return null;
    }
    pub fn agentCount(_: *const Coordinator) usize {
        return 0;
    }
    pub fn onEvent(_: *Coordinator, _: messaging.EventType, _: messaging.EventCallback) !void {
        return error.AgentDisabled;
    }
    pub fn runTask(_: *Coordinator, _: []const u8) Error![]u8 {
        return error.AgentDisabled;
    }
    pub fn getStats(_: *const Coordinator) CoordinatorStats {
        return .{};
    }
};

pub const CoordinatorStats = struct {
    agent_count: usize = 0,
    result_count: usize = 0,
    success_count: usize = 0,
    avg_duration_ns: u64 = 0,
    pub fn successRate(self: CoordinatorStats) f64 {
        if (self.result_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.success_count)) / @as(f64, @floatFromInt(self.result_count));
    }
};

pub fn isEnabled() bool {
    return false;
}
