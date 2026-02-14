//! Multi‑Agent Stub Module
//!
//! Mirrors the public API of `multi_agent/mod.zig` when the AI feature
//! is disabled. All operations return `error.AgentDisabled`.

const std = @import("std");

// Sub-modules are always available (no feature gating needed)
pub const aggregation = @import("aggregation.zig");
pub const messaging = @import("messaging.zig");

pub const Error = error{
    AgentDisabled,
    NoAgents,
    MaxAgentsReached,
    AgentNotFound,
    ExecutionFailed,
    AggregationFailed,
    Timeout,
};

pub const AgentResult = struct {
    agent_index: usize,
    response: []u8,
    success: bool,
    duration_ns: u64,
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
    pub fn agentCount(_: *const Coordinator) usize {
        return 0;
    }
    pub fn onEvent(_: *Coordinator, _: messaging.EventType, _: messaging.EventCallback) !void {
        return error.AgentDisabled;
    }
    pub fn runTask(_: *Coordinator, _: []const u8) Error![]u8 {
        return error.AgentDisabled;
    }
    pub fn getStats(self: *const Coordinator) CoordinatorStats {
        _ = self;
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
        return @as(f64, @floatFromInt(self.success_count)) /
            @as(f64, @floatFromInt(self.result_count));
    }
};

pub fn isEnabled() bool {
    return false;
}
