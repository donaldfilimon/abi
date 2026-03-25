//! Shared public types for the multi-agent feature.

const std = @import("std");
const retry = @import("../../../foundation/utils/retry.zig");
const workflow_mod = @import("workflow.zig");

pub const Error = error{
    AgentDisabled,
    NoAgents,
    MaxAgentsReached,
    AgentNotFound,
    ExecutionFailed,
    AggregationFailed,
    Timeout,
};

/// How to execute tasks across agents.
pub const ExecutionStrategy = enum {
    sequential,
    parallel,
    pipeline,
    adaptive,

    pub fn toString(self: ExecutionStrategy) []const u8 {
        return @tagName(self);
    }
};

/// How to combine results from multiple agents.
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

/// Agent execution result.
pub const AgentResult = struct {
    agent_index: usize,
    response: []u8,
    success: bool,
    duration_ns: u64,
    timed_out: bool = false,
};

/// Per-agent health tracking for circuit-breaker behavior.
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

/// Configuration for the coordinator.
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

/// Statistics about coordinator execution.
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

pub const RunnerConfig = struct {
    max_retries: u32 = 3,
    step_timeout_ms: u64 = 30_000,
    enable_negotiation: bool = false,
    restart_strategy: @import("supervisor.zig").RestartStrategy = .one_for_one,
    max_history: u32 = 100,
};

pub const WorkflowStats = struct {
    total_steps: u32 = 0,
    completed_steps: u32 = 0,
    failed_steps: u32 = 0,
    skipped_steps: u32 = 0,
    total_retries: u32 = 0,
    total_duration_ms: u64 = 0,
};

pub const StepResult = struct {
    step_id: []const u8,
    output: ?[]const u8,
    status: workflow_mod.StepStatus,
    assigned_profile: ?[]const u8,
    attempts: u32,
    duration_ms: u64,
};

pub const WorkflowResult = struct {
    success: bool,
    step_results: std.StringHashMapUnmanaged(StepResult),
    final_output: ?[]const u8,
    stats: WorkflowStats,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *WorkflowResult) void {
        var iter = self.step_results.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.output) |o| {
                self.allocator.free(o);
            }
        }
        self.step_results.deinit(self.allocator);
        if (self.final_output) |fo| {
            self.allocator.free(fo);
        }
    }
};

pub const RunError = error{
    InvalidWorkflow,
    NoAgents,
    ExecutionFailed,
    Escalated,
    OutOfMemory,
};
