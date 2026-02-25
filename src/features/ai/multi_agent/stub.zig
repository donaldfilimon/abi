//! Multi-Agent stub — disabled at compile time.
//!
//! Mirrors all public types and functions from mod.zig so that code
//! using `abi.ai.multi_agent.*` compiles regardless of the feature flag.
//! Every mutating function returns `error.AgentDisabled` or a sensible
//! zero/empty default.

const std = @import("std");
const retry = @import("../../../services/shared/utils/retry.zig");
const sync = @import("../../../services/shared/sync.zig");
const workflow_mod = @import("workflow.zig");
const blackboard_mod = @import("blackboard.zig");
const roles_mod = @import("roles.zig");
const supervisor_mod = @import("supervisor.zig");
const protocol_mod = @import("protocol.zig");

pub const aggregation = @import("aggregation.zig");
pub const messaging = @import("messaging.zig");
pub const roles = @import("roles.zig");
pub const blackboard = @import("blackboard.zig");
pub const workflow = @import("workflow.zig");
pub const supervisor = @import("supervisor.zig");
pub const protocol = @import("protocol.zig");

// ---------------------------------------------------------------------------
// Runner stub (inline — cannot import runner.zig because it depends on the
// real agents module which is unavailable when AI is disabled)
// ---------------------------------------------------------------------------

pub const runner = struct {
    pub const WorkflowRunner = StubWorkflowRunner;
};

pub const WorkflowRunner = StubWorkflowRunner;

const StubWorkflowRunner = struct {
    allocator: std.mem.Allocator,
    config: RunnerConfig,
    blackboard: blackboard_mod.Blackboard,
    persona_registry: roles_mod.PersonaRegistry,
    supervisor: supervisor_mod.Supervisor,
    event_bus: messaging.EventBus,
    conversation_manager: protocol_mod.ConversationManager,
    agent_map: std.StringHashMapUnmanaged(*anyopaque),

    pub const RunnerConfig = struct {
        max_retries: u32 = 3,
        step_timeout_ms: u64 = 30_000,
        enable_negotiation: bool = false,
        restart_strategy: supervisor_mod.RestartStrategy = .one_for_one,
        max_history: u32 = 100,
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

    pub const StepResult = struct {
        step_id: []const u8,
        output: ?[]const u8,
        status: workflow_mod.StepStatus,
        assigned_persona: ?[]const u8,
        attempts: u32,
        duration_ms: u64,
    };

    pub const WorkflowStats = struct {
        total_steps: u32 = 0,
        completed_steps: u32 = 0,
        failed_steps: u32 = 0,
        skipped_steps: u32 = 0,
        total_retries: u32 = 0,
        total_duration_ms: u64 = 0,
    };

    pub const RunError = error{
        InvalidWorkflow,
        NoAgents,
        ExecutionFailed,
        Escalated,
        OutOfMemory,
    };

    pub fn init(allocator: std.mem.Allocator, config: RunnerConfig) StubWorkflowRunner {
        return .{
            .allocator = allocator,
            .config = config,
            .blackboard = blackboard_mod.Blackboard.init(allocator, config.max_history),
            .persona_registry = roles_mod.PersonaRegistry.init(allocator),
            .supervisor = supervisor_mod.Supervisor.init(allocator, .{
                .restart_strategy = config.restart_strategy,
                .max_retries = config.max_retries,
            }),
            .event_bus = messaging.EventBus.init(allocator),
            .conversation_manager = protocol_mod.ConversationManager.init(allocator, 20),
            .agent_map = .{},
        };
    }

    pub fn deinit(self: *StubWorkflowRunner) void {
        self.agent_map.deinit(self.allocator);
        self.conversation_manager.deinit();
        self.event_bus.deinit();
        self.supervisor.deinit();
        self.persona_registry.deinit();
        self.blackboard.deinit();
    }

    /// Register a named agent — disabled stub always returns error.
    pub fn registerAgent(_: *StubWorkflowRunner, _: []const u8, _: *anyopaque) !void {
        return error.AgentDisabled;
    }

    /// Run a workflow — disabled stub always returns error.
    pub fn run(_: *StubWorkflowRunner, _: *const workflow_mod.WorkflowDef) !WorkflowResult {
        return RunError.ExecutionFailed;
    }
};

// ---------------------------------------------------------------------------
// Error set
// ---------------------------------------------------------------------------

pub const Error = error{
    AgentDisabled,
    NoAgents,
    MaxAgentsReached,
    AgentNotFound,
    ExecutionFailed,
    AggregationFailed,
    Timeout,
};

// ---------------------------------------------------------------------------
// Execution / Aggregation strategies
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Agent result / health
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Coordinator config
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

/// Coordinator holds a list of agents and orchestrates task execution.
/// Stub version: all fields present for layout parity; mutating ops return errors.
pub const Coordinator = struct {
    allocator: std.mem.Allocator = undefined,
    config: CoordinatorConfig = .{},
    agents: std.ArrayListUnmanaged(*anyopaque) = .empty,
    health: std.ArrayListUnmanaged(AgentHealth) = .empty,
    mailboxes: std.ArrayListUnmanaged(messaging.AgentMailbox) = .empty,
    results: std.ArrayListUnmanaged(AgentResult) = .empty,
    mutex: sync.Mutex = .{},
    event_bus: ?messaging.EventBus = null,

    pub fn init(allocator: std.mem.Allocator) Coordinator {
        return initWithConfig(allocator, .{});
    }

    pub fn initWithConfig(allocator: std.mem.Allocator, config: CoordinatorConfig) Coordinator {
        return .{
            .allocator = allocator,
            .config = config,
            .agents = .empty,
            .health = .empty,
            .mailboxes = .empty,
            .results = .empty,
            .mutex = .{},
            .event_bus = if (config.enable_events) messaging.EventBus.init(allocator) else null,
        };
    }

    pub fn deinit(self: *Coordinator) void {
        self.results.deinit(self.allocator);
        for (self.mailboxes.items) |*mb| mb.deinit();
        self.mailboxes.deinit(self.allocator);
        self.health.deinit(self.allocator);
        self.agents.deinit(self.allocator);
        if (self.event_bus) |*bus| bus.deinit();
        self.* = undefined;
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

// ---------------------------------------------------------------------------
// Coordinator stats
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Check if multi-agent is enabled.
pub fn isEnabled() bool {
    return false;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "stub coordinator init and deinit" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    try std.testing.expectEqual(@as(usize, 0), coord.agentCount());
}

test "stub coordinator with config" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .parallel,
        .aggregation_strategy = .vote,
        .max_agents = 10,
    });
    defer coord.deinit();

    try std.testing.expectEqual(ExecutionStrategy.parallel, coord.config.execution_strategy);
    try std.testing.expectEqual(AggregationStrategy.vote, coord.config.aggregation_strategy);
}

test "stub coordinator with event bus" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.initWithConfig(allocator, .{
        .enable_events = true,
    });
    defer coord.deinit();

    try std.testing.expect(coord.event_bus != null);
}

test "stub workflow runner init and deinit" {
    const allocator = std.testing.allocator;
    var wr = WorkflowRunner.init(allocator, .{});
    defer wr.deinit();

    try std.testing.expectEqual(@as(u32, 3), wr.config.max_retries);
}

test "stub isEnabled returns false" {
    try std.testing.expect(!isEnabled());
}

test {
    _ = aggregation;
    _ = messaging;
    _ = runner;
}

test {
    std.testing.refAllDecls(@This());
}
