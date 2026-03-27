//! Multi-Agent stub — disabled at compile time.
//!
//! Mirrors all public types and functions from mod.zig so that code
//! using `abi.ai.multi_agent.*` compiles regardless of the feature flag.
//! Every mutating function returns `error.AgentDisabled` or a sensible
//! zero/empty default.

const std = @import("std");
const sync = @import("../../../foundation/mod.zig").sync;
const agents = @import("../agents/stub.zig");
const workflow_mod = @import("workflow.zig");
const blackboard_mod = @import("blackboard.zig");
const roles_mod = @import("roles.zig");
const supervisor_mod = @import("supervisor.zig");
const protocol_mod = @import("protocol.zig");
const types = @import("types.zig");

pub const aggregation = @import("aggregation.zig");
pub const messaging = @import("messaging.zig");
pub const roles = @import("roles.zig");
pub const blackboard = @import("blackboard.zig");
pub const workflow = @import("workflow.zig");
pub const supervisor = @import("supervisor.zig");
pub const protocol = @import("protocol.zig");

// Re-export types
pub const Error = types.Error;
pub const ExecutionStrategy = types.ExecutionStrategy;
pub const AggregationStrategy = types.AggregationStrategy;
pub const AgentResult = types.AgentResult;
pub const AgentHealth = types.AgentHealth;
pub const CoordinatorConfig = types.CoordinatorConfig;
pub const CoordinatorStats = types.CoordinatorStats;
pub const RunnerConfig = types.RunnerConfig;
pub const WorkflowStats = types.WorkflowStats;
pub const StepResult = types.StepResult;
pub const WorkflowResult = types.WorkflowResult;
pub const RunError = types.RunError;

// ---------------------------------------------------------------------------
// Runner stub
// ---------------------------------------------------------------------------

pub const runner = struct {
    pub const WorkflowRunner = StubWorkflowRunner;
};

pub const WorkflowRunner = StubWorkflowRunner;

const StubWorkflowRunner = struct {
    allocator: std.mem.Allocator,
    config: RunnerConfig,
    blackboard: blackboard_mod.Blackboard,
    profile_registry: roles_mod.ProfileRegistry,
    supervisor_: supervisor_mod.Supervisor,
    event_bus: messaging.EventBus,
    conversation_manager: protocol_mod.ConversationManager,
    agent_map: std.StringHashMapUnmanaged(*anyopaque),

    pub fn init(allocator: std.mem.Allocator, config: RunnerConfig) StubWorkflowRunner {
        return .{
            .allocator = allocator,
            .config = config,
            .blackboard = blackboard_mod.Blackboard.init(allocator, config.max_history),
            .profile_registry = roles_mod.ProfileRegistry.init(allocator),
            .supervisor_ = supervisor_mod.Supervisor.init(allocator, .{
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
        self.supervisor_.deinit();
        self.profile_registry.deinit();
        self.blackboard.deinit();
    }

    pub fn registerAgent(_: *StubWorkflowRunner, _: []const u8, _: *anyopaque) !void {
        return error.AgentDisabled;
    }

    pub fn run(_: *StubWorkflowRunner, _: *const workflow_mod.WorkflowDef) !types.WorkflowResult {
        return RunError.ExecutionFailed;
    }
};

// ---------------------------------------------------------------------------
// Coordinator
// ---------------------------------------------------------------------------

pub const Coordinator = struct {
    allocator: std.mem.Allocator = undefined,
    config: CoordinatorConfig = .{},
    agents_list: std.ArrayListUnmanaged(*agents.Agent) = .empty,
    health: std.ArrayListUnmanaged(AgentHealth) = .empty,
    mailboxes: std.ArrayListUnmanaged(messaging.AgentMailbox) = .empty,
    results: std.ArrayListUnmanaged(AgentResult) = .empty,
    mutex: sync.Mutex = .{},
    event_bus: ?messaging.EventBus = null,

    pub fn init(allocator: std.mem.Allocator) Coordinator {
        return initWithConfig(allocator, .{});
    }
    pub fn initWithConfig(allocator: std.mem.Allocator, config: CoordinatorConfig) Coordinator {
        return .{ .allocator = allocator, .config = config, .event_bus = if (config.enable_events) messaging.EventBus.init(allocator) else null };
    }
    pub fn deinit(self: *Coordinator) void {
        self.results.deinit(self.allocator);
        for (self.mailboxes.items) |*mb| mb.deinit();
        self.mailboxes.deinit(self.allocator);
        self.health.deinit(self.allocator);
        self.agents_list.deinit(self.allocator);
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
    var coord = Coordinator.initWithConfig(allocator, .{ .enable_events = true });
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
