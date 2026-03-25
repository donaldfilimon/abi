//! Multi-agent stub facade.
//!
//! Mirrors the public `mod.zig` surface while the feature is disabled.

const std = @import("std");
const blackboard_mod = @import("blackboard.zig");
const roles_mod = @import("roles.zig");
const supervisor_mod = @import("supervisor.zig");
const protocol_mod = @import("protocol.zig");
const workflow_mod = @import("workflow.zig");
const coordinator_mod = @import("coordinator/stub.zig");
const types_mod = @import("types.zig");

pub const aggregation = @import("aggregation.zig");
pub const messaging = @import("messaging.zig");
pub const roles = @import("roles.zig");
pub const blackboard = @import("blackboard.zig");
pub const workflow = @import("workflow.zig");
pub const supervisor = @import("supervisor.zig");
pub const protocol = @import("protocol.zig");
pub const coordinator = coordinator_mod;
pub const types = types_mod;

pub const Error = types_mod.Error;
pub const ExecutionStrategy = types_mod.ExecutionStrategy;
pub const AggregationStrategy = types_mod.AggregationStrategy;
pub const AgentResult = types_mod.AgentResult;
pub const AgentHealth = types_mod.AgentHealth;
pub const CoordinatorConfig = types_mod.CoordinatorConfig;
pub const Coordinator = coordinator_mod.Coordinator;
pub const CoordinatorStats = types_mod.CoordinatorStats;

pub const runner = struct {
    pub const WorkflowRunner = StubWorkflowRunner;
};

pub const WorkflowRunner = StubWorkflowRunner;
pub const RunnerConfig = types_mod.RunnerConfig;
pub const WorkflowResult = types_mod.WorkflowResult;
pub const StepResult = types_mod.StepResult;
pub const WorkflowStats = types_mod.WorkflowStats;
pub const RunError = types_mod.RunError;

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
            .agent_map = .empty,
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

    pub fn run(_: *StubWorkflowRunner, _: *const workflow_mod.WorkflowDef) !WorkflowResult {
        return RunError.ExecutionFailed;
    }
};

pub fn isEnabled() bool {
    return false;
}

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
    var workflow_runner = WorkflowRunner.init(allocator, .{});
    defer workflow_runner.deinit();
    try std.testing.expectEqual(@as(u32, 3), workflow_runner.config.max_retries);
}

test "stub isEnabled returns false" {
    try std.testing.expect(!isEnabled());
}

test {
    _ = aggregation;
    _ = messaging;
    _ = roles;
    _ = blackboard;
    _ = workflow;
    _ = supervisor;
    _ = protocol;
    _ = runner;
    _ = coordinator;
    _ = types;
}

test {
    std.testing.refAllDecls(@This());
}
