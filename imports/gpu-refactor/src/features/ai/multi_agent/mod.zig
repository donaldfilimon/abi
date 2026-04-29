//! Multi-agent coordination facade.
//!
//! Provides a coordinator for running a collection of `agents.Agent`
//! instances on a given task. Execution and aggregation behavior live in
//! `coordinator/`, while this file preserves the public module surface.
//!
//! ## Example
//!
//! ```zig
//! const multi_agent = @import("multi_agent");
//!
//! var coord = multi_agent.Coordinator.initWithConfig(allocator, .{
//!     .execution_strategy = .parallel,
//!     .aggregation_strategy = .vote,
//! });
//! defer coord.deinit();
//!
//! try coord.register(agent1);
//! try coord.register(agent2);
//!
//! const result = try coord.runTask("Analyze this code");
//! defer allocator.free(result);
//! ```

const std = @import("std");
const build_options = @import("build_options");
const coordinator_mod = @import("coordinator/mod.zig");
const types_mod = @import("types.zig");

pub const aggregation = @import("aggregation.zig");
pub const messaging = @import("messaging.zig");
pub const roles = @import("roles.zig");
pub const blackboard = @import("blackboard.zig");
pub const workflow = @import("workflow.zig");
pub const supervisor = @import("supervisor.zig");
pub const protocol = @import("protocol.zig");
pub const runner = @import("runner.zig");
pub const coordinator = coordinator_mod;
pub const types = types_mod;

pub const WorkflowRunner = runner.WorkflowRunner;
pub const RunnerConfig = types_mod.RunnerConfig;
pub const WorkflowResult = types_mod.WorkflowResult;
pub const StepResult = types_mod.StepResult;
pub const WorkflowStats = types_mod.WorkflowStats;
pub const RunError = types_mod.RunError;

pub const Error = types_mod.Error;
pub const ExecutionStrategy = types_mod.ExecutionStrategy;
pub const AggregationStrategy = types_mod.AggregationStrategy;
pub const AgentResult = types_mod.AgentResult;
pub const AgentHealth = types_mod.AgentHealth;
pub const CoordinatorConfig = types_mod.CoordinatorConfig;
pub const Coordinator = coordinator_mod.Coordinator;
pub const CoordinatorStats = types_mod.CoordinatorStats;

pub fn isEnabled() bool {
    return build_options.feat_ai;
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
