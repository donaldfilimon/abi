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

test "coordinator init and deinit" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    try std.testing.expectEqual(@as(usize, 0), coord.agentCount());
}

test "coordinator with config" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .parallel,
        .aggregation_strategy = .vote,
        .max_agents = 10,
    });
    defer coord.deinit();

    try std.testing.expectEqual(ExecutionStrategy.parallel, coord.config.execution_strategy);
    try std.testing.expectEqual(AggregationStrategy.vote, coord.config.aggregation_strategy);
    try std.testing.expectEqual(@as(u32, 10), coord.config.max_agents);
}

test "coordinator with event bus" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.initWithConfig(allocator, .{
        .enable_events = true,
    });
    defer coord.deinit();

    try std.testing.expect(coord.event_bus != null);
}

test "coordinator runTask with no agents returns error" {
    const allocator = std.testing.allocator;
    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    const result = coord.runTask("test task");
    try std.testing.expectError(Error.NoAgents, result);
}

test "execution strategy toString" {
    try std.testing.expectEqualStrings("sequential", ExecutionStrategy.sequential.toString());
    try std.testing.expectEqualStrings("parallel", ExecutionStrategy.parallel.toString());
    try std.testing.expectEqualStrings("pipeline", ExecutionStrategy.pipeline.toString());
}

test "aggregation strategy toString" {
    try std.testing.expectEqualStrings("concatenate", AggregationStrategy.concatenate.toString());
    try std.testing.expectEqualStrings("vote", AggregationStrategy.vote.toString());
    try std.testing.expectEqualStrings("select_best", AggregationStrategy.select_best.toString());
    try std.testing.expectEqualStrings("merge", AggregationStrategy.merge.toString());
}

test "coordinator stats" {
    const stats = CoordinatorStats{
        .agent_count = 5,
        .result_count = 10,
        .success_count = 8,
        .avg_duration_ns = 1000,
    };

    try std.testing.expectApproxEqAbs(@as(f64, 0.8), stats.successRate(), 0.001);
}

test "coordinator config defaults" {
    const config = CoordinatorConfig.defaults();
    try std.testing.expectEqual(ExecutionStrategy.sequential, config.execution_strategy);
    try std.testing.expectEqual(AggregationStrategy.concatenate, config.aggregation_strategy);
    try std.testing.expectEqual(@as(u32, 100), config.max_agents);
    try std.testing.expect(!config.enable_events);
}

test "agent health circuit breaker" {
    var health = AgentHealth{};

    try std.testing.expect(health.canAttempt());
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), health.successRate(), 0.001);

    health.recordSuccess();
    health.recordSuccess();
    try std.testing.expectEqual(@as(u64, 2), health.total_successes);
    try std.testing.expect(health.canAttempt());

    health.recordFailure();
    health.recordFailure();
    try std.testing.expect(health.canAttempt());
    try std.testing.expectEqual(@as(u32, 2), health.consecutive_failures);

    health.recordSuccess();
    try std.testing.expectEqual(@as(u32, 0), health.consecutive_failures);
    try std.testing.expect(health.canAttempt());
}

test "agent health trips open after threshold" {
    var health = AgentHealth{ .failure_threshold = 3 };

    health.recordFailure();
    health.recordFailure();
    try std.testing.expect(health.canAttempt());
    health.recordFailure();
    try std.testing.expect(!health.canAttempt());

    health.reset();
    try std.testing.expect(health.canAttempt());
    try std.testing.expectEqual(@as(u32, 0), health.consecutive_failures);
}

test "agent health success rate" {
    var health = AgentHealth{};
    health.recordSuccess();
    health.recordSuccess();
    health.recordFailure();
    try std.testing.expectApproxEqAbs(@as(f64, 0.667), health.successRate(), 0.01);
}

test "coordinator config with retry" {
    const config = CoordinatorConfig{
        .retry_config = .{ .max_retries = 5, .initial_delay_ms = 50 },
        .circuit_breaker_threshold = 3,
    };
    try std.testing.expectEqual(@as(u32, 5), config.retry_config.max_retries);
    try std.testing.expectEqual(@as(u32, 3), config.circuit_breaker_threshold);
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
