//! Integration Tests: Multi-agent public surface

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const multi_agent = abi.ai.multi_agent;
const agents = abi.ai.agents;

test "multi-agent: public types are available" {
    _ = multi_agent.Error;
    _ = multi_agent.ExecutionStrategy;
    _ = multi_agent.AggregationStrategy;
    _ = multi_agent.AgentResult;
    _ = multi_agent.AgentHealth;
    _ = multi_agent.CoordinatorConfig;
    _ = multi_agent.CoordinatorStats;
    _ = multi_agent.RunnerConfig;
    _ = multi_agent.WorkflowStats;
    _ = multi_agent.StepResult;
    _ = multi_agent.WorkflowResult;
    _ = multi_agent.RunError;
    _ = multi_agent.WorkflowRunner;
}

test "multi-agent: isEnabled tracks the AI feature flag" {
    try std.testing.expectEqual(build_options.feat_ai, multi_agent.isEnabled());
}

test "multi-agent: coordinator init and stats surface" {
    const allocator = std.testing.allocator;
    var coord = multi_agent.Coordinator.init(allocator);
    defer coord.deinit();

    try std.testing.expectEqual(@as(usize, 0), coord.agentCount());

    const stats = coord.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.agent_count);
    try std.testing.expectEqual(@as(usize, 0), stats.result_count);
    try std.testing.expectEqual(@as(f64, 0.0), stats.successRate());

    var cfg_coord = multi_agent.Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .parallel,
        .aggregation_strategy = .vote,
        .max_agents = 4,
        .enable_events = true,
    });
    defer cfg_coord.deinit();

    try std.testing.expectEqual(multi_agent.ExecutionStrategy.parallel, cfg_coord.config.execution_strategy);
    try std.testing.expectEqual(multi_agent.AggregationStrategy.vote, cfg_coord.config.aggregation_strategy);
    try std.testing.expectEqual(@as(u32, 4), cfg_coord.config.max_agents);
    try std.testing.expect(cfg_coord.event_bus != null);
}

test "multi-agent: coordinator and workflow runner register echo agent" {
    if (!multi_agent.isEnabled()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var agent = try agents.Agent.init(allocator, .{
        .name = "multi-agent-echo",
        .backend = .echo,
    });
    defer agent.deinit();

    var coord = multi_agent.Coordinator.init(allocator);
    defer coord.deinit();
    try coord.register(&agent);
    try std.testing.expectEqual(@as(usize, 1), coord.agentCount());

    var runner = multi_agent.WorkflowRunner.init(allocator, .{});
    defer runner.deinit();
    try runner.registerAgent("multi-agent-echo", &agent);
    try std.testing.expectEqual(@as(usize, 1), runner.agent_map.count());
}

test "multi-agent: public enums stringify cleanly" {
    try std.testing.expectEqualStrings("sequential", multi_agent.ExecutionStrategy.sequential.toString());
    try std.testing.expectEqualStrings("parallel", multi_agent.ExecutionStrategy.parallel.toString());
    try std.testing.expectEqualStrings("concatenate", multi_agent.AggregationStrategy.concatenate.toString());
    try std.testing.expectEqualStrings("vote", multi_agent.AggregationStrategy.vote.toString());
}

test "multi-agent: workflow runner init surface" {
    const allocator = std.testing.allocator;
    var runner = multi_agent.WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    try std.testing.expectEqual(@as(u32, 3), runner.config.max_retries);
    try std.testing.expectEqual(@as(u64, 30_000), runner.config.step_timeout_ms);
}

test "multi-agent: runTask with no agents returns NoAgents" {
    const allocator = std.testing.allocator;
    var coord = multi_agent.Coordinator.init(allocator);
    defer coord.deinit();

    try std.testing.expectError(multi_agent.Error.NoAgents, coord.runTask("public multi-agent task"));
}

test {
    std.testing.refAllDecls(@This());
}
