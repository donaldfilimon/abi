const std = @import("std");
const multi_agent = @import("mod.zig");
const types_mod = @import("types.zig");
const workflow_mod = @import("workflow.zig");
const agents_mod = @import("../agents/mod.zig");

const Coordinator = multi_agent.Coordinator;
const CoordinatorConfig = multi_agent.CoordinatorConfig;
const CoordinatorStats = multi_agent.CoordinatorStats;
const ExecutionStrategy = multi_agent.ExecutionStrategy;
const AggregationStrategy = multi_agent.AggregationStrategy;
const AgentHealth = multi_agent.AgentHealth;
const Error = multi_agent.Error;

test "coordinator public surface basics" {
    const allocator = std.testing.allocator;
    var coordinator = multi_agent.Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .parallel,
        .aggregation_strategy = .vote,
        .enable_events = true,
    });
    defer coordinator.deinit();

    try std.testing.expectEqual(multi_agent.ExecutionStrategy.parallel, coordinator.config.execution_strategy);
    try std.testing.expectEqual(multi_agent.AggregationStrategy.vote, coordinator.config.aggregation_strategy);
    try std.testing.expect(coordinator.event_bus != null);
    try std.testing.expectEqual(@as(usize, 0), coordinator.agentCount());
    try std.testing.expectError(multi_agent.Error.NoAgents, coordinator.runTask("no agents"));
}

test "coordinator helper types remain usable" {
    var health = multi_agent.AgentHealth{ .failure_threshold = 2 };
    try std.testing.expect(health.canAttempt());
    health.recordFailure();
    try std.testing.expect(health.canAttempt());
    health.recordFailure();
    try std.testing.expect(!health.canAttempt());
    health.reset();
    try std.testing.expect(health.canAttempt());

    const stats = multi_agent.CoordinatorStats{
        .agent_count = 2,
        .result_count = 4,
        .success_count = 3,
        .avg_duration_ns = 100,
    };
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), stats.successRate(), 0.001);
}

test "workflow runner nested aliases remain source-compatible" {
    const config: multi_agent.RunnerConfig = .{
        .max_retries = 4,
    };
    try std.testing.expectEqual(@as(u32, 4), config.max_retries);

    const run_error = multi_agent.RunError.NoAgents;
    try std.testing.expectEqual(multi_agent.RunError.NoAgents, run_error);

    var step_results = std.StringHashMapUnmanaged(multi_agent.StepResult).empty;
    defer step_results.deinit(std.testing.allocator);

    const result = multi_agent.WorkflowResult{
        .success = false,
        .step_results = step_results,
        .final_output = null,
        .stats = multi_agent.WorkflowStats{},
        .allocator = std.testing.allocator,
    };
    try std.testing.expect(!result.success);
}

test "workflow runner registerAgent keeps agents.Agent signature" {
    const allocator = std.testing.allocator;
    var runner = multi_agent.WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    var agent = try agents_mod.Agent.init(allocator, .{
        .name = "echo-agent",
        .backend = .echo,
        .enable_history = false,
    });
    defer agent.deinit();

    try runner.registerAgent("echo-agent", &agent);
    try std.testing.expectEqual(@as(usize, 1), runner.agent_map.count());
    try std.testing.expect(runner.agent_map.get("echo-agent") != null);
}

test "workflow runner executes a single-step workflow" {
    const allocator = std.testing.allocator;
    var runner = multi_agent.WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    var agent = try agents_mod.Agent.init(allocator, .{
        .name = "echo-agent",
        .backend = .echo,
        .enable_history = false,
    });
    defer agent.deinit();

    try runner.registerAgent("echo-agent", &agent);

    const steps = [_]workflow_mod.Step{
        .{
            .id = "step1",
            .description = "Echo test",
            .depends_on = &.{},
            .required_capabilities = &.{},
            .input_keys = &.{},
            .output_key = "step1:output",
            .prompt_template = "Hello world",
        },
    };

    const workflow = workflow_mod.WorkflowDef{
        .id = "test-wf",
        .name = "Test Workflow",
        .description = "Single step test",
        .steps = &steps,
    };

    var result = try runner.run(&workflow);
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expectEqual(@as(u32, 1), result.stats.total_steps);
    try std.testing.expectEqual(@as(u32, 1), result.stats.completed_steps);
    try std.testing.expectEqual(@as(u32, 0), result.stats.failed_steps);
    try std.testing.expect(result.final_output != null);

    const blackboard_entry = runner.blackboard.get("step1:output");
    try std.testing.expect(blackboard_entry != null);
    try std.testing.expect(std.mem.startsWith(u8, blackboard_entry.?.value, "Echo:"));
}

test "workflow runner executes a multi-step dag" {
    const allocator = std.testing.allocator;
    var runner = multi_agent.WorkflowRunner.init(allocator, .{});
    defer runner.deinit();

    var agent = try agents_mod.Agent.init(allocator, .{
        .name = "agent1",
        .backend = .echo,
        .enable_history = false,
    });
    defer agent.deinit();

    try runner.registerAgent("agent1", &agent);
    try runner.blackboard.put("task:input", "analyze this code", "system");

    const steps = [_]workflow_mod.Step{
        .{
            .id = "step1",
            .description = "Gather info",
            .depends_on = &.{},
            .required_capabilities = &.{},
            .input_keys = &.{"task:input"},
            .output_key = "step1:output",
            .prompt_template = "Gather: {input}",
        },
        .{
            .id = "step2",
            .description = "Analyze",
            .depends_on = &.{"step1"},
            .required_capabilities = &.{},
            .input_keys = &.{"step1:output"},
            .output_key = "step2:output",
            .prompt_template = "Analyze: {input}",
        },
        .{
            .id = "step3",
            .description = "Report",
            .depends_on = &.{"step2"},
            .required_capabilities = &.{},
            .input_keys = &.{"step2:output"},
            .output_key = "step3:output",
            .prompt_template = "Report: {input}",
        },
    };

    const workflow = workflow_mod.WorkflowDef{
        .id = "dag-test",
        .name = "DAG Test",
        .description = "Three step pipeline",
        .steps = &steps,
    };

    var result = try runner.run(&workflow);
    defer result.deinit();

    try std.testing.expect(result.success);
    try std.testing.expectEqual(@as(u32, 3), result.stats.total_steps);
    try std.testing.expectEqual(@as(u32, 3), result.stats.completed_steps);
    try std.testing.expectEqual(@as(u32, 0), result.stats.failed_steps);
    try std.testing.expect(result.final_output != null);
    try std.testing.expect(runner.blackboard.get("step1:output") != null);
    try std.testing.expect(runner.blackboard.get("step2:output") != null);
    try std.testing.expect(runner.blackboard.get("step3:output") != null);
}

test "workflow runner returns public errors for invalid runs" {
    const allocator = std.testing.allocator;

    var no_agent_runner = multi_agent.WorkflowRunner.init(allocator, .{});
    defer no_agent_runner.deinit();

    const simple_steps = [_]workflow_mod.Step{
        .{
            .id = "step1",
            .description = "Test",
            .depends_on = &.{},
            .required_capabilities = &.{},
            .input_keys = &.{},
            .output_key = "step1:output",
            .prompt_template = "Test",
        },
    };

    const simple_workflow = workflow_mod.WorkflowDef{
        .id = "simple",
        .name = "Simple",
        .description = "Simple",
        .steps = &simple_steps,
    };

    try std.testing.expectError(
        multi_agent.RunError.NoAgents,
        no_agent_runner.run(&simple_workflow),
    );

    var invalid_runner = multi_agent.WorkflowRunner.init(allocator, .{});
    defer invalid_runner.deinit();

    var agent = try agents_mod.Agent.init(allocator, .{
        .name = "agent",
        .backend = .echo,
        .enable_history = false,
    });
    defer agent.deinit();
    try invalid_runner.registerAgent("agent", &agent);

    const invalid_workflow = workflow_mod.WorkflowDef{
        .id = "empty",
        .name = "Empty",
        .description = "No steps",
        .steps = &.{},
    };

    try std.testing.expectError(
        multi_agent.RunError.InvalidWorkflow,
        invalid_runner.run(&invalid_workflow),
    );
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
    std.testing.refAllDecls(@This());
}
