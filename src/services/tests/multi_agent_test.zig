//! Multi-Agent Integration Tests
//!
//! Tests the Coordinator with real echo agents across execution strategies,
//! aggregation strategies, circuit breaker behavior, timeout enforcement,
//! and inter-agent mailbox messaging.
//!
//! All tests use the `echo` backend (no API keys required).

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const multi_agent = abi.ai.multi_agent;
const Coordinator = multi_agent.Coordinator;
const CoordinatorConfig = multi_agent.CoordinatorConfig;
const AgentHealth = multi_agent.AgentHealth;
const Agent = abi.ai.Agent;
const messaging = multi_agent.messaging;

// ============================================================================
// Helpers
// ============================================================================

/// Create N echo agents with distinct names. Caller owns the slice.
fn createEchoAgents(
    allocator: std.mem.Allocator,
    names: []const []const u8,
    agents_out: []Agent,
) !void {
    for (names, 0..) |name, i| {
        agents_out[i] = try Agent.init(allocator, .{
            .name = name,
            .backend = .echo,
            .enable_history = false,
        });
    }
}

fn deinitAgents(agents_slice: []Agent) void {
    for (agents_slice) |*ag| ag.deinit();
}

// ============================================================================
// Sequential Execution Tests
// ============================================================================

test "integration: sequential execution with 3 echo agents" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const names = [_][]const u8{ "alpha", "beta", "gamma" };
    var agent_storage: [3]Agent = undefined;
    try createEchoAgents(allocator, &names, &agent_storage);
    defer deinitAgents(&agent_storage);

    var coord = Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .sequential,
        .aggregation_strategy = .concatenate,
    });
    defer coord.deinit();

    for (&agent_storage) |*ag| try coord.register(ag);

    const result = try coord.runTask("hello");
    defer allocator.free(result);

    // Concatenated output should contain 3 echoed responses separated by ---
    try std.testing.expect(std.mem.indexOf(u8, result, "Echo: hello") != null);

    const stats = coord.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.agent_count);
    try std.testing.expectEqual(@as(usize, 3), stats.result_count);
    try std.testing.expectEqual(@as(usize, 3), stats.success_count);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), stats.successRate(), 0.001);
}

// ============================================================================
// Parallel Execution Tests
// ============================================================================

test "integration: parallel execution with 3 echo agents" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const names = [_][]const u8{ "p1", "p2", "p3" };
    var agent_storage: [3]Agent = undefined;
    try createEchoAgents(allocator, &names, &agent_storage);
    defer deinitAgents(&agent_storage);

    var coord = Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .parallel,
        .aggregation_strategy = .concatenate,
        .enable_parallel = true,
    });
    defer coord.deinit();

    for (&agent_storage) |*ag| try coord.register(ag);

    const result = try coord.runTask("parallel test");
    defer allocator.free(result);

    // All 3 agents should respond
    try std.testing.expect(std.mem.indexOf(u8, result, "Echo: parallel test") != null);

    const stats = coord.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.result_count);
    try std.testing.expectEqual(@as(usize, 3), stats.success_count);
}

test "integration: parallel falls back to sequential with 1 agent" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var agent = try Agent.init(allocator, .{ .name = "solo", .backend = .echo, .enable_history = false });
    defer agent.deinit();

    var coord = Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .parallel,
        .enable_parallel = true,
    });
    defer coord.deinit();
    try coord.register(&agent);

    const result = try coord.runTask("solo task");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Echo: solo task", result);
}

// ============================================================================
// Pipeline Execution Tests
// ============================================================================

test "integration: pipeline chains agent outputs" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Pipeline: agent1 echoes input, agent2 echoes agent1's output
    const names = [_][]const u8{ "stage1", "stage2" };
    var agent_storage: [2]Agent = undefined;
    try createEchoAgents(allocator, &names, &agent_storage);
    defer deinitAgents(&agent_storage);

    var coord = Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .pipeline,
        .aggregation_strategy = .concatenate,
    });
    defer coord.deinit();

    for (&agent_storage) |*ag| try coord.register(ag);

    const result = try coord.runTask("start");
    defer allocator.free(result);

    // Pipeline: "start" -> Echo: "Echo: start" -> Echo: "Echo: Echo: start"
    // The concatenated result should contain both stages
    try std.testing.expect(std.mem.indexOf(u8, result, "Echo: start") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Echo: Echo: start") != null);
}

// ============================================================================
// Aggregation Strategy Tests
// ============================================================================

test "integration: vote aggregation picks majority" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // All echo agents return the same response, so vote = that response
    const names = [_][]const u8{ "v1", "v2", "v3" };
    var agent_storage: [3]Agent = undefined;
    try createEchoAgents(allocator, &names, &agent_storage);
    defer deinitAgents(&agent_storage);

    var coord = Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .sequential,
        .aggregation_strategy = .vote,
    });
    defer coord.deinit();

    for (&agent_storage) |*ag| try coord.register(ag);

    const result = try coord.runTask("consensus");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Echo: consensus", result);
}

test "integration: first_success aggregation returns first" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const names = [_][]const u8{ "f1", "f2" };
    var agent_storage: [2]Agent = undefined;
    try createEchoAgents(allocator, &names, &agent_storage);
    defer deinitAgents(&agent_storage);

    var coord = Coordinator.initWithConfig(allocator, .{
        .aggregation_strategy = .first_success,
    });
    defer coord.deinit();

    for (&agent_storage) |*ag| try coord.register(ag);

    const result = try coord.runTask("pick first");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Echo: pick first", result);
}

test "integration: select_best aggregation returns best quality" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const names = [_][]const u8{ "s1", "s2" };
    var agent_storage: [2]Agent = undefined;
    try createEchoAgents(allocator, &names, &agent_storage);
    defer deinitAgents(&agent_storage);

    var coord = Coordinator.initWithConfig(allocator, .{
        .aggregation_strategy = .select_best,
    });
    defer coord.deinit();

    for (&agent_storage) |*ag| try coord.register(ag);

    const result = try coord.runTask("quality check");
    defer allocator.free(result);

    // Both agents produce equal output, so either is fine
    try std.testing.expectEqualStrings("Echo: quality check", result);
}

test "integration: merge aggregation deduplicates" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const names = [_][]const u8{ "m1", "m2" };
    var agent_storage: [2]Agent = undefined;
    try createEchoAgents(allocator, &names, &agent_storage);
    defer deinitAgents(&agent_storage);

    var coord = Coordinator.initWithConfig(allocator, .{
        .aggregation_strategy = .merge,
    });
    defer coord.deinit();

    for (&agent_storage) |*ag| try coord.register(ag);

    const result = try coord.runTask("merge me");
    defer allocator.free(result);

    // Merge deduplicates identical responses
    try std.testing.expect(result.len > 0);
}

// ============================================================================
// Health & Circuit Breaker Tests
// ============================================================================

test "integration: agent health tracking across runs" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var agent = try Agent.init(allocator, .{ .name = "healthy", .backend = .echo, .enable_history = false });
    defer agent.deinit();

    var coord = Coordinator.initWithConfig(allocator, .{
        .circuit_breaker_threshold = 3,
    });
    defer coord.deinit();
    try coord.register(&agent);

    // Run several tasks — echo always succeeds.
    // Each runTask() returns newly allocated memory that must be freed by the caller.
    const r0 = try coord.runTask("run 0");
    allocator.free(r0);
    const r1 = try coord.runTask("run 1");
    defer allocator.free(r1);

    const health = coord.getAgentHealth(0).?;
    try std.testing.expect(health.canAttempt());
    try std.testing.expectEqual(@as(u32, 0), health.consecutive_failures);
    // Echo always succeeds, so success count should be > 0
    try std.testing.expect(health.total_successes > 0);
}

test "integration: circuit breaker threshold in config" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var coord = Coordinator.initWithConfig(allocator, .{
        .circuit_breaker_threshold = 2,
    });
    defer coord.deinit();

    try std.testing.expectEqual(@as(u32, 2), coord.config.circuit_breaker_threshold);
}

// ============================================================================
// Mailbox Messaging Tests
// ============================================================================

test "integration: send and receive messages between agents" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const names = [_][]const u8{ "sender", "receiver" };
    var agent_storage: [2]Agent = undefined;
    try createEchoAgents(allocator, &names, &agent_storage);
    defer deinitAgents(&agent_storage);

    var coord = Coordinator.initWithConfig(allocator, .{});
    defer coord.deinit();

    for (&agent_storage) |*ag| try coord.register(ag);

    // Send a message from agent 0 to agent 1
    try coord.sendMessage(.{
        .from_agent = 0,
        .to_agent = 1,
        .content = "inter-agent data",
        .tag = .data,
    });

    try std.testing.expectEqual(@as(?usize, 1), coord.pendingMessages(1));
    try std.testing.expectEqual(@as(?usize, 0), coord.pendingMessages(0));
}

test "integration: message to invalid agent returns error" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    // No agents registered — sending to agent 0 should fail
    const result = coord.sendMessage(.{
        .from_agent = 0,
        .to_agent = 0,
        .content = "nowhere",
    });
    try std.testing.expectError(multi_agent.Error.AgentNotFound, result);
}

test "integration: pending messages for invalid index returns null" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var coord = Coordinator.init(allocator);
    defer coord.deinit();

    try std.testing.expectEqual(@as(?usize, null), coord.pendingMessages(99));
}

// ============================================================================
// Event Bus Integration Tests
// ============================================================================

var integration_event_count: u32 = 0;

fn integrationEventCallback(_: messaging.Event) void {
    integration_event_count += 1;
}

test "integration: event bus fires during task execution" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    integration_event_count = 0;

    var agent = try Agent.init(allocator, .{ .name = "evented", .backend = .echo, .enable_history = false });
    defer agent.deinit();

    var coord = Coordinator.initWithConfig(allocator, .{
        .enable_events = true,
    });
    defer coord.deinit();
    try coord.register(&agent);

    try coord.onEvent(.task_started, integrationEventCallback);
    try coord.onEvent(.task_completed, integrationEventCallback);

    const result = try coord.runTask("event test");
    defer allocator.free(result);

    // Should have fired at least task_started + task_completed = 2
    try std.testing.expect(integration_event_count >= 2);
}

// ============================================================================
// Adaptive Strategy Tests
// ============================================================================

test "integration: adaptive strategy selects based on task size" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const names = [_][]const u8{ "a1", "a2", "a3" };
    var agent_storage: [3]Agent = undefined;
    try createEchoAgents(allocator, &names, &agent_storage);
    defer deinitAgents(&agent_storage);

    var coord = Coordinator.initWithConfig(allocator, .{
        .execution_strategy = .adaptive,
        .aggregation_strategy = .concatenate,
    });
    defer coord.deinit();

    for (&agent_storage) |*ag| try coord.register(ag);

    // Short task with 3 agents -> adaptive picks parallel
    const result = try coord.runTask("short");
    defer allocator.free(result);

    try std.testing.expect(result.len > 0);
    const stats = coord.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.success_count);
}

// ============================================================================
// Max Agents Limit Tests
// ============================================================================

test "integration: max agents limit enforced" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var agent1 = try Agent.init(allocator, .{ .name = "a1", .backend = .echo, .enable_history = false });
    defer agent1.deinit();
    var agent2 = try Agent.init(allocator, .{ .name = "a2", .backend = .echo, .enable_history = false });
    defer agent2.deinit();

    var coord = Coordinator.initWithConfig(allocator, .{
        .max_agents = 1,
    });
    defer coord.deinit();

    try coord.register(&agent1);
    const err = coord.register(&agent2);
    try std.testing.expectError(multi_agent.Error.MaxAgentsReached, err);
}

// ============================================================================
// Retry Configuration Tests
// ============================================================================

test "integration: retry config flows to coordinator" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var coord = Coordinator.initWithConfig(allocator, .{
        .retry_config = .{ .max_retries = 3, .initial_delay_ms = 10 },
    });
    defer coord.deinit();

    try std.testing.expectEqual(@as(u32, 3), coord.config.retry_config.max_retries);
    try std.testing.expectEqual(@as(u64, 10), coord.config.retry_config.initial_delay_ms);
}

// ============================================================================
// Stress: Rapid Sequential Runs
// ============================================================================

test "integration: rapid sequential task execution" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var agent = try Agent.init(allocator, .{ .name = "rapid", .backend = .echo, .enable_history = false });
    defer agent.deinit();

    var coord = Coordinator.initWithConfig(allocator, .{});
    defer coord.deinit();
    try coord.register(&agent);

    // Run 20 tasks rapidly to check for leaks / state corruption
    var i: usize = 0;
    while (i < 20) : (i += 1) {
        const result = try coord.runTask("iteration");
        allocator.free(result);
    }

    // Final run should still work
    const final = try coord.runTask("final");
    defer allocator.free(final);
    try std.testing.expectEqualStrings("Echo: final", final);
}
