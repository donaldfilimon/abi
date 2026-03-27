const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

test "orchestration public surface compiles from abi" {
    _ = abi.ai.orchestration.Router;
    _ = abi.ai.orchestration.RoutingStrategy.task_based;
    _ = abi.ai.orchestration.TaskType.coding;
    _ = abi.ai.orchestration.RouteResult;
    _ = abi.ai.orchestration.Ensemble;
    _ = abi.ai.orchestration.EnsembleMethod.voting;
    _ = abi.ai.orchestration.EnsembleResult;
    _ = abi.ai.orchestration.FallbackManager;
    _ = abi.ai.orchestration.FallbackPolicy.retry_then_fallback;
    _ = abi.ai.orchestration.HealthStatus.healthy;
    _ = abi.ai.orchestration.Orchestrator;
    _ = abi.ai.orchestration.OrchestrationConfig.defaults();
    _ = abi.ai.orchestration.ModelBackend.openai;
    _ = abi.ai.orchestration.Capability.coding;
    _ = abi.ai.orchestration.ModelConfig{ .id = "surface-check" };
    _ = abi.ai.orchestration.ModelEntry;
    _ = abi.ai.orchestration.OrchestratorStats{};

    try std.testing.expectEqual(build_options.feat_ai, abi.ai.orchestration.isEnabled());
}

test "orchestration public routing remains stable for external consumers" {
    if (!build_options.feat_ai) return error.SkipZigTest;

    var orch = try abi.ai.orchestration.Orchestrator.init(std.testing.allocator, .{
        .strategy = .task_based,
    });
    defer orch.deinit();

    try orch.registerModel(.{
        .id = "coder",
        .backend = .openai,
        .capabilities = &.{.coding},
    });
    try orch.registerModel(.{
        .id = "generalist",
        .backend = .anthropic,
    });

    const route = try orch.route("implement a stack", .coding);
    try std.testing.expectEqualStrings("coder", route.model_id);

    const ids = try orch.listModels(std.testing.allocator);
    defer std.testing.allocator.free(ids);

    try std.testing.expectEqual(@as(usize, 2), ids.len);

    const stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 2), stats.total_models);
    try std.testing.expectEqual(@as(u32, 2), stats.available_models);
}
