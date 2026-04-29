const std = @import("std");
const orchestration = @import("mod.zig");
const execution = @import("orchestrator/execution.zig");
const shared_types = @import("types.zig");

fn successDispatch(
    allocator: std.mem.Allocator,
    model: *shared_types.ModelEntry,
    prompt: []const u8,
) shared_types.OrchestrationError![]u8 {
    return std.fmt.allocPrint(allocator, "{s}:{s}", .{ model.config.id, prompt }) catch
        return shared_types.OrchestrationError.OutOfMemory;
}

fn primaryFailsDispatch(
    allocator: std.mem.Allocator,
    model: *shared_types.ModelEntry,
    prompt: []const u8,
) shared_types.OrchestrationError![]u8 {
    _ = prompt;
    if (std.mem.eql(u8, model.config.id, "primary")) {
        return shared_types.OrchestrationError.AllModelsFailed;
    }

    return std.fmt.allocPrint(allocator, "fallback:{s}", .{model.config.id}) catch
        return shared_types.OrchestrationError.OutOfMemory;
}

test "orchestration shared type exports stay aligned" {
    try std.testing.expectEqual(shared_types.RoutingStrategy.round_robin, orchestration.RoutingStrategy.round_robin);
    try std.testing.expectEqual(shared_types.ModelBackend.openai, orchestration.ModelBackend.openai);
    try std.testing.expectEqual(shared_types.EnsembleMethod.voting, orchestration.EnsembleMethod.voting);
    try std.testing.expectEqual(shared_types.FallbackPolicy.fail_fast, orchestration.FallbackPolicy.fail_fast);
}

test "orchestrator init and deinit" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{});
    defer orch.deinit();

    const stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.total_models);
    try std.testing.expectEqual(@as(u32, 0), stats.available_models);
}

test "orchestrator register and unregister models" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{});
    defer orch.deinit();

    try orch.registerModel(.{ .id = "model-a" });
    try orch.registerModel(.{ .id = "model-b" });

    try std.testing.expectError(
        orchestration.OrchestrationError.DuplicateModelId,
        orch.registerModel(.{ .id = "model-a" }),
    );

    try orch.unregisterModel("model-b");

    const stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.total_models);
    try std.testing.expectEqual(@as(u32, 1), stats.available_models);
}

test "orchestrator round robin routing" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{
        .strategy = .round_robin,
    });
    defer orch.deinit();

    try orch.registerModel(.{ .id = "model-a" });
    try orch.registerModel(.{ .id = "model-b" });

    const result1 = try orch.route("test", null);
    const result2 = try orch.route("test", null);

    try std.testing.expect(!std.mem.eql(u8, result1.model_id, result2.model_id));
}

test "orchestrator task based selection" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{
        .strategy = .task_based,
    });
    defer orch.deinit();

    try orch.registerModel(.{
        .id = "coder",
        .capabilities = &.{.coding},
    });
    try orch.registerModel(.{
        .id = "writer",
        .capabilities = &.{.creative},
    });

    const coding_route = try orch.route("implement a queue", .coding);
    try std.testing.expectEqualStrings("coder", coding_route.model_id);

    const creative_route = try orch.route("write a poem", .creative);
    try std.testing.expectEqualStrings("writer", creative_route.model_id);
}

test "orchestrator enable and disable model" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{});
    defer orch.deinit();

    try orch.registerModel(.{ .id = "toggle-model" });
    try orch.setModelEnabled("toggle-model", false);

    const stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.available_models);
    try std.testing.expectError(
        orchestration.OrchestrationError.NoModelsAvailable,
        orch.route("test", null),
    );
}

test "orchestrator health transitions" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{});
    defer orch.deinit();

    try orch.registerModel(.{ .id = "health-model" });

    try orch.setModelHealth("health-model", .unhealthy);
    var stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.available_models);

    try orch.setModelHealth("health-model", .healthy);
    stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 1), stats.available_models);
}

test "orchestrator stats aggregation" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{});
    defer orch.deinit();

    try orch.registerModel(.{ .id = "stats-a" });
    try orch.registerModel(.{ .id = "stats-b" });

    const model_a = orch.getModel("stats-a").?;
    model_a.total_requests = 3;
    model_a.total_failures = 1;
    model_a.active_requests = 1;

    const model_b = orch.getModel("stats-b").?;
    model_b.total_requests = 2;
    model_b.active_requests = 1;

    const stats = orch.getStats();
    try std.testing.expectEqual(@as(u32, 2), stats.total_models);
    try std.testing.expectEqual(@as(u64, 5), stats.total_requests);
    try std.testing.expectEqual(@as(u64, 1), stats.total_failures);
    try std.testing.expectEqual(@as(u32, 2), stats.active_requests);
    try std.testing.expect(@abs(stats.successRate() - 0.8) < 0.0001);
}

test "orchestrator single execution updates runtime counters" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{});
    defer orch.deinit();

    try orch.registerModel(.{ .id = "model-a" });

    const response = try execution.executeSingleWithDispatch(
        &orch,
        "model-a",
        "hello",
        std.testing.allocator,
        successDispatch,
    );
    defer std.testing.allocator.free(response);

    try std.testing.expectEqualStrings("model-a:hello", response);

    const model = orch.getModel("model-a").?;
    try std.testing.expectEqual(@as(u64, 1), model.total_requests);
    try std.testing.expectEqual(@as(u64, 0), model.total_failures);
    try std.testing.expectEqual(@as(u32, 0), model.active_requests);
    try std.testing.expectEqual(@as(u32, 0), model.consecutive_failures);
}

test "orchestrator fallback continues after primary failure" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{
        .enable_fallback = true,
        .max_retries = 3,
    });
    defer orch.deinit();

    try orch.registerModel(.{
        .id = "primary",
        .priority = 1,
    });
    try orch.registerModel(.{
        .id = "secondary",
        .priority = 2,
    });

    const response = try execution.executeWithDispatch(
        &orch,
        .{
            .model_id = "primary",
            .model_name = "primary",
            .backend = .openai,
            .prompt = "hello",
        },
        std.testing.allocator,
        primaryFailsDispatch,
    );
    defer std.testing.allocator.free(response);

    try std.testing.expectEqualStrings("fallback:secondary", response);
    try std.testing.expectEqual(@as(u64, 1), orch.getModel("primary").?.total_failures);
    try std.testing.expectEqual(@as(u64, 1), orch.getModel("secondary").?.total_requests);
}

test "orchestrator ensemble minimum model enforcement" {
    if (!orchestration.isEnabled()) return error.SkipZigTest;

    var orch = try orchestration.Orchestrator.init(std.testing.allocator, .{
        .enable_ensemble = true,
        .min_ensemble_models = 2,
    });
    defer orch.deinit();

    try orch.registerModel(.{ .id = "solo" });

    try std.testing.expectError(
        orchestration.OrchestrationError.InsufficientModelsForEnsemble,
        orch.executeEnsemble("hello", null, std.testing.allocator),
    );
}

test {
    std.testing.refAllDecls(@This());
}
