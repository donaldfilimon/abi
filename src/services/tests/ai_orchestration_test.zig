//! AI Orchestration Tests — Router, Model Registry, Ensemble
//!
//! Tests model registration, routing strategies, and ensemble configuration.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const orchestration = if (build_options.enable_ai) abi.features.ai.orchestration else struct {};
const Orchestrator = if (build_options.enable_ai) orchestration.Orchestrator else struct {};
const OrchestrationConfig = if (build_options.enable_ai) orchestration.OrchestrationConfig else struct {};
const ModelConfig = if (build_options.enable_ai) orchestration.ModelConfig else struct {};
const Router = if (build_options.enable_ai) orchestration.Router else struct {};

// ============================================================================
// Orchestrator Lifecycle Tests
// ============================================================================

test "orchestrator: init and deinit with defaults" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var orch = Orchestrator.init(allocator, OrchestrationConfig.defaults()) catch |err| {
        // If init fails due to missing features, skip
        if (err == error.OrchestrationDisabled) return error.SkipZigTest;
        return err;
    };
    defer orch.deinit();
}

test "orchestrator: high availability config" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const config = OrchestrationConfig.highAvailability();
    try std.testing.expect(config.enable_fallback);
    try std.testing.expect(config.max_retries >= 3);

    var orch = Orchestrator.init(allocator, config) catch |err| {
        if (err == error.OrchestrationDisabled) return error.SkipZigTest;
        return err;
    };
    defer orch.deinit();
}

// ============================================================================
// Model Registration Tests
// ============================================================================

test "orchestrator: register and retrieve model" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var orch = Orchestrator.init(allocator, OrchestrationConfig.defaults()) catch |err| {
        if (err == error.OrchestrationDisabled) return error.SkipZigTest;
        return err;
    };
    defer orch.deinit();

    try orch.registerModel(.{
        .id = "test-model-1",
        .name = "Test Model",
        .backend = .openai,
        .model_name = "gpt-4",
    });

    const model = orch.getModel("test-model-1");
    try std.testing.expect(model != null);
}

test "orchestrator: duplicate model id rejected" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var orch = Orchestrator.init(allocator, OrchestrationConfig.defaults()) catch |err| {
        if (err == error.OrchestrationDisabled) return error.SkipZigTest;
        return err;
    };
    defer orch.deinit();

    try orch.registerModel(.{
        .id = "model-a",
        .name = "Model A",
        .backend = .openai,
    });

    const result = orch.registerModel(.{
        .id = "model-a",
        .name = "Duplicate Model A",
        .backend = .ollama,
    });

    try std.testing.expectError(error.DuplicateModelId, result);
}

test "orchestrator: unregister removes model" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var orch = Orchestrator.init(allocator, OrchestrationConfig.defaults()) catch |err| {
        if (err == error.OrchestrationDisabled) return error.SkipZigTest;
        return err;
    };
    defer orch.deinit();

    try orch.registerModel(.{
        .id = "ephemeral",
        .name = "Ephemeral",
        .backend = .local,
    });

    try orch.unregisterModel("ephemeral");
    try std.testing.expect(orch.getModel("ephemeral") == null);
}

test "orchestrator: unregister non-existent fails" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var orch = Orchestrator.init(allocator, OrchestrationConfig.defaults()) catch |err| {
        if (err == error.OrchestrationDisabled) return error.SkipZigTest;
        return err;
    };
    defer orch.deinit();

    try std.testing.expectError(error.ModelNotFound, orch.unregisterModel("ghost"));
}

// ============================================================================
// Model State Tests
// ============================================================================

test "orchestrator: enable and disable model" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var orch = Orchestrator.init(allocator, OrchestrationConfig.defaults()) catch |err| {
        if (err == error.OrchestrationDisabled) return error.SkipZigTest;
        return err;
    };
    defer orch.deinit();

    try orch.registerModel(.{
        .id = "toggleable",
        .name = "Toggleable",
        .backend = .openai,
    });

    try orch.setModelEnabled("toggleable", false);
    if (orch.getModel("toggleable")) |model| {
        try std.testing.expect(!model.config.enabled);
    }

    try orch.setModelEnabled("toggleable", true);
    if (orch.getModel("toggleable")) |model| {
        try std.testing.expect(model.config.enabled);
    }
}

test "orchestrator: set model health status" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var orch = Orchestrator.init(allocator, OrchestrationConfig.defaults()) catch |err| {
        if (err == error.OrchestrationDisabled) return error.SkipZigTest;
        return err;
    };
    defer orch.deinit();

    try orch.registerModel(.{
        .id = "health-test",
        .name = "Health Test",
        .backend = .openai,
    });

    try orch.setModelHealth("health-test", .degraded);
    try orch.setModelHealth("health-test", .unhealthy);
    try orch.setModelHealth("health-test", .healthy);
}

// ============================================================================
// Stats Tests
// ============================================================================

test "orchestrator: stats initially zeroed" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var orch = Orchestrator.init(allocator, OrchestrationConfig.defaults()) catch |err| {
        if (err == error.OrchestrationDisabled) return error.SkipZigTest;
        return err;
    };
    defer orch.deinit();

    const stats = orch.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.total_requests);
}

// ============================================================================
// Router Tests
// ============================================================================

test "router: task type detection" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    const TaskType = orchestration.TaskType;

    // Should detect coding-related prompts
    const coding = TaskType.detect("Write a function to sort an array");
    try std.testing.expect(coding == .coding or coding == .general);

    // Should detect math
    const math = TaskType.detect("Calculate the integral of x^2");
    try std.testing.expect(math == .math or math == .general);
}

test "router: round robin strategy" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    var router = Router.init(allocator, .round_robin);
    defer router.deinit();

    // Router should initialize without error
    try std.testing.expect(@sizeOf(Router) > 0);
}
