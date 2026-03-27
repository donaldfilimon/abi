//! Focused agents unit-test root that keeps the module path anchored at `src/`.

const std = @import("std");
const agents = @import("features/ai/agents/mod.zig");
const agents_tests = @import("features/ai/agents/tests.zig");
const agent_types = @import("features/ai/agents/types.zig");

test {
    std.testing.refAllDecls(agents_tests);
}

// ── AgentConfig validation tests ───────────────────────────────────────

test "agent config validate rejects empty name" {
    const cfg = agent_types.AgentConfig{ .name = "" };
    try std.testing.expectError(agent_types.AgentError.InvalidConfiguration, cfg.validate());
}

test "agent config validate rejects zero max_tokens" {
    const cfg = agent_types.AgentConfig{ .name = "test", .max_tokens = 0 };
    try std.testing.expectError(agent_types.AgentError.InvalidConfiguration, cfg.validate());
}

test "agent config validate rejects max_tokens over limit" {
    const cfg = agent_types.AgentConfig{ .name = "test", .max_tokens = agent_types.MAX_TOKENS_LIMIT + 1 };
    try std.testing.expectError(agent_types.AgentError.InvalidConfiguration, cfg.validate());
}

test "agent config validate accepts valid defaults" {
    const cfg = agent_types.AgentConfig{ .name = "valid-agent" };
    try cfg.validate();
}

test "agent config validate boundary temperatures" {
    // Exact boundary values should be valid
    const cfg_min = agent_types.AgentConfig{ .name = "t", .temperature = agent_types.MIN_TEMPERATURE };
    try cfg_min.validate();
    const cfg_max = agent_types.AgentConfig{ .name = "t", .temperature = agent_types.MAX_TEMPERATURE };
    try cfg_max.validate();
}

// ── BackendMetrics tests ───────────────────────────────────────────────

test "backend metrics initial state" {
    const m = agent_types.BackendMetrics{};
    try std.testing.expectEqual(@as(u32, 0), m.totalCalls());
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), m.successRate(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), m.avgLatencyMs(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), m.avgQuality(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), m.score(), 1e-5);
}

test "backend metrics record success and failure" {
    var m = agent_types.BackendMetrics{};
    m.record(true, 100, 0.9);
    m.record(true, 200, 0.8);
    m.record(false, 50, 0.0);

    try std.testing.expectEqual(@as(u32, 3), m.totalCalls());
    try std.testing.expectEqual(@as(u32, 2), m.success_count);
    try std.testing.expectEqual(@as(u32, 1), m.failure_count);
    // Success rate: 2/3
    try std.testing.expect(m.successRate() > 0.6);
    try std.testing.expect(m.successRate() < 0.7);
    // Average latency: (100+200+50)/3
    try std.testing.expect(m.avgLatencyMs() > 100);
}

// ── WorkloadType tests ─────────────────────────────────────────────────

test "workload type gpu intensive classification" {
    try std.testing.expect(agent_types.WorkloadType.training.gpuIntensive());
    try std.testing.expect(agent_types.WorkloadType.fine_tuning.gpuIntensive());
    try std.testing.expect(!agent_types.WorkloadType.inference.gpuIntensive());
    try std.testing.expect(!agent_types.WorkloadType.embedding.gpuIntensive());
}

test "workload type memory intensive classification" {
    try std.testing.expect(agent_types.WorkloadType.training.memoryIntensive());
    try std.testing.expect(agent_types.WorkloadType.batch_inference.memoryIntensive());
    try std.testing.expect(!agent_types.WorkloadType.inference.memoryIntensive());
}

test "workload type name strings" {
    try std.testing.expectEqualStrings("Inference", agent_types.WorkloadType.inference.name());
    try std.testing.expectEqualStrings("Training", agent_types.WorkloadType.training.name());
    try std.testing.expectEqualStrings("Embedding", agent_types.WorkloadType.embedding.name());
}

// ── Priority tests ─────────────────────────────────────────────────────

test "priority weights are ordered correctly" {
    try std.testing.expect(agent_types.Priority.low.weight() < agent_types.Priority.normal.weight());
    try std.testing.expect(agent_types.Priority.normal.weight() < agent_types.Priority.high.weight());
    try std.testing.expect(agent_types.Priority.high.weight() < agent_types.Priority.critical.weight());
}

// ── ToolRegistry standalone tests ──────────────────────────────────────

test "tool registry init and count" {
    const allocator = std.testing.allocator;
    var registry = agent_types.ToolRegistry.init(allocator);
    defer registry.deinit();

    try std.testing.expectEqual(@as(usize, 0), registry.count());
    try std.testing.expect(!registry.contains("any_tool"));
}

// ── GpuAgentStats tests ───────────────────────────────────────────────

test "gpu agent stats initial success rate is 1.0" {
    const stats = agent_types.GpuAgentStats{};
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), stats.successRate(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), stats.gpuUtilizationRate(), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), stats.avgTokensPerRequest(), 1e-5);
}

test "gpu agent stats update confidence" {
    var stats = agent_types.GpuAgentStats{};
    stats.gpu_accelerated = 1;
    stats.updateConfidence(0.8);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), stats.avg_scheduling_confidence, 1e-5);
}

test "gpu agent stats update latency" {
    var stats = agent_types.GpuAgentStats{};
    stats.total_requests = 1;
    stats.updateLatency(100);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), stats.avg_latency_ms, 1e-3);
}
