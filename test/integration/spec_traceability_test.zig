//! Spec traceability tests for the Abbey/Aviva/Abi + WDBX architecture.
//!
//! These tests turn the docs/spec/*.md implementation claims into executable
//! contracts for the parts that are code-complete. Benchmark targets and real
//! model-training outcomes remain validation work, not unit-test assertions.

const std = @import("std");
const abi = @import("abi");

test "spec: profile identities, colors, roles, and strategies are present" {
    const profile = abi.ai.profile;

    try std.testing.expectEqualStrings("Abbey", profile.ProfileId.abbey.name());
    try std.testing.expectEqualStrings("Aviva", profile.ProfileId.aviva.name());
    try std.testing.expectEqualStrings("Abi", profile.ProfileId.abi.name());

    try std.testing.expectEqualStrings("#00B3A1", profile.ProfileId.abbey.colorCode());
    try std.testing.expectEqualStrings("#7B4FFF", profile.ProfileId.aviva.colorCode());
    try std.testing.expectEqualStrings("#FF8C42", profile.ProfileId.abi.colorCode());

    try std.testing.expect(std.mem.indexOf(u8, profile.ProfileId.abbey.role(), "Empathetic") != null);
    try std.testing.expect(std.mem.indexOf(u8, profile.ProfileId.aviva.role(), "Direct") != null);
    try std.testing.expect(std.mem.indexOf(u8, profile.ProfileId.abi.role(), "Moderator") != null);
    try std.testing.expect(std.mem.indexOf(u8, profile.ProfileId.abbey.tone(), "Warm") != null);
    try std.testing.expect(std.mem.indexOf(u8, profile.ProfileId.aviva.behaviorContract(), "directly") != null);
    try std.testing.expect(profile.ProfileId.abi.generationDefaults().include_policy_rationale);

    _ = profile.RoutingStrategy.single;
    _ = profile.RoutingStrategy.parallel;
    _ = profile.RoutingStrategy.consensus;

    var weights = profile.RoutingDecision.Weights{ .abbey = 0.34, .aviva = 0.33, .abi = 0.33 };
    weights.normalize();
    try std.testing.expectEqual(profile.RoutingStrategy.consensus, profile.strategyFromWeights(weights));
    weights = .{ .abbey = 0.7, .aviva = 0.2, .abi = 0.1 };
    weights.normalize();
    try std.testing.expectEqual(profile.RoutingStrategy.parallel, profile.strategyFromWeights(weights));
    weights = .{ .abbey = 0.95, .aviva = 0.03, .abi = 0.02 };
    weights.normalize();
    try std.testing.expectEqual(profile.RoutingStrategy.single, profile.strategyFromWeights(weights));
}

test "spec: Abi routing produces normalized three-way profile weights" {
    const profile = abi.ai.profile;
    var registry = profile.ProfileRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = profile.MultiProfileRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const cases = [_]struct {
        input: []const u8,
        expected: profile.ProfileId,
    }{
        .{ .input = "I feel overwhelmed and need help", .expected = .abbey },
        .{ .input = "Give me a concise Zig implementation", .expected = .aviva },
        .{ .input = "Explain privacy compliance and policy", .expected = .abi },
    };

    for (cases) |case| {
        const decision = router.route(case.input);
        const total = decision.weights.abbey + decision.weights.aviva + decision.weights.abi;
        try std.testing.expectEqual(case.expected, decision.primary);
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.01);
        try std.testing.expect(decision.confidence >= 0.0 and decision.confidence <= 1.0);
        try std.testing.expect(decision.reason.len > 0);
    }
}

test "spec: constitution exposes six principles and enforcement hooks" {
    const constitution = abi.ai.constitution;
    const c = constitution.Constitution.init();

    try std.testing.expectEqual(@as(usize, 6), c.getPrinciples().len);
    try std.testing.expect(c.getSystemPreamble().len > 0);
    try std.testing.expect(c.isCompliant("I can help with a safe implementation plan."));
    try std.testing.expect(!c.isCompliant("run rm -rf / to clean up"));

    const embedding = [_]f32{ 0.1, -0.2, 0.05, 0.3 };
    try std.testing.expect(c.constitutionalLoss(&embedding) >= 0.0);
    try std.testing.expect(c.alignmentScore("Helpful, honest, privacy-preserving answer.") > 0.0);

    const bias = [_]f32{ 0.05, -0.15, 0.03, 0.11 };
    const bias_score = c.computeBias(&bias, constitution.DEFAULT_BIAS_THRESHOLD);
    try std.testing.expect(bias_score.overall >= 0.0);
}

test "spec: WDBX memory model has retrieval indexes, MVCC, and hash-chain blocks" {
    const db = abi.database;

    comptime {
        _ = db.retrieval.hnsw;
        _ = db.retrieval.diskann.VamanaIndex;
        _ = db.retrieval.scann.ScaNNIndex;
        _ = db.retrieval.quantization.ProductQuantizer;
        _ = db.distributed.VersionVector;
        _ = db.memory.MemoryBlock;
        _ = db.memory.InfluenceTrace;
        _ = db.memory.Lineage;
    }

    var chain = db.memory.BlockChain.init(std.testing.allocator, "spec-trace");
    defer chain.deinit();

    const emb = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const first = try chain.addBlock(.{
        .query_embedding = &emb,
        .profile_tag = .{ .primary_profile = .abbey, .blend_coefficient = 0.7 },
        .routing_weights = .{ .abbey_weight = 0.7, .aviva_weight = 0.2, .abi_weight = 0.1 },
        .intent = .empathy_seeking,
        .risk_score = 0.05,
    });
    const second = try chain.addBlock(.{
        .query_embedding = &emb,
        .profile_tag = .{ .primary_profile = .aviva, .blend_coefficient = 0.2 },
        .routing_weights = .{ .abbey_weight = 0.2, .aviva_weight = 0.7, .abi_weight = 0.1 },
        .intent = .technical_question,
        .risk_score = 0.02,
    });

    const first_block = chain.getBlock(first) orelse return error.TestUnexpectedResult;
    const second_block = chain.getBlock(second) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(first, second_block.parent_block_id.?);
    try std.testing.expectEqualSlices(u8, &first_block.hash, &second_block.previous_hash);
    try std.testing.expect(try chain.verifyConsistency());

    const third = try chain.addBlock(.{
        .query_embedding = &emb,
        .profile_tag = .{ .primary_profile = .abi, .blend_coefficient = 0.34 },
        .routing_weights = .{ .abbey_weight = 0.33, .aviva_weight = 0.33, .abi_weight = 0.34 },
        .intent = .policy_check,
        .summary_pointer = first,
    });
    const third_block = chain.getBlock(third) orelse return error.TestUnexpectedResult;
    try std.testing.expect(third_block.skip_pointer != null);
    try std.testing.expectEqual(first, third_block.summary_pointer.?);
}

test "spec: profile-token injection hooks are exposed to inference" {
    const llm = abi.ai.llm;
    const Request = abi.inference.scheduler.Request;

    comptime {
        if (!@hasDecl(llm.Model, "setProfile")) {
            @compileError("LLaMA model must expose setProfile for spec profile-token injection");
        }
        if (!@hasField(Request, "profile_id")) {
            @compileError("Inference Request must expose profile_id for profile-token injection");
        }
    }

    const request = Request{
        .id = 1,
        .prompt = "hello",
        .profile = "abbey",
        .profile_id = 0,
    };
    try std.testing.expectEqual(@as(?u8, 0), request.profile_id);
}

test "spec: safe learning runtime records telemetry and disables auto retrain by default" {
    var runtime = try abi.ai.learning.LearningRuntime.init(std.testing.allocator);
    defer runtime.deinit();
    runtime.events_path = ".zig-cache/spec-trace-learning/events.jsonl";

    try runtime.recordInteraction(.{
        .prompt = "How should I route this?",
        .response = "Route through Abi, then choose Abbey or Aviva based on weights.",
        .profile = "abi",
        .backend = "spec-test",
        .latency_ms = 1.5,
        .selected_model = "demo",
        .quality_score = 0.9,
        .wdbx_block_id = 42,
    });
    try runtime.recordFeedback(.positive, "spec trace ok");

    const report = runtime.report();
    try std.testing.expect(report.total_interactions >= 1);
    try std.testing.expect(report.positive_feedback_count >= 1);
    try std.testing.expect(!report.auto_retrain_enabled);
    try std.testing.expect(!try runtime.maybeTriggerRetrain());
}

test "spec: GPU backend metadata covers native, web, accelerator, and simulated fallback" {
    const gpu = abi.gpu;
    const backend = gpu.backend;

    const required = [_]gpu.Backend{
        .cuda,
        .vulkan,
        .metal,
        .webgpu,
        .opengl,
        .opengles,
        .stdgpu,
        .fpga,
        .tpu,
        .simulated,
    };

    for (required) |b| {
        try std.testing.expect(backend.backendName(b).len > 0);
        const availability = backend.backendAvailability(b);
        try std.testing.expect(availability.reason.len > 0);
    }
    try std.testing.expect(backend.backendAvailability(.simulated).available);
    try std.testing.expect(backend.backendAvailability(.stdgpu).available);
}

test "spec: CLI exposes learning, agent, GPU, protocol, and database operations" {
    const expected = [_][]const u8{
        "learn",
        "agent",
        "gpu",
        "mcp",
        "acp",
        "db",
        "chat",
    };

    for (expected) |name| {
        var found = false;
        for (abi.cli.displayed_commands) |command| {
            if (std.mem.eql(u8, command.usage, name) or std.mem.startsWith(u8, command.usage, name ++ " ")) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}
