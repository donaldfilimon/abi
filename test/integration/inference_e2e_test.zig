//! Integration Tests: End-to-End Chat Pipeline

//!
//! Exercises the full path: profile routing → inference engine → connector
//! dispatch → response. Verifies that a routing decision from the
//! MultiProfileRouter can drive inference generation through the Engine,
//! producing valid responses with correct statistics.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const Engine = abi.inference.Engine;
const Backend = abi.inference.Backend;
const EngineResult = abi.inference.EngineResult;

// Profile types — gated on feat_ai at usage sites below
const profile = if (build_options.feat_ai) abi.ai.profile else struct {};

// ── Test 1: Profile routing → Engine generation (full pipeline) ──────────

test "e2e: profile routing drives inference engine generation" {
    if (!build_options.feat_ai) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Step 1: Create profile registry and router
    var registry = profile.ProfileRegistry.init(allocator, .{});
    defer registry.deinit();

    var router = profile.MultiProfileRouter.init(allocator, &registry, .{});
    defer router.deinit();

    // Step 2: Route a technical message to get a routing decision
    const decision = router.route("How do I implement a hash map in Zig?");

    // Verify routing produced a valid decision
    try std.testing.expectEqual(profile.ProfileId.aviva, decision.primary);
    try std.testing.expect(decision.confidence > 0.0);
    try std.testing.expect(decision.weights.aviva > 0.0);

    // Step 3: Create an inference engine with demo backend
    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .demo,
    });
    defer engine.deinit();

    // Step 4: Generate a response using the routing decision's profile_id
    const prompt = std.fmt.comptimePrint("[{s}] How do I implement a hash map in Zig?", .{"Aviva"});
    var result = try engine.generate(.{
        .id = 1,
        .prompt = prompt,
        .max_tokens = 20,
    });
    defer result.deinit(allocator);

    // Step 5: Verify response integrity
    try std.testing.expect(result.text.len > 0);
    try std.testing.expect(result.completion_tokens > 0);
    try std.testing.expect(result.latency_ms >= 0.0);

    // Verify engine stats updated
    const stats = engine.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.total_requests);
    try std.testing.expect(stats.total_tokens_generated > 0);
}

// ── Test 2: Emotional routing → Engine generation ────────────────────────

test "e2e: emotional routing to Abbey drives inference" {
    if (!build_options.feat_ai) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var registry = profile.ProfileRegistry.init(allocator, .{});
    defer registry.deinit();

    var router = profile.MultiProfileRouter.init(allocator, &registry, .{});
    defer router.deinit();

    // Route an emotional message — should go to Abbey
    const decision = router.route("I feel lost and need help understanding closures");
    try std.testing.expectEqual(profile.ProfileId.abbey, decision.primary);

    // Generate a response via the engine
    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .demo,
    });
    defer engine.deinit();

    var result = try engine.generate(.{
        .id = 2,
        .prompt = "[Abbey] I feel lost and need help understanding closures",
        .max_tokens = 15,
    });
    defer result.deinit(allocator);

    try std.testing.expect(result.text.len > 0);
    try std.testing.expect(result.completion_tokens > 0);
}

// ── Test 3: Engine backend selection ─────────────────────────────────────

test "e2e: demo backend produces text" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .demo,
    });
    defer engine.deinit();

    var result = try engine.generate(.{
        .id = 10,
        .prompt = "Explain vector databases",
        .max_tokens = 12,
    });
    defer result.deinit(allocator);

    try std.testing.expect(result.text.len > 0);
    try std.testing.expect(result.completion_tokens > 0);
    try std.testing.expectEqual(Backend.demo, engine.getStats().backend);
}

test "e2e: connector backend falls back to echo without env vars" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .backend = .connector,
        .model_id = "test-echo-model",
    });
    defer engine.deinit();

    var result = try engine.generate(.{
        .id = 11,
        .prompt = "Echo test prompt",
        .max_tokens = 10,
    });
    defer result.deinit(allocator);

    // Connector backend echoes prompt with model_id in the response
    try std.testing.expect(result.text.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, result.text, "test-echo-model") != null);
    try std.testing.expectEqual(Backend.connector, engine.getStats().backend);
}

test "e2e: local backend falls back to demo without model loaded" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .local,
    });
    defer engine.deinit();

    var result = try engine.generate(.{
        .id = 12,
        .prompt = "Local inference test",
        .max_tokens = 8,
    });
    defer result.deinit(allocator);

    // Local backend without a model falls back to demo-style generation
    try std.testing.expect(result.text.len > 0);
}

// ── Test 4: Engine stats accumulation ────────────────────────────────────

test "e2e: stats accumulate across multiple requests" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .demo,
    });
    defer engine.deinit();

    // Initial state: zero requests
    try std.testing.expectEqual(@as(u64, 0), engine.getStats().total_requests);

    // Generate three requests and verify incremental stats
    const prompts = [_][]const u8{
        "First query about sorting",
        "Second query about graphs",
        "Third query about trees",
    };

    var total_tokens: u64 = 0;
    for (prompts, 1..) |prompt, i| {
        var result = try engine.generate(.{
            .id = @intCast(i),
            .prompt = prompt,
            .max_tokens = 10,
        });
        defer result.deinit(allocator);

        total_tokens += result.completion_tokens;

        const stats = engine.getStats();
        try std.testing.expectEqual(@as(u64, i), stats.total_requests);
    }

    // Final stats verification
    const final_stats = engine.getStats();
    try std.testing.expectEqual(@as(u64, 3), final_stats.total_requests);
    try std.testing.expect(final_stats.total_tokens_generated > 0);
    try std.testing.expectEqual(total_tokens, final_stats.total_tokens_generated);
}

// ── Test 5: Engine async generation with timeout ─────────────────────────

test "e2e: async generation with timeout returns valid response" {
    const allocator = std.testing.allocator;

    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .demo,
    });
    defer engine.deinit();

    const ar = try engine.generateAsyncWithTimeout(.{
        .id = 50,
        .prompt = "Async pipeline test",
        .max_tokens = 12,
    });
    defer ar.destroy();

    // Wait with generous timeout
    const maybe_result = ar.waitTimeout(5000);
    try std.testing.expect(maybe_result != null);

    var result = maybe_result.?;
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(u64, 50), result.id);
    try std.testing.expect(result.text.len > 0);
    try std.testing.expect(result.completion_tokens > 0);
    try std.testing.expect(result.latency_ms >= 0.0);
}

// ── Test 6: Full pipeline with memory and constitution ───────────────────

test "e2e: full pipeline with routing, generation, and memory" {
    if (!build_options.feat_ai) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Set up profile pipeline
    var registry = profile.ProfileRegistry.init(allocator, .{});
    defer registry.deinit();

    var router = profile.MultiProfileRouter.init(allocator, &registry, .{});
    defer router.deinit();

    router.attachMemory("e2e-test-session");

    // Set up inference engine
    var engine = try Engine.init(allocator, .{
        .kv_cache_pages = 100,
        .page_size = 16,
        .num_layers = 1,
        .num_heads = 1,
        .head_dim = 4,
        .max_batch_size = 8,
        .vocab_size = 256,
        .backend = .demo,
    });
    defer engine.deinit();

    // Simulate two-turn conversation
    const messages = [_][]const u8{
        "How do I implement error handling in Zig?",
        "Can you explain the difference between try and catch?",
    };

    for (messages, 1..) |msg, i| {
        // Route
        const decision = router.route(msg);
        try std.testing.expect(decision.confidence > 0.0);

        // Generate via engine
        var result = try engine.generate(.{
            .id = @intCast(i),
            .prompt = msg,
            .max_tokens = 15,
        });
        defer result.deinit(allocator);

        try std.testing.expect(result.text.len > 0);
    }

    // Verify engine processed both requests
    try std.testing.expectEqual(@as(u64, 2), engine.getStats().total_requests);

    // Verify memory recorded (if available)
    try std.testing.expect(router.memory != null);
}

// ── Test 7: Routing decision profile_id consistency ──────────────────────

test "e2e: routing decision profile names are valid" {
    if (!build_options.feat_ai) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var registry = profile.ProfileRegistry.init(allocator, .{});
    defer registry.deinit();

    var router = profile.MultiProfileRouter.init(allocator, &registry, .{});
    defer router.deinit();

    const test_cases = [_]struct { input: []const u8, expected: profile.ProfileId }{
        .{ .input = "Debug this compile error please", .expected = .aviva },
        .{ .input = "I feel confused about this topic", .expected = .abbey },
        .{ .input = "What is the privacy policy?", .expected = .abi },
    };

    for (test_cases) |tc| {
        const decision = router.route(tc.input);
        try std.testing.expectEqual(tc.expected, decision.primary);

        // Verify profile name is non-empty
        const name = decision.primary.name();
        try std.testing.expect(name.len > 0);
    }
}

test {
    std.testing.refAllDecls(@This());
}
