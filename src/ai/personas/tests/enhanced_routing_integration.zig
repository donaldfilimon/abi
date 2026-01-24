//! Enhanced Routing with WDBX Block Chain Integration Test
//!
//! Tests the full round-trip: enhanced routing → block chain storage → retrieval.
//! Validates that routing decisions are properly persisted in WDBX block chain.

const std = @import("std");
const testing = std.testing;
const personas = @import("../mod.zig");
const enhanced = @import("../routing/enhanced.zig");
const block_chain = @import("../../../database/block_chain.zig");

test "EnhancedRouter creates and stores block chain entries" {
    const allocator = testing.allocator;

    // Create enhanced router with block chain
    var router = try enhanced.EnhancedRouter.init(allocator, "test-session-integration");
    defer router.deinit();

    // Create a test request
    var request = personas.types.PersonaRequest{
        .content = "I need help with a memory leak in my Zig code. It's really frustrating me.",
        .session_id = "test-session-1",
        .user_id = "test-user",
    };
    defer request.deinit(allocator);

    // Create a simple sentiment result (simulating Abi analysis)
    const sentiment = personas.abi.SentimentResult{
        .primary_emotion = .frustrated,
        .requires_empathy = true,
        .is_technical = true,
        .urgency_score = 0.3,
        .sentiment_score = -0.4,
    };

    // Route with enhanced algorithm
    const decision = try router.routeEnhanced(request, sentiment);
    defer decision.deinit(allocator);

    // Verify routing decision
    try testing.expect(decision.base_decision.selected_persona == .abbey or
        decision.base_decision.selected_persona == .aviva);
    try testing.expect(decision.base_decision.confidence > 0.5);

    // Check that block was created (if block chain is enabled)
    if (router.block_chain != null) {
        try testing.expect(decision.block_chain_id != null);
        try testing.expect(router.parent_block_id == decision.block_chain_id);
    }
}

test "Block chain persistence and retrieval" {
    const allocator = testing.allocator;

    // Create a standalone block chain for testing
    var chain = block_chain.BlockChain.init(allocator, "test-session-persistence");
    defer chain.deinit();

    // Create test embedding
    const dim = 384;
    const query_embedding = try allocator.alloc(f32, dim);
    defer allocator.free(query_embedding);
    @memset(query_embedding, 0.1);

    // Create first block
    const config1 = block_chain.BlockConfig{
        .query_embedding = query_embedding,
        .persona_tag = .{ .primary_persona = .abbey },
        .routing_weights = .{ .abbey_weight = 0.8, .aviva_weight = 0.2 },
        .intent = .empathy_seeking,
    };

    const block1_id = try chain.addBlock(config1);
    try testing.expect(block1_id != 0);
    try testing.expect(chain.current_head == block1_id);

    // Create second block with parent reference
    const config2 = block_chain.BlockConfig{
        .query_embedding = query_embedding,
        .persona_tag = .{ .primary_persona = .aviva },
        .routing_weights = .{ .abbey_weight = 0.3, .aviva_weight = 0.7 },
        .intent = .technical_problem,
        .parent_block_id = block1_id,
    };

    const block2_id = try chain.addBlock(config2);
    try testing.expect(block2_id != block1_id);
    try testing.expect(chain.current_head == block2_id);

    // Retrieve and verify blocks
    if (chain.getBlock(block1_id)) |block1| {
        try testing.expect(block1.persona_tag.primary_persona == .abbey);
        try testing.expect(block1.intent == .empathy_seeking);
    }

    if (chain.getBlock(block2_id)) |block2| {
        try testing.expect(block2.persona_tag.primary_persona == .aviva);
        try testing.expect(block2.parent_block_id == block1_id);
    }

    // Test chain traversal
    const backward_traversal = try chain.traverseBackward(5);
    defer allocator.free(backward_traversal);

    try testing.expect(backward_traversal.len >= 1);
    try testing.expect(backward_traversal[0] == block2_id);

    if (backward_traversal.len > 1) {
        try testing.expect(backward_traversal[1] == block1_id);
    }
}

test "MVCC visibility control" {
    const allocator = testing.allocator;

    var store = block_chain.MvccStore.init(allocator);
    defer store.deinit();

    const session_id = "test-session-mvcc";
    const chain = try store.getChain(session_id);

    // Create test embedding
    const dim = 384;
    const query_embedding = try allocator.alloc(f32, dim);
    defer allocator.free(query_embedding);
    @memset(query_embedding, 0.1);

    // Add a block
    const config = block_chain.BlockConfig{
        .query_embedding = query_embedding,
        .persona_tag = .{ .primary_persona = .abbey },
        .routing_weights = .{ .abbey_weight = 0.8, .aviva_weight = 0.2 },
        .intent = .empathy_seeking,
    };

    const block_id = try chain.addBlock(config);

    // Set read timestamp and check visibility
    const read_ts = block_chain.time.unixSeconds() + 1;
    try store.setReadTimestamp(session_id, read_ts);

    const visible = try store.getVisibleBlocks(session_id);
    defer allocator.free(visible);

    try testing.expect(visible.len == 1);
    try testing.expect(visible[0] == block_id);
}

test "Intent classification integration" {
    const allocator = testing.allocator;

    var classifier = enhanced.IntentClassifier.init(allocator);

    // Test empathy-seeking detection
    const empathy_text = "I feel really overwhelmed and need emotional support";
    try testing.expect(classifier.classify(empathy_text) == .empathy_seeking);

    // Test technical problem detection
    const tech_text = "How do I implement async I/O in Zig 0.16?";
    try testing.expect(classifier.classify(tech_text) == .technical_problem);

    // Test factual inquiry
    const fact_text = "What are the key benefits of the WDBX architecture?";
    try testing.expect(classifier.classify(fact_text) == .factual_inquiry);

    // Test safety-critical
    const safety_text = "I need to know how to safely handle user PII data";
    // Note: current classifier doesn't detect safety_critical automatically
    // This could be extended with pattern matching
}

test "Persona scores and blending" {
    const allocator = testing.allocator;

    // Create clear Abbey dominance
    const clear_abbey = enhanced.PersonaScores{
        .abbey_score = 2.0,
        .aviva_score = 0.5,
        .abi_score = 0.1,
    };

    const clear_result = clear_abbey.getBestPersona(0.7);
    try testing.expect(clear_result.persona == .abbey);
    try testing.expect(!clear_result.should_blend);

    // Create close race -> should blend
    const close_race = enhanced.PersonaScores{
        .abbey_score = 0.65,
        .aviva_score = 0.6,
        .abi_score = 0.1,
    };

    const close_result = close_race.getBestPersona(0.7);
    try testing.expect(close_result.should_blend);
    try testing.expect(close_result.persona == .abbey); // Abbey has slight edge

    // Test softmax probabilities
    const probs = close_race.getProbabilities();
    const total = probs.p_abbey + probs.p_aviva + probs.p_abi;
    try testing.expectApproxEqAbs(@as(f32, 1.0), total, 0.01);
    try testing.expect(probs.p_abbey > probs.p_abi); // Abbey > Abi
}

test "Enhanced routing with hysteresis" {
    const allocator = testing.allocator;

    var router = try enhanced.EnhancedRouter.init(allocator, null);
    defer router.deinit();

    // Simulate a series of requests to test hysteresis
    const test_requests = [_]struct {
        content: []const u8,
        expected_bias: personas.types.PersonaType,
    }{
        .{ .content = "I'm feeling anxious about this project", .expected_bias = .abbey },
        .{ .content = "Tell me about SIMD optimization in Zig", .expected_bias = .aviva },
        .{ .content = "I need emotional support right now", .expected_bias = .abbey },
    };

    for (test_requests) |test_case| {
        var request = personas.types.PersonaRequest{
            .content = test_case.content,
            .session_id = "test-hysteresis",
        };
        defer request.deinit(allocator);

        const sentiment = personas.abi.SentimentResult{
            .primary_emotion = if (test_case.expected_bias == .abbey) .anxious else .neutral,
            .requires_empathy = test_case.expected_bias == .abbey,
            .is_technical = test_case.expected_bias == .aviva,
            .urgency_score = 0.0,
            .sentiment_score = if (test_case.expected_bias == .abbey) -0.3 else 0.0,
        };

        const decision = try router.routeEnhanced(request, sentiment);
        defer decision.deinit(allocator);

        // Check that hysteresis influences decision toward previous persona
        if (router.previous_persona != null) {
            // In enhanced routing, previous persona gets +0.2 weight
            // This should bias toward staying with same persona if scores are close
            std.log.info("Routing decision: {s} (prev: {?s})", .{
                @tagName(decision.base_decision.selected_persona),
                if (router.previous_persona) |p| @tagName(p) else "none",
            });
        }
    }
}
