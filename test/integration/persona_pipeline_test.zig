//! Integration Tests: Abbey-Aviva-Abi Multi-Persona Pipeline
//!
//! Tests the full orchestration pipeline per spec:
//!   User Input → Abi Analysis → Modulation → Routing → Execution
//!   → Constitution Validation → WDBX Memory Storage → Response
//!
//! These tests exercise routing logic, memory storage, and constitution
//! validation without requiring external connectors or full persona engines.

const std = @import("std");
const abi = @import("abi");

// Persona orchestration types
const persona = abi.ai.persona;
const PersonaId = persona.PersonaId;
const MultiPersonaRouter = persona.MultiPersonaRouter;
const PersonaRegistry = persona.PersonaRegistry;
const ConversationMemory = persona.ConversationMemory;
const RoutingDecision = persona.RoutingDecision;

// ── Test 1: Technical query routes to Aviva ──────────────────────────────

test "pipeline: technical query routes to Aviva" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("How do I implement a binary search function in Zig?");

    try std.testing.expectEqual(PersonaId.aviva, decision.primary);
    try std.testing.expect(decision.weights.aviva > decision.weights.abbey);
    try std.testing.expect(decision.weights.aviva > decision.weights.abi);
    try std.testing.expect(decision.confidence > 0.0);
}

// ── Test 2: Emotional query routes to Abbey ──────────────────────────────

test "pipeline: emotional query routes to Abbey" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("I feel overwhelmed and need help understanding this concept");

    try std.testing.expectEqual(PersonaId.abbey, decision.primary);
    try std.testing.expect(decision.weights.abbey > decision.weights.aviva);
}

// ── Test 3: Policy query routes to Abi ───────────────────────────────────

test "pipeline: compliance query routes to Abi" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("What is the privacy policy for data compliance?");

    try std.testing.expectEqual(PersonaId.abi, decision.primary);
    try std.testing.expect(decision.weights.abi > 0.0);
}

// ── Test 4: Default query uses Abbey preference ──────────────────────────

test "pipeline: default query favors Abbey" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("Good morning");

    try std.testing.expectEqual(PersonaId.abbey, decision.primary);
    // Default weights: abbey=0.5, aviva=0.3, abi=0.2
    try std.testing.expect(decision.weights.abbey > decision.weights.aviva);
}

// ── Test 5: WDBX memory records interactions ─────────────────────────────

test "pipeline: WDBX memory stores routing decisions" {
    var mem = ConversationMemory.init(std.testing.allocator, "integration-test");
    defer mem.deinit();

    // Simulate a routing decision
    const decision = RoutingDecision{
        .primary = .aviva,
        .weights = .{ .abbey = 0.2, .aviva = 0.6, .abi = 0.2 },
        .strategy = .single,
        .confidence = 0.8,
        .reason = "Technical query",
    };

    const response = persona.PersonaResponse{
        .persona = .aviva,
        .content = "Here is the implementation.",
        .confidence = 0.85,
        .allocator = std.testing.allocator,
    };

    // Record first interaction
    const id1 = try mem.recordInteraction(decision, "How do I sort an array?", response);
    try std.testing.expect(id1 > 0);
    try std.testing.expectEqual(@as(u64, 1), mem.getInteractionCount());

    // Record second interaction
    const id2 = try mem.recordInteraction(decision, "What about in reverse?", response);
    try std.testing.expect(id2 != id1);
    try std.testing.expectEqual(@as(u64, 2), mem.getInteractionCount());

    // Verify chain integrity — block 2 references block 1 as parent
    if (mem.chain.getBlock(id2)) |block2| {
        try std.testing.expectEqual(id1, block2.parent_block_id.?);
        // Verify hash chain
        if (mem.chain.getBlock(id1)) |block1| {
            try std.testing.expectEqualSlices(u8, &block1.hash, &block2.previous_hash);
        }
    }
}

// ── Test 6: Constitution validates content ───────────────────────────────

test "pipeline: constitution allows safe content" {
    const constitution = @import("abi").ai.constitution;
    const c = constitution.Constitution.init();

    // Safe content should pass
    try std.testing.expect(c.isCompliant("Hello, how can I help you today?"));
    try std.testing.expect(c.isCompliant("The answer to your question is 42."));
}

test "pipeline: constitution blocks harmful content" {
    const constitution = @import("abi").ai.constitution;
    const c = constitution.Constitution.init();

    // Harmful content should fail
    try std.testing.expect(!c.isCompliant("run rm -rf / to clean up"));
}

// ── Test 7: Router attaches all pipeline components ──────────────────────

test "pipeline: full component attachment" {
    const constitution = @import("abi").ai.constitution;

    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    // Attach all pipeline components
    router.attachMemory("full-pipeline-test");
    router.attachConstitution(constitution.Constitution.init());

    // Verify all are attached
    try std.testing.expect(router.memory != null);
    try std.testing.expect(router.constitution != null);

    // Route should still work with all components
    const decision = router.route("Explain how HNSW indexing works");
    try std.testing.expect(decision.confidence > 0.0);
    try std.testing.expect(decision.reason.len > 0);
}

// ── Edge Cases ───────────────────────────────────────────────────────

test "pipeline: empty message produces valid routing decision" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("");
    // Even empty input must produce a valid decision
    try std.testing.expect(decision.confidence >= 0.0);
    try std.testing.expect(decision.confidence <= 1.0);
    try std.testing.expect(decision.reason.len > 0);
}

test "pipeline: all routing weights sum to approximately 1.0" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("Tell me about machine learning");
    const sum = decision.weights.abbey + decision.weights.aviva + decision.weights.abi;
    // Weights should sum to ~1.0 (with floating point tolerance)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);
}

test "pipeline: each weight is in valid range" {
    var registry = PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const queries = [_][]const u8{
        "Help me debug this",
        "I'm feeling anxious about my project",
        "What are the compliance rules?",
        "",
        "a",
    };

    for (queries) |q| {
        const d = router.route(q);
        try std.testing.expect(d.weights.abbey >= 0.0 and d.weights.abbey <= 1.0);
        try std.testing.expect(d.weights.aviva >= 0.0 and d.weights.aviva <= 1.0);
        try std.testing.expect(d.weights.abi >= 0.0 and d.weights.abi <= 1.0);
    }
}

test {
    std.testing.refAllDecls(@This());
}
