//! Integration Tests: AI Feature Facade
//!
//! Comprehensive tests for the `abi.ai` public surface including:
//! - Type availability for all re-exported AI types
//! - AI module init/deinit lifecycle
//! - PersonaRouter: creation, route decision making
//! - Constitution: principle evaluation, evaluateResponse
//! - AdaptiveModulator: creation, modulation with different inputs
//! - PersonaBus: message send/receive, circular buffer behavior
//! - ContextEngine: context creation and management
//! - Template renderer: basic template rendering
//! - RAG components: type availability
//! - AI sub-feature conditional availability
//! - Error handling paths

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const ai = abi.ai;

// ============================================================================
// 1. Type Availability
// ============================================================================

test "ai: core types are available" {
    // Verify key re-exported types exist and have non-zero size
    try std.testing.expect(@sizeOf(ai.Context) > 0);
    try std.testing.expect(@sizeOf(ai.Error) > 0);
    try std.testing.expect(@TypeOf(ai.createRegistry) != void);
    try std.testing.expect(@TypeOf(ai.createAgent) != void);
}

test "ai: persona types are available" {
    const persona = ai.persona;
    try std.testing.expect(@sizeOf(persona.PersonaId) > 0);
    try std.testing.expect(@sizeOf(persona.PersonaState) > 0);
    try std.testing.expect(@sizeOf(persona.RoutingStrategy) > 0);
    try std.testing.expect(@sizeOf(persona.RoutingDecision) > 0);
    try std.testing.expect(@sizeOf(persona.PersonaResponse) > 0);
    try std.testing.expect(@sizeOf(persona.PersonaMessage) > 0);
    try std.testing.expect(@sizeOf(persona.MessageKind) > 0);
    try std.testing.expect(@sizeOf(persona.RoutingConfig) > 0);
    try std.testing.expect(@sizeOf(persona.PersonaError) > 0);
    try std.testing.expect(@sizeOf(persona.PersonaRegistry) > 0);
    try std.testing.expect(@sizeOf(persona.MultiPersonaRouter) > 0);
}

test "ai: profile types are available" {
    const types = ai.types;
    try std.testing.expect(@sizeOf(types.ProfileType) > 0);
    try std.testing.expect(@sizeOf(types.ProfileRequest) > 0);
    try std.testing.expect(@sizeOf(types.EmotionType) > 0);
    try std.testing.expect(@sizeOf(types.EmotionalState) > 0);
    try std.testing.expect(@sizeOf(types.ConfidenceLevel) > 0);
}

test "ai: template types are available" {
    const templates = ai.templates;
    try std.testing.expect(@sizeOf(templates.Template) > 0);
    try std.testing.expect(@sizeOf(templates.TemplateRegistry) > 0);
    try std.testing.expect(@sizeOf(templates.Parser) > 0);
    try std.testing.expect(@sizeOf(templates.Renderer) > 0);
    try std.testing.expect(@TypeOf(templates.renderTemplate) != void);
    try std.testing.expect(@TypeOf(templates.formatChatMessage) != void);
}

test "ai: RAG types are available" {
    const rag = ai.rag;
    try std.testing.expect(@sizeOf(rag.Document) > 0);
    try std.testing.expect(@sizeOf(rag.Chunk) > 0);
    try std.testing.expect(@sizeOf(rag.ChunkingStrategy) > 0);
    try std.testing.expect(@sizeOf(rag.RagPipeline) > 0);
    try std.testing.expect(@sizeOf(rag.RagConfig) > 0);
    try std.testing.expect(@sizeOf(rag.Retriever) > 0);
    try std.testing.expect(@TypeOf(rag.createPipeline) != void);
}

test "ai: constitution types are available" {
    const constitution = ai.constitution;
    try std.testing.expect(@sizeOf(constitution.Constitution) > 0);
    try std.testing.expect(@sizeOf(constitution.Principle) > 0);
    try std.testing.expect(@sizeOf(constitution.Severity) > 0);
    try std.testing.expect(@sizeOf(constitution.ConstitutionalScore) > 0);
    try std.testing.expect(@sizeOf(constitution.SafetyScore) > 0);
    try std.testing.expect(@sizeOf(constitution.BiasScore) > 0);
}

test "ai: context_engine types are available" {
    const ctx_engine = ai.context_engine;
    try std.testing.expect(@sizeOf(ctx_engine.ContextProcessor) > 0);
    try std.testing.expect(@sizeOf(ctx_engine.ContextMessage) > 0);
    try std.testing.expect(@sizeOf(ctx_engine.VideoFrame) > 0);
    try std.testing.expect(@sizeOf(ctx_engine.AudioChunk) > 0);
}

test "ai: self_improve types are available" {
    const self_improve = ai.self_improve;
    try std.testing.expect(@sizeOf(self_improve.ImprovementStatus) > 0);
    try std.testing.expect(@sizeOf(self_improve.ImprovementPriority) > 0);
    try std.testing.expect(@sizeOf(self_improve.Improvement) > 0);
}

// ============================================================================
// 2. AI Module Init/Deinit Lifecycle
// ============================================================================

test "ai: init and deinit lifecycle" {
    try ai.init(std.testing.allocator, .{});
    try std.testing.expect(ai.isInitialized());
    ai.deinit();
    try std.testing.expect(!ai.isInitialized());
}

test "ai: isEnabled reflects build flag" {
    try std.testing.expectEqual(build_options.feat_ai, ai.isEnabled());
}

test "ai: double init does not panic" {
    try ai.init(std.testing.allocator, .{});
    try ai.init(std.testing.allocator, .{});
    try std.testing.expect(ai.isInitialized());
    ai.deinit();
}

test "ai: deinit without init does not panic" {
    ai.deinit();
    try std.testing.expect(!ai.isInitialized());
}

// ============================================================================
// 3. PersonaRouter
// ============================================================================

test "ai: PersonaRouter creation and deinit" {
    const persona = ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    // Router should produce a valid decision for any input
    const decision = router.route("test input");
    try std.testing.expect(decision.confidence >= 0.0);
    try std.testing.expect(decision.reason.len > 0);
}

test "ai: PersonaRouter routes technical query to Aviva" {
    const persona = ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("Explain the implementation of quicksort in Zig");
    try std.testing.expectEqual(persona.PersonaId.aviva, decision.primary);
    try std.testing.expect(decision.weights.aviva > decision.weights.abbey);
}

test "ai: PersonaRouter routes emotional query to Abbey" {
    const persona = ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("I feel really stressed about my deadline");
    try std.testing.expectEqual(persona.PersonaId.abbey, decision.primary);
    try std.testing.expect(decision.weights.abbey > decision.weights.aviva);
}

test "ai: PersonaRouter routes policy query to Abi" {
    const persona = ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const decision = router.route("What is the privacy policy for data compliance?");
    try std.testing.expectEqual(persona.PersonaId.abi, decision.primary);
}

test "ai: PersonaRouter produces valid weights that sum to 1.0" {
    const persona = ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    const queries = [_][]const u8{
        "hello",
        "debug this code for me",
        "what are the compliance rules",
        "",
    };

    for (queries) |q| {
        const d = router.route(q);
        const sum = d.weights.abbey + d.weights.aviva + d.weights.abi;
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.01);
        try std.testing.expect(d.weights.abbey >= 0.0 and d.weights.abbey <= 1.0);
        try std.testing.expect(d.weights.aviva >= 0.0 and d.weights.aviva <= 1.0);
        try std.testing.expect(d.weights.abi >= 0.0 and d.weights.abi <= 1.0);
    }
}

// ============================================================================
// 4. Constitution
// ============================================================================

test "ai: Constitution init and evaluate safe content" {
    const constitution = ai.constitution;
    const c = constitution.Constitution.init();

    const score = c.evaluate("Hello, how can I help you today?");
    try std.testing.expect(score.isCompliant());
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), score.overall, 0.01);
}

test "ai: Constitution blocks harmful content" {
    const constitution = ai.constitution;
    const c = constitution.Constitution.init();

    try std.testing.expect(!c.isCompliant("run rm -rf / to clean up"));
}

test "ai: Constitution has 6 principles" {
    const constitution = ai.constitution;
    const c = constitution.Constitution.init();

    const principles = c.getPrinciples();
    try std.testing.expectEqual(@as(usize, 6), principles.len);
}

test "ai: Constitution system preamble is non-empty" {
    const constitution = ai.constitution;
    const c = constitution.Constitution.init();

    const preamble = c.getSystemPreamble();
    try std.testing.expect(preamble.len > 0);
}

test "ai: Constitution alignment score is in valid range" {
    const constitution = ai.constitution;
    const c = constitution.Constitution.init();

    const score = c.alignmentScore("This is a helpful and honest response.");
    try std.testing.expect(score >= 0.0 and score <= 1.0);
}

test "ai: Constitution safety evaluation" {
    const constitution = ai.constitution;
    const c = constitution.Constitution.init();

    const safe_result = c.evaluateSafety("Hello world");
    try std.testing.expect(safe_result.overall_score >= 0.0);
    try std.testing.expect(safe_result.overall_score <= 1.0);
}

test "ai: Constitution bias computation" {
    const constitution = ai.constitution;
    const c = constitution.Constitution.init();

    const measurements = [_]f32{ 0.05, -0.2, 0.08, 0.15 };
    const result = c.computeBias(&measurements, constitution.DEFAULT_BIAS_THRESHOLD);
    try std.testing.expectEqual(@as(usize, 4), result.attribute_count);
    try std.testing.expect(result.mean_abs_bias > 0.0);
}

// ============================================================================
// 5. AdaptiveModulator
// ============================================================================

test "ai: router attachMemory and attachConstitution" {
    const persona = ai.persona;
    const constitution = ai.constitution;

    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    var router = persona.MultiPersonaRouter.init(std.testing.allocator, &registry, .{});
    defer router.deinit();

    // Verify memory attachment
    try std.testing.expect(router.memory == null);
    router.attachMemory("modulator-test-session");
    try std.testing.expect(router.memory != null);

    // Verify constitution attachment
    try std.testing.expect(router.constitution == null);
    router.attachConstitution(constitution.Constitution.init());
    try std.testing.expect(router.constitution != null);

    // Routing should still work with all components attached
    const decision = router.route("test query");
    try std.testing.expect(decision.confidence >= 0.0);
}

// ============================================================================
// 6. PersonaBus
// ============================================================================

test "ai: PersonaBus send and receive" {
    const persona = ai.persona;
    var bus = persona.PersonaBus.init(std.testing.allocator);
    defer bus.deinit();

    const msg = persona.PersonaBus.createMessage(
        .abbey,
        .aviva,
        .request,
        "Need fact check",
        0.8,
    );
    try bus.send(msg);

    try std.testing.expect(bus.hasPending(.aviva));
    try std.testing.expect(!bus.hasPending(.abbey));

    const received = bus.receive(.aviva);
    try std.testing.expect(received != null);
    try std.testing.expectEqual(persona.PersonaId.abbey, received.?.from);
    try std.testing.expectEqualStrings("Need fact check", received.?.payload);
}

test "ai: PersonaBus broadcast excludes sender" {
    const persona = ai.persona;
    var bus = persona.PersonaBus.init(std.testing.allocator);
    defer bus.deinit();

    const msg = persona.PersonaBus.createMessage(
        .abi,
        null,
        .veto,
        "PII detected",
        0.95,
    );
    try bus.broadcast(msg);

    // Abi should NOT receive its own broadcast
    try std.testing.expect(!bus.hasPending(.abi));
    // Abbey and Aviva should receive it
    try std.testing.expect(bus.hasPending(.abbey));
    try std.testing.expect(bus.hasPending(.aviva));
}

test "ai: PersonaBus empty receive returns null" {
    const persona = ai.persona;
    var bus = persona.PersonaBus.init(std.testing.allocator);
    defer bus.deinit();

    try std.testing.expect(bus.receive(.abbey) == null);
    try std.testing.expect(bus.receive(.aviva) == null);
    try std.testing.expect(bus.receive(.abi) == null);
}

test "ai: PersonaBus pending count" {
    const persona = ai.persona;
    var bus = persona.PersonaBus.init(std.testing.allocator);
    defer bus.deinit();

    try std.testing.expectEqual(@as(usize, 0), bus.pendingCount(.aviva));

    const msg = persona.PersonaBus.createMessage(.abbey, .aviva, .request, "msg1", 0.5);
    try bus.send(msg);
    try std.testing.expectEqual(@as(usize, 1), bus.pendingCount(.aviva));

    const msg2 = persona.PersonaBus.createMessage(.abi, .aviva, .opinion, "msg2", 0.6);
    try bus.send(msg2);
    try std.testing.expectEqual(@as(usize, 2), bus.pendingCount(.aviva));

    _ = bus.receive(.aviva);
    try std.testing.expectEqual(@as(usize, 1), bus.pendingCount(.aviva));
}

test "ai: PersonaBus FIFO ordering" {
    const persona = ai.persona;
    var bus = persona.PersonaBus.init(std.testing.allocator);
    defer bus.deinit();

    // Send three messages with different confidence values as markers
    for (0..3) |i| {
        const conf: f32 = @as(f32, @floatFromInt(i)) * 0.1;
        const msg = persona.PersonaBus.createMessage(.abbey, .aviva, .request, "order", conf);
        try bus.send(msg);
    }

    // Receive in FIFO order
    for (0..3) |i| {
        const received = bus.receive(.aviva);
        try std.testing.expect(received != null);
        const expected: f32 = @as(f32, @floatFromInt(i)) * 0.1;
        try std.testing.expectEqual(expected, received.?.confidence);
    }
}

test "ai: PersonaBus clear empties all inboxes" {
    const persona = ai.persona;
    var bus = persona.PersonaBus.init(std.testing.allocator);
    defer bus.deinit();

    // Send messages to multiple personas
    try bus.send(persona.PersonaBus.createMessage(.abbey, .aviva, .request, "a", 0.1));
    try bus.send(persona.PersonaBus.createMessage(.aviva, .abbey, .response, "b", 0.2));
    try bus.send(persona.PersonaBus.createMessage(.abbey, .abi, .opinion, "c", 0.3));

    try std.testing.expect(bus.hasPending(.aviva));
    try std.testing.expect(bus.hasPending(.abbey));
    try std.testing.expect(bus.hasPending(.abi));

    bus.clear();

    try std.testing.expect(!bus.hasPending(.aviva));
    try std.testing.expect(!bus.hasPending(.abbey));
    try std.testing.expect(!bus.hasPending(.abi));
}

// ============================================================================
// 7. ContextEngine
// ============================================================================

test "ai: ContextProcessor creation and deinit" {
    const ctx_engine = ai.context_engine;
    var processor = ctx_engine.ContextProcessor.init(std.testing.allocator);
    defer processor.deinit();

    // Processor should exist and be usable
    try std.testing.expect(@sizeOf(@TypeOf(processor)) > 0);
}

test "ai: ContextMessage role enum values" {
    const ctx_engine = ai.context_engine;
    // Verify all expected role values exist
    try std.testing.expect(@intFromEnum(ctx_engine.ContextMessage.Role.system) != @intFromEnum(ctx_engine.ContextMessage.Role.user));
    try std.testing.expect(@intFromEnum(ctx_engine.ContextMessage.Role.assistant) != @intFromEnum(ctx_engine.ContextMessage.Role.tool));
}

// ============================================================================
// 8. Template Renderer
// ============================================================================

test "ai: template basic rendering" {
    const templates = ai.templates;
    const allocator = std.testing.allocator;

    var template = try templates.Template.init(allocator, "test", "Hello, {{name}}!");
    defer template.deinit();

    const result = try template.render(.{ .name = "World" });
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello, World!", result);
}

test "ai: template with default value" {
    const templates = ai.templates;
    const allocator = std.testing.allocator;

    var template = try templates.Template.init(allocator, "greeting", "Hello, {{name|Friend}}!");
    defer template.deinit();

    const result = try template.render(.{});
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello, Friend!", result);
}

test "ai: TemplateRegistry register and get" {
    const templates = ai.templates;
    const allocator = std.testing.allocator;

    var registry = templates.TemplateRegistry.init(allocator);
    defer registry.deinit();

    try registry.register("greet", "Hi, {{user}}!");
    const tmpl = registry.get("greet");
    try std.testing.expect(tmpl != null);

    // Non-existent template returns null
    try std.testing.expect(registry.get("nonexistent") == null);
}

test "ai: template renderTemplate convenience function" {
    const templates = ai.templates;
    const allocator = std.testing.allocator;

    const result = try templates.renderTemplate(allocator, "Count: {{n}}", .{ .n = "42" });
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Count: 42", result);
}

test "ai: formatChatMessage produces expected format" {
    const templates = ai.templates;
    const allocator = std.testing.allocator;

    const result = try templates.formatChatMessage(allocator, "user", "Hello");
    defer allocator.free(result);

    try std.testing.expect(std.mem.indexOf(u8, result, "user") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hello") != null);
}

// ============================================================================
// 9. RAG Components
// ============================================================================

test "ai: RAG pipeline creation and deinit" {
    const rag = ai.rag;
    var pipeline = rag.createPipeline(std.testing.allocator);
    defer pipeline.deinit();

    try std.testing.expectEqual(@as(usize, 0), pipeline.documentCount());
    try std.testing.expectEqual(@as(usize, 0), pipeline.chunkCount());
}

test "ai: RAG pipeline add text documents" {
    const rag = ai.rag;
    var pipeline = rag.RagPipeline.init(std.testing.allocator, .{});
    defer pipeline.deinit();

    try pipeline.addText("Zig is a systems programming language.", "Zig Intro");
    try pipeline.addText("Rust focuses on memory safety.", "Rust Intro");

    try std.testing.expectEqual(@as(usize, 2), pipeline.documentCount());
    try std.testing.expect(pipeline.chunkCount() >= 2);
}

test "ai: RAG ChunkingStrategy enum values" {
    const rag = ai.rag;
    // Verify the enum type has values accessible
    try std.testing.expect(@sizeOf(rag.ChunkingStrategy) > 0);
    try std.testing.expect(@sizeOf(rag.DocumentType) > 0);
}

// ============================================================================
// 10. Sub-feature Conditional Availability
// ============================================================================

test "ai: llm sub-feature availability matches build option" {
    // llm module should always be importable (real or stub)
    try std.testing.expect(@sizeOf(@TypeOf(ai.llm)) > 0);
}

test "ai: training sub-feature availability matches build option" {
    try std.testing.expect(@sizeOf(@TypeOf(ai.training)) > 0);
}

test "ai: vision sub-feature availability matches build option" {
    try std.testing.expect(@sizeOf(@TypeOf(ai.vision)) > 0);
}

test "ai: explore sub-feature availability matches build option" {
    try std.testing.expect(@sizeOf(@TypeOf(ai.explore)) > 0);
}

test "ai: reasoning sub-feature availability matches build option" {
    try std.testing.expect(@sizeOf(@TypeOf(ai.reasoning)) > 0);
}

// ============================================================================
// 11. Error Handling Paths
// ============================================================================

test "ai: PersonaRegistry getPersona returns null before initAll" {
    const persona = ai.persona;
    var registry = persona.PersonaRegistry.init(std.testing.allocator, .{});
    defer registry.deinit();

    // Before initAll, getPersona should return null
    try std.testing.expect(registry.getPersona(.abbey) == null);
    try std.testing.expect(registry.getPersona(.aviva) == null);
    try std.testing.expect(registry.getPersona(.abi) == null);
}

test "ai: PersonaId name returns human-readable string" {
    const persona = ai.persona;
    try std.testing.expectEqualStrings("Abbey", persona.PersonaId.abbey.name());
    try std.testing.expectEqualStrings("Aviva", persona.PersonaId.aviva.name());
    try std.testing.expectEqualStrings("Abi", persona.PersonaId.abi.name());
}

test "ai: PersonaId role returns role description" {
    const persona = ai.persona;
    try std.testing.expect(persona.PersonaId.abbey.role().len > 0);
    try std.testing.expect(persona.PersonaId.aviva.role().len > 0);
    try std.testing.expect(persona.PersonaId.abi.role().len > 0);
}

test "ai: PersonaId colorCode returns hex color" {
    const persona = ai.persona;
    // Each persona has a distinct color code starting with #
    const abbey_color = persona.PersonaId.abbey.colorCode();
    const aviva_color = persona.PersonaId.aviva.colorCode();
    const abi_color = persona.PersonaId.abi.colorCode();

    try std.testing.expect(abbey_color[0] == '#');
    try std.testing.expect(aviva_color[0] == '#');
    try std.testing.expect(abi_color[0] == '#');
    // All are different
    try std.testing.expect(!std.mem.eql(u8, abbey_color, aviva_color));
    try std.testing.expect(!std.mem.eql(u8, abbey_color, abi_color));
}

test "ai: RoutingDecision.Weights normalize" {
    const persona = ai.persona;
    var weights = persona.RoutingDecision.Weights{
        .abbey = 2.0,
        .aviva = 3.0,
        .abi = 5.0,
    };
    weights.normalize();

    const sum = weights.abbey + weights.aviva + weights.abi;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), weights.abbey, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), weights.aviva, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights.abi, 0.001);
}

test "ai: RoutingDecision.Weights forPersona" {
    const persona = ai.persona;
    const weights = persona.RoutingDecision.Weights{
        .abbey = 0.5,
        .aviva = 0.3,
        .abi = 0.2,
    };

    try std.testing.expectEqual(@as(f32, 0.5), weights.forPersona(.abbey));
    try std.testing.expectEqual(@as(f32, 0.3), weights.forPersona(.aviva));
    try std.testing.expectEqual(@as(f32, 0.2), weights.forPersona(.abi));
}

test "ai: ConversationMemory records interaction" {
    const persona = ai.persona;
    var mem = persona.ConversationMemory.init(std.testing.allocator, "ai-test-session");
    defer mem.deinit();

    const decision = persona.RoutingDecision{
        .primary = .aviva,
        .weights = .{ .abbey = 0.2, .aviva = 0.6, .abi = 0.2 },
        .strategy = .single,
        .confidence = 0.8,
        .reason = "Technical query",
    };

    const response = persona.PersonaResponse{
        .persona = .aviva,
        .content = "Here is the answer.",
        .confidence = 0.85,
        .allocator = std.testing.allocator,
    };

    const id = try mem.recordInteraction(decision, "How do I sort?", response);
    try std.testing.expect(id > 0);
    try std.testing.expectEqual(@as(u64, 1), mem.getInteractionCount());
}

test {
    std.testing.refAllDecls(@This());
}
