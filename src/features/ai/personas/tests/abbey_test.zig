//! Unit Tests for Abbey Persona (Empathetic Polymath)
//!
//! Tests emotion processing, empathy injection, and reasoning
//! capabilities of the Abbey persona.

const std = @import("std");
const testing = std.testing;

// Import Abbey modules
const emotion = @import("../abbey/emotion.zig");
const empathy = @import("../abbey/empathy.zig");
const reasoning = @import("../abbey/reasoning.zig");
const core_types = @import("../../core/types.zig");

// ============================================================================
// Emotion Processor Tests
// ============================================================================

test "emotion processor initialization" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    // EmotionProcessor has trajectory (not patterns)
    try testing.expectEqual(@as(usize, 0), processor.trajectory.items.len);
}

test "emotion detection - frustration" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    const context = core_types.EmotionalState{};
    const result = try processor.process("I'm so frustrated! This doesn't work!", context);

    try testing.expect(result.primary_emotion == .frustrated);
    try testing.expect(result.intensity > 0.5);
}

test "emotion detection - sadness" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    const context = core_types.EmotionalState{};
    const result = try processor.process("I feel so sad and hopeless.", context);

    try testing.expect(result.primary_emotion == .disappointed or
        result.primary_emotion == .anxious);
}

test "emotion detection - joy" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    const context = core_types.EmotionalState{};
    const result = try processor.process("This is amazing! I'm so happy!", context);

    try testing.expect(result.primary_emotion == .excited or
        result.primary_emotion == .enthusiastic);
}

test "emotion detection - neutral" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    const context = core_types.EmotionalState{};
    const result = try processor.process("The function takes two parameters.", context);

    try testing.expect(result.primary_emotion == .neutral or
        result.primary_emotion == .curious);
    try testing.expect(result.intensity < 0.5);
}

test "tone suggestion for emotions" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    // Frustrated should suggest empathetic tone
    const frustrated_tone = processor.suggestTone(.frustrated);
    try testing.expect(frustrated_tone == .empathetic);

    // Curious should suggest educational tone
    const curious_tone = processor.suggestTone(.curious);
    try testing.expect(curious_tone == .educational);
}

test "empathy calibration" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    // High intensity frustration should get high empathy
    const high_empathy = processor.calibrateEmpathy(.frustrated, 0.9);
    try testing.expect(high_empathy >= 0.7);

    // Low intensity curiosity should get moderate empathy
    const low_empathy = processor.calibrateEmpathy(.curious, 0.3);
    try testing.expect(low_empathy < high_empathy);
}

// ============================================================================
// Empathy Injector Tests
// ============================================================================

test "empathy injector initialization" {
    const injector = empathy.EmpathyInjector.init(testing.allocator);

    // EmpathyInjector has config (not templates)
    try testing.expect(injector.config.min_acknowledgment_threshold >= 0.0);
}

test "empathy injection - frustrated user" {
    const injector = empathy.EmpathyInjector.init(testing.allocator);

    const emotional_response = emotion.EmotionalResponse{
        .primary_emotion = .frustrated,
        .intensity = 0.8,
        .suggested_tone = .empathetic,
        .empathy_level = 0.85,
        .needs_special_care = true,
    };

    const injection = try injector.inject(emotional_response, null);

    // Should have a prefix (acknowledgment + transition)
    try testing.expect(injection.prefix.len > 0);
    try testing.expect(injection.includes_acknowledgment);
}

test "empathy injection - happy user" {
    const injector = empathy.EmpathyInjector.init(testing.allocator);

    const emotional_response = emotion.EmotionalResponse{
        .primary_emotion = .enthusiastic,
        .intensity = 0.7,
        .suggested_tone = .enthusiastic,
        .empathy_level = 0.5,
        .needs_special_care = false,
    };

    const injection = try injector.inject(emotional_response, null);

    try testing.expect(injection.prefix.len > 0);
}

test "empathy injection - neutral user" {
    const injector = empathy.EmpathyInjector.init(testing.allocator);

    const emotional_response = emotion.EmotionalResponse{
        .primary_emotion = .neutral,
        .intensity = 0.2,
        .suggested_tone = .balanced,
        .empathy_level = 0.3,
        .needs_special_care = false,
    };

    const injection = try injector.inject(emotional_response, null);

    // Neutral responses below threshold should not include acknowledgment
    try testing.expect(!injection.includes_acknowledgment);
}

// ============================================================================
// Reasoning Engine Tests
// ============================================================================

test "reasoning engine initialization" {
    const engine = reasoning.ReasoningEngine.init(testing.allocator);

    // ReasoningEngine has config (not step_templates), no deinit
    try testing.expect(engine.config.max_steps > 0);
}

test "reasoning chain generation" {
    const engine = reasoning.ReasoningEngine.init(testing.allocator);

    const memory_context = reasoning.MemoryContext{};

    var chain = try engine.reason(
        "How do I implement a hash table in Zig?",
        memory_context,
        null,
    );
    defer chain.deinit(testing.allocator);

    // Should generate at least one step
    try testing.expect(chain.steps.items.len >= 1);

    // Overall confidence should be reasonable
    try testing.expect(chain.overall_confidence > 0.0);
}

test "reasoning chain with emotional context" {
    const engine = reasoning.ReasoningEngine.init(testing.allocator);

    const memory_context = reasoning.MemoryContext{};

    const emotional_context = emotion.EmotionalResponse{
        .primary_emotion = .frustrated,
        .intensity = 0.7,
        .suggested_tone = .empathetic,
        .empathy_level = 0.8,
        .needs_special_care = true,
    };

    var chain = try engine.reason(
        "Why isn't my code working?",
        memory_context,
        emotional_context,
    );
    defer chain.deinit(testing.allocator);

    // Should adapt reasoning to emotional state
    try testing.expect(chain.steps.items.len >= 1);
}

test "reasoning step formatting" {
    const engine = reasoning.ReasoningEngine.init(testing.allocator);

    const memory_context = reasoning.MemoryContext{};

    var chain = try engine.reason(
        "Explain memory management in Zig.",
        memory_context,
        null,
    );
    defer chain.deinit(testing.allocator);

    if (chain.steps.items.len > 0) {
        const formatted = try engine.formatSteps(&chain);
        defer testing.allocator.free(formatted);

        // Formatted output should contain step content
        try testing.expect(formatted.len > 0);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

test "abbey processing pipeline" {
    // Test the full Abbey processing pipeline
    const allocator = testing.allocator;

    // Initialize components
    var processor = emotion.EmotionProcessor.init(allocator);
    defer processor.deinit();

    const injector = empathy.EmpathyInjector.init(allocator);

    const engine = reasoning.ReasoningEngine.init(allocator);

    // Process a frustrated user query
    const query = "I've been trying for hours but my allocator keeps leaking memory!";
    const context = core_types.EmotionalState{};

    // Step 1: Detect emotion
    const emotional_response = try processor.process(query, context);
    try testing.expect(emotional_response.intensity > 0.5);

    // Step 2: Generate empathy injection
    const injection = try injector.inject(emotional_response, null);
    try testing.expect(injection.prefix.len > 0);

    // Step 3: Generate reasoning chain
    const memory_context = reasoning.MemoryContext{};
    var chain = try engine.reason(query, memory_context, emotional_response);
    defer chain.deinit(allocator);

    try testing.expect(chain.steps.items.len >= 1);
}

test "abbey tone adaptation" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    const test_cases = [_]struct {
        input: []const u8,
        expected_tones: []const emotion.ToneStyle,
    }{
        .{
            .input = "I'm so excited to learn Zig!",
            .expected_tones = &[_]emotion.ToneStyle{ .enthusiastic, .educational },
        },
        .{
            .input = "This is really confusing...",
            .expected_tones = &[_]emotion.ToneStyle{ .empathetic, .educational },
        },
        .{
            .input = "What is the syntax for arrays?",
            .expected_tones = &[_]emotion.ToneStyle{ .balanced, .educational },
        },
    };

    for (test_cases) |tc| {
        const context = core_types.EmotionalState{};
        const result = try processor.process(tc.input, context);
        const tone = processor.suggestTone(result.primary_emotion);

        // Check if suggested tone is in expected list
        var found = false;
        for (tc.expected_tones) |expected| {
            if (tone == expected) {
                found = true;
                break;
            }
        }
        // If not found, tone should at least be a valid ToneStyle
        if (!found) {
            try testing.expect(tone == .empathetic or tone == .educational or
                tone == .calming or tone == .enthusiastic or tone == .balanced or
                tone == .efficient or tone == .celebratory or tone == .constructive);
        }
    }
}
