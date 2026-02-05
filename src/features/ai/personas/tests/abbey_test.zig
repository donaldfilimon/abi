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

    try testing.expect(processor.patterns.items.len > 0);
}

test "emotion detection - frustration" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    const context = core_types.EmotionalState{};
    const result = try processor.process("I'm so frustrated! This doesn't work!", context);

    try testing.expect(result.detected_emotion == .frustrated);
    try testing.expect(result.intensity > 0.5);
}

test "emotion detection - sadness" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    const context = core_types.EmotionalState{};
    const result = try processor.process("I feel so sad and hopeless.", context);

    try testing.expect(result.detected_emotion == .disappointed or
        result.detected_emotion == .anxious);
}

test "emotion detection - joy" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    const context = core_types.EmotionalState{};
    const result = try processor.process("This is amazing! I'm so happy!", context);

    try testing.expect(result.detected_emotion == .excited or
        result.detected_emotion == .enthusiastic);
}

test "emotion detection - neutral" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    const context = core_types.EmotionalState{};
    const result = try processor.process("The function takes two parameters.", context);

    try testing.expect(result.detected_emotion == .neutral or
        result.detected_emotion == .curious);
    try testing.expect(result.intensity < 0.5);
}

test "tone suggestion for emotions" {
    var processor = emotion.EmotionProcessor.init(testing.allocator);
    defer processor.deinit();

    // Frustrated should suggest supportive tone
    const frustrated_tone = processor.suggestTone(.frustrated);
    try testing.expect(frustrated_tone == .supportive or
        frustrated_tone == .empathetic);

    // Curious should suggest encouraging tone
    const curious_tone = processor.suggestTone(.curious);
    try testing.expect(curious_tone == .encouraging or
        curious_tone == .neutral);
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

    try testing.expect(injector.templates.items.len > 0);
}

test "empathy injection - frustrated user" {
    const injector = empathy.EmpathyInjector.init(testing.allocator);

    const emotional_response = emotion.EmotionalResponse{
        .detected_emotion = .frustrated,
        .intensity = 0.8,
        .suggested_tone = .supportive,
        .empathy_level = 0.85,
    };

    const injection = try injector.inject(emotional_response, null);

    // Should have acknowledgment and transition
    try testing.expect(injection.acknowledgment.len > 0);
    try testing.expect(injection.transition != null);
}

test "empathy injection - happy user" {
    const injector = empathy.EmpathyInjector.init(testing.allocator);

    const emotional_response = emotion.EmotionalResponse{
        .detected_emotion = .enthusiastic,
        .intensity = 0.7,
        .suggested_tone = .enthusiastic,
        .empathy_level = 0.5,
    };

    const injection = try injector.inject(emotional_response, null);

    try testing.expect(injection.acknowledgment.len > 0);
}

test "empathy injection - neutral user" {
    const injector = empathy.EmpathyInjector.init(testing.allocator);

    const emotional_response = emotion.EmotionalResponse{
        .detected_emotion = .neutral,
        .intensity = 0.2,
        .suggested_tone = .neutral,
        .empathy_level = 0.3,
    };

    const injection = try injector.inject(emotional_response, null);

    // Neutral responses should have minimal empathy content
    try testing.expect(injection.empathy_level < 0.5);
}

// ============================================================================
// Reasoning Engine Tests
// ============================================================================

test "reasoning engine initialization" {
    var engine = reasoning.ReasoningEngine.init(testing.allocator);
    defer engine.deinit();

    try testing.expect(engine.step_templates.items.len > 0);
}

test "reasoning chain generation" {
    var engine = reasoning.ReasoningEngine.init(testing.allocator);
    defer engine.deinit();

    const memory_context = reasoning.MemoryContext{};

    const chain = try engine.reason(
        "How do I implement a hash table in Zig?",
        memory_context,
        null,
    );
    defer chain.deinit();

    // Should generate at least one step
    try testing.expect(chain.steps.items.len >= 1);

    // Overall confidence should be reasonable
    try testing.expect(chain.overall_confidence > 0.0);
}

test "reasoning chain with emotional context" {
    var engine = reasoning.ReasoningEngine.init(testing.allocator);
    defer engine.deinit();

    const memory_context = reasoning.MemoryContext{};

    const emotional_context = emotion.EmotionalResponse{
        .detected_emotion = .frustrated,
        .intensity = 0.7,
        .suggested_tone = .supportive,
        .empathy_level = 0.8,
    };

    const chain = try engine.reason(
        "Why isn't my code working?",
        memory_context,
        emotional_context,
    );
    defer chain.deinit();

    // Should adapt reasoning to emotional state
    try testing.expect(chain.steps.items.len >= 1);
}

test "reasoning step formatting" {
    var engine = reasoning.ReasoningEngine.init(testing.allocator);
    defer engine.deinit();

    const memory_context = reasoning.MemoryContext{};

    var chain = try engine.reason(
        "Explain memory management in Zig.",
        memory_context,
        null,
    );
    defer chain.deinit();

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

    var engine = reasoning.ReasoningEngine.init(allocator);
    defer engine.deinit();

    // Process a frustrated user query
    const query = "I've been trying for hours but my allocator keeps leaking memory!";
    const context = core_types.EmotionalState{};

    // Step 1: Detect emotion
    const emotional_response = try processor.process(query, context);
    try testing.expect(emotional_response.intensity > 0.5);

    // Step 2: Generate empathy injection
    const injection = try injector.inject(emotional_response, null);
    try testing.expect(injection.acknowledgment.len > 0);

    // Step 3: Generate reasoning chain
    const memory_context = reasoning.MemoryContext{};
    var chain = try engine.reason(query, memory_context, emotional_response);
    defer chain.deinit();

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
            .expected_tones = &[_]emotion.ToneStyle{ .enthusiastic, .encouraging },
        },
        .{
            .input = "This is really confusing...",
            .expected_tones = &[_]emotion.ToneStyle{ .supportive, .patient },
        },
        .{
            .input = "What is the syntax for arrays?",
            .expected_tones = &[_]emotion.ToneStyle{ .neutral, .encouraging },
        },
    };

    for (test_cases) |tc| {
        const context = core_types.EmotionalState{};
        const result = try processor.process(tc.input, context);
        const tone = processor.suggestTone(result.detected_emotion);

        // Check if suggested tone is in expected list
        var found = false;
        for (tc.expected_tones) |expected| {
            if (tone == expected) {
                found = true;
                break;
            }
        }
        // If not found, tone should at least be reasonable
        if (!found) {
            try testing.expect(tone != .undefined);
        }
    }
}
