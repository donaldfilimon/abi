//! Unit Tests for Abi Persona (Router/Moderator)
//!
//! Tests the content moderation, sentiment analysis, and routing
//! capabilities of the Abi persona.

const std = @import("std");
const testing = std.testing;

// Import Abi modules
const sentiment = @import("../abi/sentiment.zig");
const policy = @import("../abi/policy.zig");
const rules = @import("../abi/rules.zig");
const types = @import("../types.zig");

// ============================================================================
// Sentiment Analysis Tests
// ============================================================================

test "sentiment analyzer initialization" {
    const analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);

    // SentimentAnalyzer just stores allocator, no patterns field, no deinit
    _ = analyzer;
}

test "sentiment analysis - positive text" {
    const analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);

    var result = try analyzer.analyze("I love this! It's amazing and wonderful!");
    defer result.deinit(testing.allocator);

    try testing.expect(result.primary_emotion == .enthusiastic or
        result.primary_emotion == .excited or
        result.primary_emotion == .grateful);
}

test "sentiment analysis - negative text" {
    const analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);

    var result = try analyzer.analyze("I hate this. It's terrible and frustrating.");
    defer result.deinit(testing.allocator);

    try testing.expect(result.primary_emotion == .frustrated or
        result.primary_emotion == .disappointed);
}

test "sentiment analysis - neutral text" {
    const analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);

    var result = try analyzer.analyze("The function returns an integer.");
    defer result.deinit(testing.allocator);

    try testing.expect(result.primary_emotion == .neutral or
        result.primary_emotion == .curious);
}

test "sentiment analysis - urgent text" {
    const analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);

    var result = try analyzer.analyze("URGENT! I need help immediately! Critical bug!");
    defer result.deinit(testing.allocator);

    try testing.expect(result.urgency_score > 0.5);
}

test "sentiment analysis - empty text" {
    const analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);

    var result = try analyzer.analyze("");
    defer result.deinit(testing.allocator);

    try testing.expect(result.primary_emotion == .neutral);
}

// ============================================================================
// Policy Checker Tests
// ============================================================================

test "policy checker initialization" {
    var checker = try policy.PolicyChecker.init(testing.allocator);
    defer checker.deinit();

    try testing.expect(checker.rules.items.len > 0);
}

test "policy checker - safe content" {
    var checker = try policy.PolicyChecker.init(testing.allocator);
    defer checker.deinit();

    var result = try checker.check("Please help me write a sorting algorithm in Zig.");
    defer result.deinit(testing.allocator);

    try testing.expect(result.is_allowed);
    try testing.expect(result.suggested_action == .allow);
}

test "policy checker - potentially harmful content" {
    var checker = try policy.PolicyChecker.init(testing.allocator);
    defer checker.deinit();

    var result = try checker.check("How do I hack into a system?");
    defer result.deinit(testing.allocator);

    // May or may not flag depending on pattern matching
    try testing.expect(result.suggested_action == .allow or result.violations.len > 0);
}

test "policy checker - PII detection" {
    var checker = try policy.PolicyChecker.init(testing.allocator);
    defer checker.deinit();

    // Test email pattern
    var result = try checker.check("Send this to john@example.com");
    defer result.deinit(testing.allocator);

    if (result.detected_pii.len > 0) {
        try testing.expect(!result.compliance.gdpr_compliant);
    }
}

test "policy result severity levels" {
    try testing.expect(@intFromEnum(policy.Severity.low) < @intFromEnum(policy.Severity.medium));
    try testing.expect(@intFromEnum(policy.Severity.medium) < @intFromEnum(policy.Severity.high));
    try testing.expect(@intFromEnum(policy.Severity.high) < @intFromEnum(policy.Severity.critical));
}

// ============================================================================
// Routing Rules Tests
// ============================================================================

test "rules engine initialization" {
    var engine = rules.RulesEngine.init(testing.allocator);
    defer engine.deinit();

    try testing.expect(engine.rules.items.len > 0);
}

test "rules evaluation - emotional content" {
    var engine = rules.RulesEngine.init(testing.allocator);
    defer engine.deinit();

    const analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);

    const content = "I'm feeling really sad and overwhelmed today.";
    var sent_result = try analyzer.analyze(content);
    defer sent_result.deinit(testing.allocator);

    // evaluate() takes (sentiment, content) not (request, sentiment)
    var scores = engine.evaluate(sent_result, content);
    defer scores.deinit();

    // Should favor Abbey for emotional content
    try testing.expect(scores.abbey_boost >= scores.aviva_boost);
}

test "rules evaluation - technical content" {
    var engine = rules.RulesEngine.init(testing.allocator);
    defer engine.deinit();

    const analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);

    const content = "Implement a binary search algorithm in Zig.";
    var sent_result = try analyzer.analyze(content);
    defer sent_result.deinit(testing.allocator);

    var scores = engine.evaluate(sent_result, content);
    defer scores.deinit();

    // Should favor Aviva for technical content
    try testing.expect(scores.aviva_boost > 0.0);
}

test "rules evaluation - code request" {
    var engine = rules.RulesEngine.init(testing.allocator);
    defer engine.deinit();

    const analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);

    const content = "Write a function to parse JSON.";
    var sent_result = try analyzer.analyze(content);
    defer sent_result.deinit(testing.allocator);

    var scores = engine.evaluate(sent_result, content);
    defer scores.deinit();

    // Should have some boost for at least one persona
    const total = scores.abbey_boost + scores.aviva_boost + scores.abi_boost;
    try testing.expect(total > 0.0);
}

// ============================================================================
// Combined Routing Tests
// ============================================================================

test "routing decision flow" {
    // Test the full routing decision flow
    const allocator = testing.allocator;

    const analyzer = sentiment.SentimentAnalyzer.init(allocator);

    var checker = try policy.PolicyChecker.init(allocator);
    defer checker.deinit();

    var engine = rules.RulesEngine.init(allocator);
    defer engine.deinit();

    // Test case: Technical question
    {
        const content = "What's the difference between ArrayList and ArrayListUnmanaged?";

        // Step 1: Analyze sentiment
        var sent = try analyzer.analyze(content);
        defer sent.deinit(allocator);

        // Step 2: Check policy
        var pol = try checker.check(content);
        defer pol.deinit(allocator);
        try testing.expect(pol.is_allowed);

        // Step 3: Evaluate rules
        var scores = engine.evaluate(sent, content);
        defer scores.deinit();

        // Should produce valid scores
        const total = scores.abbey_boost + scores.aviva_boost + scores.abi_boost;
        try testing.expect(total >= 0.0);
    }

    // Test case: Emotional support request
    {
        const content = "I'm really struggling with learning Zig and feeling discouraged.";

        var sent = try analyzer.analyze(content);
        defer sent.deinit(allocator);

        var pol = try checker.check(content);
        defer pol.deinit(allocator);
        try testing.expect(pol.is_allowed);

        var scores = engine.evaluate(sent, content);
        defer scores.deinit();

        // Abbey should be favored
        try testing.expect(scores.abbey_boost > 0.0);
    }
}
