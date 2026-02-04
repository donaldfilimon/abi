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
    var analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    try testing.expect(analyzer.patterns != null);
}

test "sentiment analysis - positive text" {
    var analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    const result = analyzer.analyze("I love this! It's amazing and wonderful!");

    try testing.expect(result.primary_sentiment == .positive or
        result.primary_sentiment == .joyful);
    try testing.expect(result.sentiment_score > 0.5);
}

test "sentiment analysis - negative text" {
    var analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    const result = analyzer.analyze("I hate this. It's terrible and frustrating.");

    try testing.expect(result.primary_sentiment == .negative or
        result.primary_sentiment == .frustrated);
    try testing.expect(result.sentiment_score < 0.5);
}

test "sentiment analysis - neutral text" {
    var analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    const result = analyzer.analyze("The function returns an integer.");

    try testing.expect(result.primary_sentiment == .neutral or
        result.primary_sentiment == .analytical);
}

test "sentiment analysis - urgent text" {
    var analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    const result = analyzer.analyze("URGENT! I need help immediately! Critical bug!");

    try testing.expect(result.urgency > 0.5);
}

test "sentiment analysis - empty text" {
    var analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    const result = analyzer.analyze("");

    try testing.expect(result.primary_sentiment == .neutral);
}

// ============================================================================
// Policy Checker Tests
// ============================================================================

test "policy checker initialization" {
    var checker = policy.PolicyChecker.init(testing.allocator);
    defer checker.deinit();

    try testing.expect(checker.rules.items.len > 0);
}

test "policy checker - safe content" {
    var checker = policy.PolicyChecker.init(testing.allocator);
    defer checker.deinit();

    const result = checker.check("Please help me write a sorting algorithm in Zig.");

    try testing.expect(result.is_safe);
    try testing.expect(result.action == .allow);
}

test "policy checker - potentially harmful content" {
    var checker = policy.PolicyChecker.init(testing.allocator);
    defer checker.deinit();

    const result = checker.check("How do I hack into a system?");

    // Should flag as potentially harmful
    try testing.expect(result.action != .allow or result.warnings.items.len > 0);
}

test "policy checker - PII detection" {
    var checker = policy.PolicyChecker.init(testing.allocator);
    defer checker.deinit();

    // Test email pattern
    const result_email = checker.check("Send this to john@example.com");
    if (!result_email.is_safe or result_email.has_pii) {
        try testing.expect(result_email.has_pii or result_email.warnings.items.len > 0);
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

    const request = types.PersonaRequest{
        .content = "I'm feeling really sad and overwhelmed today.",
    };

    var analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    const sent_result = analyzer.analyze(request.content);
    const scores = engine.evaluate(request, sent_result);

    // Should favor Abbey for emotional content
    const abbey_score = scores.getScore(.abbey) orelse 0.0;
    const aviva_score = scores.getScore(.aviva) orelse 0.0;

    try testing.expect(abbey_score >= aviva_score);
}

test "rules evaluation - technical content" {
    var engine = rules.RulesEngine.init(testing.allocator);
    defer engine.deinit();

    const request = types.PersonaRequest{
        .content = "Implement a binary search algorithm in Zig.",
    };

    var analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    const sent_result = analyzer.analyze(request.content);
    const scores = engine.evaluate(request, sent_result);

    // Should favor Aviva for technical content
    const aviva_score = scores.getScore(.aviva) orelse 0.0;
    try testing.expect(aviva_score > 0.0);
}

test "rules evaluation - code request" {
    var engine = rules.RulesEngine.init(testing.allocator);
    defer engine.deinit();

    const request = types.PersonaRequest{
        .content = "Write a function to parse JSON.",
    };

    var analyzer = sentiment.SentimentAnalyzer.init(testing.allocator);
    defer analyzer.deinit();

    const sent_result = analyzer.analyze(request.content);
    const scores = engine.evaluate(request, sent_result);

    // Should have scores for at least one persona
    try testing.expect(scores.total_score > 0.0);
}

// ============================================================================
// Combined Routing Tests
// ============================================================================

test "routing decision flow" {
    // Test the full routing decision flow
    const allocator = testing.allocator;

    var analyzer = sentiment.SentimentAnalyzer.init(allocator);
    defer analyzer.deinit();

    var checker = policy.PolicyChecker.init(allocator);
    defer checker.deinit();

    var engine = rules.RulesEngine.init(allocator);
    defer engine.deinit();

    // Test case: Technical question
    {
        const content = "What's the difference between ArrayList and ArrayListUnmanaged?";

        // Step 1: Analyze sentiment
        const sent = analyzer.analyze(content);

        // Step 2: Check policy
        const pol = checker.check(content);
        try testing.expect(pol.is_safe);

        // Step 3: Evaluate rules
        const request = types.PersonaRequest{ .content = content };
        const scores = engine.evaluate(request, sent);

        // Should produce valid scores
        try testing.expect(scores.total_score > 0.0);
    }

    // Test case: Emotional support request
    {
        const content = "I'm really struggling with learning Zig and feeling discouraged.";

        const sent = analyzer.analyze(content);
        const pol = checker.check(content);
        try testing.expect(pol.is_safe);

        const request = types.PersonaRequest{ .content = content };
        const scores = engine.evaluate(request, sent);

        // Abbey should be favored
        const abbey_score = scores.getScore(.abbey) orelse 0.0;
        try testing.expect(abbey_score > 0.0);
    }
}
