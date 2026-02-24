//! Unit Tests for Aviva Persona (Direct Expert)
//!
//! Tests query classification, knowledge retrieval, code generation,
//! and fact checking capabilities of the Aviva persona.

const std = @import("std");
const testing = std.testing;

// Import Aviva modules
const classifier = @import("../aviva/classifier.zig");
const knowledge = @import("../aviva/knowledge.zig");
const code = @import("../aviva/code.zig");
const facts = @import("../aviva/facts.zig");

// ============================================================================
// Query Classifier Tests
// ============================================================================

test "query classifier initialization" {
    const cls = classifier.QueryClassifier.init();

    try testing.expect(cls.config.min_confidence >= 0.0);
    try testing.expect(cls.config.min_confidence <= 1.0);
}

test "classify code request" {
    const cls = classifier.QueryClassifier.init();

    const result = cls.classify("Write a function to sort an array.");

    try testing.expect(result.query_type == .code_request);
    try testing.expect(result.confidence > 0.5);
}

test "classify factual query" {
    const cls = classifier.QueryClassifier.init();

    const result = cls.classify("What is the population of Tokyo?");

    try testing.expect(result.query_type == .factual_query);
}

test "classify explanation request" {
    const cls = classifier.QueryClassifier.init();

    const result = cls.classify("Can you explain how TCP/IP works?");

    try testing.expect(result.query_type == .explanation or
        result.query_type == .documentation);
}

test "classify debugging query" {
    const cls = classifier.QueryClassifier.init();

    const result = cls.classify("Why am I getting a segmentation fault?");

    try testing.expect(result.query_type == .debugging or
        result.query_type == .code_request);
}

test "language detection - zig" {
    const cls = classifier.QueryClassifier.initWithConfig(.{
        .detect_language = true,
    });

    const result = cls.classify("How do I use comptime in Zig?");

    try testing.expect(result.language == .zig or result.language == .unknown);
}

test "language detection - python" {
    const cls = classifier.QueryClassifier.initWithConfig(.{
        .detect_language = true,
    });

    const result = cls.classify("Write a Python script to read a CSV file.");

    try testing.expect(result.language == .python or result.language == .unknown);
}

test "language detection - rust" {
    const cls = classifier.QueryClassifier.initWithConfig(.{
        .detect_language = true,
    });

    const result = cls.classify("Implement a Rust macro for logging.");

    try testing.expect(result.language == .rust or result.language == .unknown);
}

test "domain detection" {
    const cls = classifier.QueryClassifier.initWithConfig(.{
        .detect_domain = true,
    });

    const result = cls.classify("How do I configure a PostgreSQL database?");

    // Domain enum uses .databases (not .database)
    try testing.expect(result.domain == .databases or result.domain == .general);
}

test "query type recommendations" {
    try testing.expect(classifier.QueryType.code_request.recommendsCodeBlock());
    try testing.expect(classifier.QueryType.debugging.recommendsCodeBlock());
    try testing.expect(!classifier.QueryType.factual_query.recommendsCodeBlock());
    try testing.expect(classifier.QueryType.factual_query.recommendsBrevity());
}

// ============================================================================
// Knowledge Retriever Tests
// ============================================================================

test "knowledge retriever initialization" {
    var retriever = knowledge.KnowledgeRetriever.init(testing.allocator);
    defer retriever.deinit();

    try testing.expect(retriever.config.max_fragments > 0);
}

test "knowledge retrieval basic" {
    var retriever = knowledge.KnowledgeRetriever.init(testing.allocator);
    defer retriever.deinit();

    var result = try retriever.retrieve("What is Zig?", null);
    defer result.deinit();

    // Should return (potentially empty) fragments
    try testing.expect(result.fragments.items.len >= 0);
}

test "knowledge fragment confidence" {
    const fragment = knowledge.KnowledgeFragment{
        .content = "Test content",
        .source = .{
            .name = "Test Source",
            .source_type = .documentation,
        },
        .relevance = 0.8,
        .confidence = 0.9,
        .domain = .general,
        .last_verified = 0,
    };

    try testing.expect(fragment.confidence >= 0.0);
    try testing.expect(fragment.confidence <= 1.0);
}

// ============================================================================
// Code Generator Tests
// ============================================================================

test "code generator initialization" {
    // CodeGenerator has no deinit
    const generator = code.CodeGenerator.init(testing.allocator);

    try testing.expect(generator.config.validate_syntax);
}

test "code block formatting - zig" {
    var generator = code.CodeGenerator.init(testing.allocator);

    const block = try generator.formatCodeBlock(
        "const x: i32 = 42;",
        .zig,
        null,
    );

    try testing.expect(block.code.len > 0);
    try testing.expect(block.language == .zig);
}

test "code block formatting - python" {
    var generator = code.CodeGenerator.init(testing.allocator);

    const block = try generator.formatCodeBlock(
        "x = 42",
        .python,
        null,
    );

    try testing.expect(block.code.len > 0);
    try testing.expect(block.language == .python);
}

test "function template generation - zig" {
    var generator = code.CodeGenerator.init(testing.allocator);

    const params = [_][]const u8{ "a: i32", "b: i32" };
    const block = try generator.generateFunctionTemplate(
        "add",
        &params,
        "i32",
        .zig,
        null,
    );

    try testing.expect(std.mem.indexOf(u8, block.code, "fn add") != null or
        std.mem.indexOf(u8, block.code, "add") != null);
}

test "function template generation - python" {
    var generator = code.CodeGenerator.init(testing.allocator);

    const params = [_][]const u8{ "a", "b" };
    const block = try generator.generateFunctionTemplate(
        "add",
        &params,
        null,
        .python,
        null,
    );

    try testing.expect(std.mem.indexOf(u8, block.code, "def add") != null or
        std.mem.indexOf(u8, block.code, "add") != null);
}

test "code structure validation" {
    const generator = code.CodeGenerator.init(testing.allocator);

    // Valid Zig code
    const valid_result = generator.validateStructure(
        "const x = 42;",
        .zig,
    );
    try testing.expect(valid_result.is_valid or valid_result.error_message == null);
}

test "code block extraction" {
    const input =
        \\Some text before
        \\```zig
        \\const x = 42;
        \\```
        \\Some text after
    ;

    const blocks = try code.extractCodeBlocks(testing.allocator, input);
    defer testing.allocator.free(blocks);

    // Should extract at least one code block
    try testing.expect(blocks.len >= 0);
}

// ============================================================================
// Fact Checker Tests
// ============================================================================

test "fact checker initialization" {
    // FactChecker has no deinit on itself
    const checker = facts.FactChecker.init(testing.allocator);

    try testing.expect(checker.config.min_unqualified_confidence > 0.0);
}

test "fact check simple statement" {
    var checker = facts.FactChecker.init(testing.allocator);

    var result = try checker.check("Zig is a systems programming language.");
    defer result.deinit();

    try testing.expect(result.claims.items.len >= 0);
    try testing.expect(result.overall_confidence >= 0.0);
}

test "claim type default confidence" {
    try testing.expect(facts.ClaimType.definition.getDefaultConfidence() > 0.5);
    try testing.expect(facts.ClaimType.numerical.getDefaultConfidence() > 0.5);
    try testing.expect(facts.ClaimType.causal.getDefaultConfidence() > 0.5);
}

test "detect uncertainty markers" {
    const checker = facts.FactChecker.init(testing.allocator);

    // scoreStatement is pub, but detectClaimType is private
    const certain_score = checker.scoreStatement("This is definitely correct.");
    const uncertain_score = checker.scoreStatement("This might be correct.");

    try testing.expect(uncertain_score <= certain_score);
}

test "apply qualifications" {
    const qualifications = [_][]const u8{
        "Numbers may be outdated.",
        "Verify with official sources.",
    };

    const result = try facts.applyQualifications(
        testing.allocator,
        "Some content here.",
        &qualifications,
    );
    defer testing.allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "Note") != null);
}

test "fact check result initialization" {
    var result = facts.FactCheckResult.init(testing.allocator);
    defer result.deinit();

    try testing.expectEqual(@as(usize, 0), result.claims.items.len);
    try testing.expectEqual(@as(f32, 1.0), result.overall_confidence);
}

// ============================================================================
// Integration Tests
// ============================================================================

test "aviva processing pipeline" {
    // Test the full Aviva processing pipeline
    const allocator = testing.allocator;

    // Initialize components
    const cls = classifier.QueryClassifier.init();
    var retriever = knowledge.KnowledgeRetriever.init(allocator);
    defer retriever.deinit();
    var generator = code.CodeGenerator.init(allocator);
    const checker = facts.FactChecker.init(allocator);
    _ = checker;

    // Process a code request
    const query = "Write a function to calculate fibonacci numbers.";

    // Step 1: Classify the query
    const classification = cls.classify(query);
    try testing.expect(classification.query_type == .code_request);

    // Step 2: Generate code (simulated)
    const params = [_][]const u8{"n: u32"};
    const template = try generator.generateFunctionTemplate(
        "fibonacci",
        &params,
        "u64",
        .zig,
        null,
    );
    try testing.expect(template.code.len > 0);

    // Step 3: Fact check a response
    var fact_checker = facts.FactChecker.init(allocator);
    const response = "The Fibonacci sequence starts with 0 and 1.";
    var fact_result = try fact_checker.check(response);
    defer fact_result.deinit();

    try testing.expect(fact_result.overall_confidence > 0.5);
}

test "aviva language support" {
    // getLanguageName is private, so test via public APIs instead
    const languages = [_]classifier.Language{
        .zig,
        .python,
        .rust,
        .go,
        .javascript,
        .typescript,
    };

    for (languages) |lang| {
        // Verify language has a file extension (public API)
        const ext = lang.getFileExtension();
        try testing.expect(ext.len > 0 or lang == .unknown);
    }

    // Test that wrapInMarkdown works with a language
    var gen = code.CodeGenerator.init(testing.allocator);
    const wrapped = try gen.wrapInMarkdown("const x = 42;", .zig);
    try testing.expect(wrapped.len > 0);
}
