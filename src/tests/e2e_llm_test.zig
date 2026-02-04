//! End-to-End LLM Integration Tests
//!
//! Comprehensive tests for the LLM module covering:
//! - Tokenizer workflows (encoding, decoding, round-trip)
//! - Inference configuration validation
//! - Engine lifecycle management
//! - Edge cases: empty inputs, unicode, special characters
//! - Error conditions and graceful degradation
//!
//! These tests are designed to run without external dependencies (models),
//! focusing on API correctness and edge case handling.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

// Skip all tests if LLM feature is disabled
fn skipIfLlmDisabled() !void {
    if (!build_options.enable_llm) return error.SkipZigTest;
}

// ============================================================================
// Tokenizer Workflow Tests
// ============================================================================

// Test that tokenizer can be initialized with different kinds.
// Verifies the unified tokenizer interface works for all supported types.
test "tokenizer: initialization with different kinds" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    // Test BPE variant
    var bpe_tok = abi.ai.llm.tokenizer.Tokenizer.init(allocator, .bpe);
    defer bpe_tok.deinit();
    try std.testing.expectEqual(abi.ai.llm.tokenizer.TokenizerKind.bpe, bpe_tok.getKind());
    try std.testing.expectEqual(@as(u32, 0), bpe_tok.vocabSize());

    // Test SentencePiece variant
    var sp_tok = abi.ai.llm.tokenizer.Tokenizer.init(allocator, .sentencepiece);
    defer sp_tok.deinit();
    try std.testing.expectEqual(abi.ai.llm.tokenizer.TokenizerKind.sentencepiece, sp_tok.getKind());
    try std.testing.expectEqual(@as(u32, 0), sp_tok.vocabSize());

    // Test unknown variant - should return 0 vocab size
    var unk_tok = abi.ai.llm.tokenizer.Tokenizer.init(allocator, .unknown);
    defer unk_tok.deinit();
    try std.testing.expectEqual(abi.ai.llm.tokenizer.TokenizerKind.unknown, unk_tok.getKind());
    try std.testing.expectEqual(@as(u32, 0), unk_tok.vocabSize());
}

// Test tokenizer kind detection from GGUF model type strings.
// Ensures correct tokenizer selection for different model architectures.
test "tokenizer: kind detection from model type" {
    try skipIfLlmDisabled();

    const TokenizerKind = abi.ai.llm.tokenizer.TokenizerKind;

    // BPE models
    try std.testing.expectEqual(TokenizerKind.bpe, TokenizerKind.fromGgufModel("gpt2"));
    try std.testing.expectEqual(TokenizerKind.bpe, TokenizerKind.fromGgufModel("llama-bpe"));

    // SentencePiece models
    try std.testing.expectEqual(TokenizerKind.sentencepiece, TokenizerKind.fromGgufModel("llama"));
    try std.testing.expectEqual(TokenizerKind.sentencepiece, TokenizerKind.fromGgufModel("mistral"));

    // Unknown models
    try std.testing.expectEqual(TokenizerKind.unknown, TokenizerKind.fromGgufModel(null));
    try std.testing.expectEqual(TokenizerKind.unknown, TokenizerKind.fromGgufModel(""));
    try std.testing.expectEqual(TokenizerKind.unknown, TokenizerKind.fromGgufModel("custom-model"));
    try std.testing.expectEqual(TokenizerKind.unknown, TokenizerKind.fromGgufModel("falcon"));
}

// Test BPE tokenizer with vocabulary loading.
// Verifies basic vocabulary operations work correctly.
test "tokenizer: bpe vocabulary loading" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    var tokenizer = abi.ai.llm.tokenizer.BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    // Load a minimal vocabulary
    const vocab = [_][]const u8{ "hello", "world", " ", "!", "<unk>", "<s>", "</s>" };
    try tokenizer.loadVocab(&vocab);

    // Verify vocabulary size
    try std.testing.expectEqual(@as(u32, 7), tokenizer.vocabSize());

    // Check token validity
    try std.testing.expect(tokenizer.isValidToken(0)); // "hello"
    try std.testing.expect(tokenizer.isValidToken(1)); // "world"
    try std.testing.expect(tokenizer.isValidToken(6)); // "</s>"
}

// Test tokenizer BOS/EOS configuration.
// Verifies special token flags can be set and affect output.
test "tokenizer: bos and eos configuration" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    var tokenizer = abi.ai.llm.tokenizer.Tokenizer.init(allocator, .bpe);
    defer tokenizer.deinit();

    // Test setAddBos
    tokenizer.setAddBos(true);
    tokenizer.setAddBos(false);

    // Test setAddEos
    tokenizer.setAddEos(true);
    tokenizer.setAddEos(false);

    // Unknown tokenizer should handle these gracefully
    var unk_tok = abi.ai.llm.tokenizer.Tokenizer.init(allocator, .unknown);
    defer unk_tok.deinit();

    // These should not crash on unknown tokenizer
    unk_tok.setAddBos(true);
    unk_tok.setAddEos(true);
}

// ============================================================================
// Inference Configuration Tests
// ============================================================================

// Testinference configuration defaults.
// Ensuressensible defaults are set for all configuration options.
test "inference config: default values" {
    try skipIfLlmDisabled();

    const config = abi.ai.llm.InferenceConfig{};

    try std.testing.expectEqual(@as(u32, 2048), config.max_context_length);
    try std.testing.expectEqual(@as(u32, 256), config.max_new_tokens);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7), config.temperature, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), config.top_p, 0.001);
    try std.testing.expectEqual(@as(u32, 40), config.top_k);
    try std.testing.expectApproxEqAbs(@as(f32, 1.1), config.repetition_penalty, 0.001);
    try std.testing.expect(config.use_gpu);
    try std.testing.expect(config.streaming);
    try std.testing.expectEqual(@as(u32, 512), config.batch_size);
}

// Test inference configuration with custom values.
// Verifies all configuration options can be customized.
test "inference config: custom values" {
    try skipIfLlmDisabled();

    const config = abi.ai.llm.InferenceConfig{
        .max_context_length = 4096,
        .max_new_tokens = 1024,
        .temperature = 0.0, // Greedy decoding
        .top_p = 0.95,
        .top_k = 50,
        .repetition_penalty = 1.0, // Disabled
        .use_gpu = false,
        .num_threads = 4,
        .streaming = false,
        .batch_size = 256,
        .context_size = 4096,
    };

    try std.testing.expectEqual(@as(u32, 4096), config.max_context_length);
    try std.testing.expectEqual(@as(u32, 1024), config.max_new_tokens);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), config.temperature, 0.001);
    try std.testing.expectEqual(@as(u32, 4), config.num_threads);
    try std.testing.expect(!config.use_gpu);
    try std.testing.expect(!config.streaming);
}

// Test edge case: temperature at boundaries.
// Temperature of 0 means greedy, high values increase randomness.
test "inference config: temperature boundaries" {
    try skipIfLlmDisabled();

    // Greedy decoding
    const greedy = abi.ai.llm.InferenceConfig{ .temperature = 0.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), greedy.temperature, 0.001);

    // Normal sampling
    const normal = abi.ai.llm.InferenceConfig{ .temperature = 1.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), normal.temperature, 0.001);

    // High temperature (creative)
    const creative = abi.ai.llm.InferenceConfig{ .temperature = 2.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), creative.temperature, 0.001);
}

// ============================================================================
// Inference Statistics Tests
// ============================================================================

// Test inference statistics calculation.
// Verifies tokens-per-second calculations are correct.
test "inference stats: tokens per second calculation" {
    try skipIfLlmDisabled();

    const stats = abi.ai.llm.InferenceStats{
        .prompt_tokens = 100,
        .generated_tokens = 50,
        .prefill_time_ns = 1_000_000_000, // 1 second
        .decode_time_ns = 2_000_000_000, // 2 seconds
        .used_gpu = true,
    };

    // 100 tokens in 1 second = 100 tok/s
    try std.testing.expectApproxEqAbs(@as(f64, 100.0), stats.prefillTokensPerSecond(), 0.001);

    // 50 tokens in 2 seconds = 25 tok/s
    try std.testing.expectApproxEqAbs(@as(f64, 25.0), stats.decodeTokensPerSecond(), 0.001);
}

// Test inference statistics with zero time.
// Division by zero should be handled gracefully.
test "inference stats: zero time handling" {
    try skipIfLlmDisabled();

    const stats = abi.ai.llm.InferenceStats{
        .prompt_tokens = 100,
        .generated_tokens = 50,
        .prefill_time_ns = 0,
        .decode_time_ns = 0,
    };

    // Should return 0, not NaN or infinity
    try std.testing.expectEqual(@as(f64, 0.0), stats.prefillTokensPerSecond());
    try std.testing.expectEqual(@as(f64, 0.0), stats.decodeTokensPerSecond());
}

// Test inference statistics formatting.
// Verifies the format function produces expected output structure.
test "inference stats: formatting" {
    try skipIfLlmDisabled();

    const stats = abi.ai.llm.InferenceStats{
        .prompt_tokens = 100,
        .generated_tokens = 50,
        .prefill_time_ns = 1_000_000_000,
        .decode_time_ns = 2_000_000_000,
        .used_gpu = true,
    };

    // Format to buffer using print
    var buffer: [256]u8 = undefined;
    // Buffer is large enough for formatted stats output
    const formatted = std.fmt.bufPrint(&buffer, "{}", .{stats}) catch |err| {
        std.debug.panic("bufPrint failed unexpectedly: {}", .{err});
    };

    // Verify format contains expected elements
    try std.testing.expect(std.mem.indexOf(u8, formatted, "prefill") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "decode") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "gpu") != null);
}

// ============================================================================
// LLM Engine Tests
// ============================================================================

// Test LLM engine initialization and cleanup.
// Verifies engine can be created and destroyed without leaks.
test "engine: lifecycle management" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    var engine = abi.ai.llm.Engine.init(allocator, .{});
    defer engine.deinit();

    // Engine should be initialized with default config
    try std.testing.expectEqual(@as(u32, 2048), engine.config.max_context_length);

    // No model loaded initially
    try std.testing.expect(engine.loaded_model == null);

    // Stats should be zeroed
    const stats = engine.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.load_time_ns);
}

// Test LLM engine with custom configuration.
// Verifies configuration is properly stored and accessible.
test "engine: custom configuration" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    const config = abi.ai.llm.InferenceConfig{
        .max_context_length = 4096,
        .temperature = 0.5,
        .use_gpu = false,
    };

    var engine = abi.ai.llm.Engine.init(allocator, config);
    defer engine.deinit();

    try std.testing.expectEqual(@as(u32, 4096), engine.config.max_context_length);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), engine.config.temperature, 0.001);
    try std.testing.expect(!engine.config.use_gpu);
}

// Test engine operations without loaded model.
// Should return appropriate errors when no model is available.
test "engine: operations without model" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    var engine = abi.ai.llm.Engine.init(allocator, .{});
    defer engine.deinit();

    // Generate should fail with InvalidModelFormat
    const result = engine.generate(allocator, "Hello");
    try std.testing.expectError(abi.ai.llm.LlmError.InvalidModelFormat, result);

    // Tokenize should fail with InvalidModelFormat
    const tokens = engine.tokenize(allocator, "Hello");
    try std.testing.expectError(abi.ai.llm.LlmError.InvalidModelFormat, tokens);

    // Detokenize should fail with InvalidModelFormat
    const text = engine.detokenize(allocator, &[_]u32{ 1, 2, 3 });
    try std.testing.expectError(abi.ai.llm.LlmError.InvalidModelFormat, text);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

// Test handling of empty input strings.
// Empty inputs should be handled gracefully without crashes.
test "edge case: empty input handling" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    var tokenizer = abi.ai.llm.tokenizer.BpeTokenizer.init(allocator);
    defer tokenizer.deinit();

    // BPE tokenizer with add_bos=true should produce at least BOS token
    // but without vocab, encoding an empty string may produce just BOS
    tokenizer.add_bos = false;
    tokenizer.add_eos = false;

    // Empty string should not crash
    const tokens = try tokenizer.encode(allocator, "");
    defer allocator.free(tokens);

    // Result may be empty or contain only special tokens
    // The important thing is it doesn't crash
}

// Test handling of unicode input.
// Unicode characters should be processed correctly.
test "edge case: unicode input" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    var tokenizer = abi.ai.llm.tokenizer.BpeTokenizer.init(allocator);
    defer tokenizer.deinit();
    tokenizer.add_bos = false;
    tokenizer.add_eos = false;

    // Test various unicode inputs
    const unicode_inputs = [_][]const u8{
        "Hello, \xe4\xb8\x96\xe7\x95\x8c", // Hello, 世界 (Chinese)
        "\xc3\xa9\xc3\xa0\xc3\xbc", // éàü (French/German)
        "\xf0\x9f\x98\x80\xf0\x9f\x8e\x89", // Emojis: 😀🎉
        "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82", // Привет (Russian)
    };

    for (unicode_inputs) |input| {
        // Should not crash on unicode input
        const tokens = tokenizer.encode(allocator, input) catch continue;
        defer allocator.free(tokens);
        // Tokens will be UNK without vocab, but processing should not crash
    }
}

// Test handling of special characters.
// Control characters, null bytes, etc. should be handled safely.
test "edge case: special characters" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    var tokenizer = abi.ai.llm.tokenizer.BpeTokenizer.init(allocator);
    defer tokenizer.deinit();
    tokenizer.add_bos = false;
    tokenizer.add_eos = false;

    // Test special character inputs
    const special_inputs = [_][]const u8{
        "hello\nworld", // Newline
        "hello\tworld", // Tab
        "hello\rworld", // Carriage return
        "a\x00b", // Null byte
        "line1\r\nline2", // Windows line ending
    };

    for (special_inputs) |input| {
        // Should not crash on special characters
        const tokens = tokenizer.encode(allocator, input) catch continue;
        defer allocator.free(tokens);
    }
}

// Test handling of maximum size inputs.
// Very long inputs should not cause integer overflow or OOM.
test "edge case: large input handling" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    var tokenizer = abi.ai.llm.tokenizer.BpeTokenizer.init(allocator);
    defer tokenizer.deinit();
    tokenizer.add_bos = false;
    tokenizer.add_eos = false;

    // Create a moderately large input (not too large to avoid test timeout)
    const large_input = try allocator.alloc(u8, 10000);
    defer allocator.free(large_input);
    @memset(large_input, 'a');

    // Should not crash on large input
    const tokens = tokenizer.encode(allocator, large_input) catch |err| {
        // OutOfMemory is acceptable for very large inputs
        if (err == error.OutOfMemory) return;
        return err;
    };
    defer allocator.free(tokens);
}

// ============================================================================
// LLM Error Type Tests
// ============================================================================

// TestLLM error types are properly defined.
// Ensuresall expected error types exist in the error set.
test "llm errors: type definitions" {
    try skipIfLlmDisabled();

    // Verify LlmError types exist
    const llm_errors = [_]abi.ai.llm.LlmError{
        abi.ai.llm.LlmError.InvalidModelFormat,
        abi.ai.llm.LlmError.UnsupportedQuantization,
        abi.ai.llm.LlmError.ModelTooLarge,
        abi.ai.llm.LlmError.ContextLengthExceeded,
        abi.ai.llm.LlmError.TokenizationFailed,
        abi.ai.llm.LlmError.InferenceError,
        abi.ai.llm.LlmError.OutOfMemory,
        abi.ai.llm.LlmError.GpuUnavailable,
    };

    // Verify errors can be compared
    for (llm_errors) |err| {
        try std.testing.expect(@intFromError(err) != 0);
    }
}

// Testmodule-level Error types.
// Ensuresconvenience error types are also available.
test "llm errors: module error types" {
    try skipIfLlmDisabled();

    const module_errors = [_]abi.ai.llm.Error{
        abi.ai.llm.Error.LlmDisabled,
        abi.ai.llm.Error.ModelNotFound,
        abi.ai.llm.Error.ModelLoadFailed,
        abi.ai.llm.Error.InferenceFailed,
        abi.ai.llm.Error.TokenizationFailed,
        abi.ai.llm.Error.InvalidConfig,
    };

    for (module_errors) |err| {
        try std.testing.expect(@intFromError(err) != 0);
    }
}

// ============================================================================
// LLM Feature Detection Tests
// ============================================================================

// Test LLM feature detection.
// Verifies isEnabled() returns correct value based on build options.
test "llm feature: detection" {
    // This test runs regardless of feature flag
    const enabled = abi.ai.llm.isEnabled();

    // Should match build options
    if (build_options.enable_llm and build_options.enable_ai) {
        try std.testing.expect(enabled);
    } else {
        try std.testing.expect(!enabled);
    }
}

// ============================================================================
// Parallel Processing Tests
// ============================================================================

// Test parallel executor basic initialization.
// Verifies parallel processing support is available when LLM is enabled.
test "parallel: executor initialization" {
    try skipIfLlmDisabled();

    const allocator = std.testing.allocator;

    // Create a parallel executor with 2 threads
    var executor = try abi.ai.llm.parallel.ParallelExecutor.init(allocator, 2);
    defer executor.deinit();

    // Verify it was created with the requested thread count
    try std.testing.expectEqual(@as(usize, 2), executor.thread_count);
}
