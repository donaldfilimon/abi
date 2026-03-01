//! AI Evaluation Metrics Tests — BLEU, ROUGE, Perplexity
//!
//! Comprehensive correctness tests for the AI evaluation module.
//! Tests mathematical properties, edge cases, and known reference values.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const eval = if (build_options.enable_ai) abi.features.ai.eval else struct {};

// ============================================================================
// BLEU Score Tests
// ============================================================================

test "bleu: identical sentences score 1.0" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const result = try eval.computeBleu(
        allocator,
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox jumps over the lazy dog",
        4,
        .none,
    );

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.score, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.brevity_penalty, 0.0001);
    for (result.precisions) |p| {
        try std.testing.expectApproxEqAbs(@as(f64, 1.0), p, 0.0001);
    }
}

test "bleu: completely different sentences score near 0" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const result = try eval.computeBleu(
        allocator,
        "alpha beta gamma delta",
        "one two three four five six",
        4,
        .none,
    );

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.score, 0.001);
}

test "bleu: brevity penalty applied for short hypothesis" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const result = try eval.computeBleu(
        allocator,
        "the cat",
        "the cat sat on the mat by the door",
        4,
        .method2,
    );

    // BP < 1 when hypothesis shorter than reference
    try std.testing.expect(result.brevity_penalty < 1.0);
    try std.testing.expect(result.brevity_penalty > 0.0);
    try std.testing.expect(result.hypothesis_length < result.reference_length);
}

test "bleu: no brevity penalty when hypothesis is longer" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const result = try eval.computeBleu(
        allocator,
        "the cat sat on the mat by the door and the window",
        "the cat sat on the mat",
        4,
        .method2,
    );

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.brevity_penalty, 0.0001);
}

test "bleu: smoothing method2 handles zero n-gram counts" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Short sentence where higher n-grams have 0 matches
    const result = try eval.computeBleu(
        allocator,
        "hello world",
        "the quick brown fox",
        4,
        .method2,
    );

    // method2 adds 1 to numerator and denominator, so score > 0
    try std.testing.expect(result.score >= 0.0);
    try std.testing.expect(result.score < 1.0);
}

test "bleu: multi-reference picks best match" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const refs = [_][]const u8{
        "the cat is on the mat",
        "there is a cat on the mat",
        "the cat sat on the mat",
    };

    const result = try eval.computeBleuMultiRef(
        allocator,
        "the cat sat on the mat",
        &refs,
        4,
        .none,
    );

    // Perfect match with third reference
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.score, 0.0001);
}

// ============================================================================
// ROUGE Score Tests
// ============================================================================

test "rouge-1: perfect overlap yields f1=1.0" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const result = try eval.computeRouge(
        allocator,
        "the quick brown fox",
        "the quick brown fox",
        .rouge_1,
    );

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.f1, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.precision, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.recall, 0.0001);
}

test "rouge-2: measures bigram overlap" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const result = try eval.computeRouge(
        allocator,
        "the cat sat on",
        "the cat sat on the mat",
        .rouge_2,
    );

    // Hypothesis has 3 bigrams ("the cat", "cat sat", "sat on")
    // All appear in reference → precision = 1.0
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.precision, 0.0001);
    // Reference has 5 bigrams, 3 overlap → recall = 3/5 = 0.6
    try std.testing.expectApproxEqAbs(@as(f64, 0.6), result.recall, 0.0001);
}

test "rouge-l: measures longest common subsequence" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const result = try eval.computeRouge(
        allocator,
        "the fox quick brown",
        "the quick brown fox",
        .rouge_l,
    );

    // LCS is "the quick brown" (3 words) or "the fox" + subsequence
    // LCS("the fox quick brown", "the quick brown fox") = "the quick brown" = 3
    // Precision = 3/4 = 0.75, Recall = 3/4 = 0.75
    try std.testing.expect(result.f1 > 0.5);
    try std.testing.expect(result.f1 < 1.0);
}

test "rouge: empty hypothesis yields zero scores" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    const result = try eval.computeRouge(
        allocator,
        "",
        "the quick brown fox",
        .rouge_1,
    );

    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.f1, 0.0001);
}

test "rouge: precision and recall are complementary" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Short hypothesis, long reference: high precision, low recall
    const r1 = try eval.computeRouge(
        allocator,
        "the cat",
        "the cat sat on the mat by the door",
        .rouge_1,
    );

    // Long hypothesis, short reference: low precision, high recall
    const r2 = try eval.computeRouge(
        allocator,
        "the cat sat on the mat by the door",
        "the cat",
        .rouge_1,
    );

    // r1: precision should be high (both words in ref)
    try std.testing.expect(r1.precision >= r1.recall);
    // r2: recall should be high (ref words all in hyp)
    try std.testing.expect(r2.recall >= r2.precision);
}

// ============================================================================
// Perplexity Tests
// ============================================================================

test "perplexity: uniform distribution equals vocab size" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    // PPL = V for uniform distribution over V tokens
    const vocab_sizes = [_]f64{ 2, 10, 100, 1000, 50000 };

    for (vocab_sizes) |v| {
        const log_prob = @log(1.0 / v);
        const probs = [_]f64{ log_prob, log_prob, log_prob, log_prob, log_prob };
        const result = eval.computePerplexity(&probs);

        try std.testing.expectApproxEqAbs(v, result.perplexity, v * 0.001);
    }
}

test "perplexity: perfect prediction yields ppl=1" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    const log_probs = [_]f64{ @log(1.0), @log(1.0), @log(1.0) };
    const result = eval.computePerplexity(&log_probs);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.perplexity, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), result.cross_entropy, 0.0001);
}

test "perplexity: empty input yields infinity" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    const result = eval.computePerplexity(&.{});

    try std.testing.expect(std.math.isInf(result.perplexity));
    try std.testing.expectEqual(@as(usize, 0), result.num_tokens);
}

test "perplexity: lower is better (higher prob = lower ppl)" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    // Model A: 50% correct
    const log_probs_a = [_]f64{ @log(0.5), @log(0.5), @log(0.5) };
    const result_a = eval.computePerplexity(&log_probs_a);

    // Model B: 80% correct
    const log_probs_b = [_]f64{ @log(0.8), @log(0.8), @log(0.8) };
    const result_b = eval.computePerplexity(&log_probs_b);

    try std.testing.expect(result_b.perplexity < result_a.perplexity);
}

test "perplexity: bpc roundtrip conversion" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    const test_ppls = [_]f64{ 1, 2, 10, 100, 1000 };

    for (test_ppls) |ppl| {
        const bpc = eval.perplexityToBpc(ppl);
        const recovered = eval.perplexityFromBpc(bpc);
        try std.testing.expectApproxEqAbs(ppl, recovered, ppl * 0.0001);
    }
}

test "perplexity: cross-entropy conversion" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    // PPL = exp(CE), so CE = ln(PPL)
    const ce_values = [_]f64{ 0.5, 1.0, 2.0, 5.0 };

    for (ce_values) |ce| {
        const ppl = eval.perplexityFromCrossEntropy(ce);
        try std.testing.expectApproxEqAbs(@exp(ce), ppl, 0.0001);
    }
}

test "perplexity: aggregate weighted by token count" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    // Two results with different token counts
    const r1 = eval.PerplexityResult{
        .perplexity = 10,
        .avg_log_prob = -@log(10.0),
        .cross_entropy = @log(10.0),
        .num_tokens = 100,
    };
    const r2 = eval.PerplexityResult{
        .perplexity = 100,
        .avg_log_prob = -@log(100.0),
        .cross_entropy = @log(100.0),
        .num_tokens = 100,
    };
    const equal_weight = [_]eval.PerplexityResult{ r1, r2 };

    const agg = eval.aggregatePerplexity(&equal_weight);

    // With equal weights, aggregate PPL should be geometric mean
    try std.testing.expect(agg.perplexity > 10);
    try std.testing.expect(agg.perplexity < 100);
    try std.testing.expectEqual(@as(usize, 200), agg.num_tokens);
}

test "perplexity: from probabilities handles near-zero" {
    if (!build_options.enable_ai) return error.SkipZigTest;

    // Near-zero probabilities should not produce NaN
    const probs = [_]f64{ 0.001, 0.0001, 0.00001 };
    const result = eval.computePerplexityFromProbs(&probs);

    try std.testing.expect(!std.math.isNan(result.perplexity));
    try std.testing.expect(!std.math.isInf(result.perplexity));
    try std.testing.expect(result.perplexity > 1.0);
}

test "perplexity: windowed detects anomaly" {
    if (!build_options.enable_ai) return error.SkipZigTest;
    const allocator = std.testing.allocator;

    // Create sequence with a sudden drop in probability
    var log_probs: [20]f64 = undefined;
    for (0..10) |i| {
        log_probs[i] = @log(0.8); // High probability region
    }
    for (10..20) |i| {
        log_probs[i] = @log(0.01); // Low probability region (anomaly)
    }

    const windows = try eval.computeWindowedPerplexity(
        allocator,
        &log_probs,
        5,
    );
    defer allocator.free(windows);

    // Early windows should have low perplexity
    try std.testing.expect(windows[0].perplexity < 5.0);
    // Later windows should have high perplexity
    try std.testing.expect(windows[windows.len - 1].perplexity > 10.0);
}
