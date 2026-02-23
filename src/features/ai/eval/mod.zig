//! Evaluation framework for language models.
//!
//! Provides metrics for evaluating generation quality including BLEU, ROUGE,
//! perplexity, and custom evaluation functions.

const std = @import("std");
const bleu = @import("bleu.zig");
const rouge = @import("rouge.zig");
const perplexity = @import("perplexity.zig");
const metrics = @import("metrics.zig");

pub const BleuScore = bleu.BleuScore;
pub const computeBleu = bleu.computeBleu;
pub const computeBleuMultiRef = bleu.computeBleuMultiRef;

pub const RougeScore = rouge.RougeScore;
pub const RougeType = rouge.RougeType;
pub const computeRouge = rouge.computeRouge;
pub const computeRougeN = rouge.computeRougeN;
pub const computeRougeL = rouge.computeRougeL;

pub const PerplexityResult = perplexity.PerplexityResult;
pub const computePerplexity = perplexity.computePerplexity;

pub const TokenMetrics = metrics.TokenMetrics;
pub const TextStatistics = metrics.TextStatistics;
pub const computeF1 = metrics.computeF1;
pub const computeExactMatch = metrics.computeExactMatch;
pub const computeTokenMetrics = metrics.computeTokenMetrics;
pub const computeTextStatistics = metrics.computeTextStatistics;
pub const computeNormalizedExactMatch = metrics.computeNormalizedExactMatch;
pub const computeCER = metrics.computeCER;
pub const computeWER = metrics.computeWER;
pub const levenshteinDistance = metrics.levenshteinDistance;

// Perplexity utilities
pub const perplexityFromCrossEntropy = perplexity.perplexityFromCrossEntropy;
pub const perplexityFromBpc = perplexity.perplexityFromBpc;
pub const perplexityToBpc = perplexity.perplexityToBpc;
pub const aggregatePerplexity = perplexity.aggregatePerplexity;
pub const computeWindowedPerplexity = perplexity.computeWindowedPerplexity;
pub const computePerplexityFromProbs = perplexity.computePerplexityFromProbs;

// BLEU utilities
pub const SmoothingMethod = bleu.SmoothingMethod;

// Shared tokenizer
pub const tokenizer = @import("tokenizer.zig");
pub const tokenize = tokenizer.tokenize;

/// Evaluation configuration.
pub const EvalConfig = struct {
    /// Maximum n-gram size for BLEU.
    max_ngram: u32 = 4,
    /// Smoothing method for BLEU.
    bleu_smoothing: bleu.SmoothingMethod = .method1,
    /// ROUGE types to compute.
    rouge_types: []const RougeType = &[_]RougeType{ .rouge_1, .rouge_2, .rouge_l },
    /// Whether to lowercase for comparison.
    case_insensitive: bool = true,
    /// Enable token-level metrics.
    compute_token_metrics: bool = true,
};

/// Complete evaluation result.
pub const EvaluationResult = struct {
    /// BLEU score (0-1).
    bleu: ?BleuScore,
    /// ROUGE scores by type.
    rouge: ?[]RougeScore,
    /// Perplexity.
    perplexity: ?PerplexityResult,
    /// Exact match ratio (0-1).
    exact_match: f64,
    /// F1 score (0-1).
    f1: f64,
    /// Token-level statistics.
    token_metrics: ?TokenMetrics,
    /// Text statistics for hypothesis.
    hypothesis_stats: TextStatistics,
    /// Text statistics for reference.
    reference_stats: TextStatistics,

    pub fn deinit(self: *EvaluationResult, allocator: std.mem.Allocator) void {
        if (self.rouge) |r| {
            allocator.free(r);
        }
        self.* = undefined;
    }
};

/// Evaluation report for multiple samples.
pub const EvaluationReport = struct {
    /// Number of samples evaluated.
    num_samples: usize,
    /// Average BLEU score.
    avg_bleu: f64,
    /// Average ROUGE-1 F1.
    avg_rouge1_f1: f64,
    /// Average ROUGE-2 F1.
    avg_rouge2_f1: f64,
    /// Average ROUGE-L F1.
    avg_rougeL_f1: f64,
    /// Average perplexity.
    avg_perplexity: f64,
    /// Exact match ratio.
    exact_match_ratio: f64,
    /// Average F1 score.
    avg_f1: f64,
    /// Standard deviation of BLEU.
    std_bleu: f64,
    /// Minimum BLEU.
    min_bleu: f64,
    /// Maximum BLEU.
    max_bleu: f64,
};

/// Main evaluator interface.
pub const Evaluator = struct {
    allocator: std.mem.Allocator,
    config: EvalConfig,

    pub fn init(allocator: std.mem.Allocator, config: EvalConfig) Evaluator {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    /// Evaluate a single hypothesis against a reference.
    pub fn evaluate(
        self: *Evaluator,
        hypothesis: []const u8,
        reference: []const u8,
    ) !EvaluationResult {
        var result = EvaluationResult{
            .bleu = null,
            .rouge = null,
            .perplexity = null,
            .exact_match = 0,
            .f1 = 0,
            .token_metrics = null,
            .hypothesis_stats = metrics.computeTextStatistics(hypothesis),
            .reference_stats = metrics.computeTextStatistics(reference),
        };

        // Preprocess if case insensitive
        const hyp = if (self.config.case_insensitive)
            try toLower(self.allocator, hypothesis)
        else
            try self.allocator.dupe(u8, hypothesis);
        defer self.allocator.free(hyp);

        const ref = if (self.config.case_insensitive)
            try toLower(self.allocator, reference)
        else
            try self.allocator.dupe(u8, reference);
        defer self.allocator.free(ref);

        // Compute BLEU
        result.bleu = try computeBleu(
            self.allocator,
            hyp,
            ref,
            self.config.max_ngram,
            self.config.bleu_smoothing,
        );

        // Compute ROUGE
        var rouge_results = std.ArrayListUnmanaged(RougeScore).empty;
        errdefer rouge_results.deinit(self.allocator);

        for (self.config.rouge_types) |rouge_type| {
            const score = try computeRouge(self.allocator, hyp, ref, rouge_type);
            try rouge_results.append(self.allocator, score);
        }
        result.rouge = try rouge_results.toOwnedSlice(self.allocator);

        // Compute exact match and F1
        result.exact_match = computeExactMatch(hyp, ref);
        result.f1 = try computeF1(self.allocator, hyp, ref);

        // Compute token metrics if enabled
        if (self.config.compute_token_metrics) {
            result.token_metrics = try metrics.computeTokenMetrics(self.allocator, hyp, ref);
        }

        return result;
    }

    /// Evaluate multiple hypothesis-reference pairs.
    pub fn evaluateBatch(
        self: *Evaluator,
        hypotheses: []const []const u8,
        references: []const []const u8,
    ) !EvaluationReport {
        if (hypotheses.len != references.len) {
            return error.LengthMismatch;
        }
        if (hypotheses.len == 0) {
            return error.EmptyInput;
        }

        var bleu_scores = std.ArrayListUnmanaged(f64).empty;
        defer bleu_scores.deinit(self.allocator);

        var rouge1_scores = std.ArrayListUnmanaged(f64).empty;
        defer rouge1_scores.deinit(self.allocator);

        var rouge2_scores = std.ArrayListUnmanaged(f64).empty;
        defer rouge2_scores.deinit(self.allocator);

        var rougeL_scores = std.ArrayListUnmanaged(f64).empty;
        defer rougeL_scores.deinit(self.allocator);

        var f1_scores = std.ArrayListUnmanaged(f64).empty;
        defer f1_scores.deinit(self.allocator);

        var exact_matches: usize = 0;

        for (hypotheses, references) |hyp, ref| {
            var result = try self.evaluate(hyp, ref);
            defer result.deinit(self.allocator);

            if (result.bleu) |b| {
                try bleu_scores.append(self.allocator, b.score);
            }

            if (result.rouge) |r| {
                for (r) |score| {
                    switch (score.rouge_type) {
                        .rouge_1 => try rouge1_scores.append(self.allocator, score.f1),
                        .rouge_2 => try rouge2_scores.append(self.allocator, score.f1),
                        .rouge_l => try rougeL_scores.append(self.allocator, score.f1),
                        else => {},
                    }
                }
            }

            try f1_scores.append(self.allocator, result.f1);
            if (result.exact_match > 0.5) exact_matches += 1;
        }

        const n = hypotheses.len;
        const n_f = @as(f64, @floatFromInt(n));

        return .{
            .num_samples = n,
            .avg_bleu = mean(bleu_scores.items),
            .avg_rouge1_f1 = mean(rouge1_scores.items),
            .avg_rouge2_f1 = mean(rouge2_scores.items),
            .avg_rougeL_f1 = mean(rougeL_scores.items),
            .avg_perplexity = 0, // Not computed in batch
            .exact_match_ratio = @as(f64, @floatFromInt(exact_matches)) / n_f,
            .avg_f1 = mean(f1_scores.items),
            .std_bleu = standardDeviation(bleu_scores.items),
            .min_bleu = if (bleu_scores.items.len > 0) min(bleu_scores.items) else 0,
            .max_bleu = if (bleu_scores.items.len > 0) max(bleu_scores.items) else 0,
        };
    }
};

const string_utils = @import("../../../services/shared/utils.zig");
const toLower = string_utils.string.toLowerAscii;

fn mean(values: []const f64) f64 {
    if (values.len == 0) return 0;
    var sum: f64 = 0;
    for (values) |v| sum += v;
    return sum / @as(f64, @floatFromInt(values.len));
}

fn standardDeviation(values: []const f64) f64 {
    if (values.len < 2) return 0;
    const avg = mean(values);
    var sum_sq: f64 = 0;
    for (values) |v| {
        const diff = v - avg;
        sum_sq += diff * diff;
    }
    return @sqrt(sum_sq / @as(f64, @floatFromInt(values.len - 1)));
}

fn min(values: []const f64) f64 {
    var m = values[0];
    for (values[1..]) |v| {
        if (v < m) m = v;
    }
    return m;
}

fn max(values: []const f64) f64 {
    var m = values[0];
    for (values[1..]) |v| {
        if (v > m) m = v;
    }
    return m;
}

test "evaluator initialization" {
    const allocator = std.testing.allocator;
    const evaluator = Evaluator.init(allocator, .{});
    _ = evaluator;
}

test "single evaluation" {
    const allocator = std.testing.allocator;
    var evaluator = Evaluator.init(allocator, .{});

    var result = try evaluator.evaluate(
        "the cat sat on the mat",
        "the cat sat on the mat",
    );
    defer result.deinit(allocator);

    // Perfect match should have high scores
    try std.testing.expect(result.exact_match > 0.5);
    try std.testing.expect(result.f1 > 0.9);
}

test "batch evaluation" {
    const allocator = std.testing.allocator;
    var evaluator = Evaluator.init(allocator, .{});

    const hypotheses = [_][]const u8{
        "the cat sat on the mat",
        "hello world",
        "foo bar baz",
    };
    const references = [_][]const u8{
        "the cat sat on the mat",
        "hello there world",
        "completely different text",
    };

    const report = try evaluator.evaluateBatch(&hypotheses, &references);

    try std.testing.expectEqual(@as(usize, 3), report.num_samples);
    try std.testing.expect(report.avg_bleu > 0);
    try std.testing.expect(report.avg_f1 > 0);
    try std.testing.expect(report.exact_match_ratio > 0); // At least one exact match
}

test "batch evaluation length mismatch" {
    const allocator = std.testing.allocator;
    var evaluator = Evaluator.init(allocator, .{});

    const hypotheses = [_][]const u8{ "a", "b" };
    const references = [_][]const u8{"a"};

    const result = evaluator.evaluateBatch(&hypotheses, &references);
    try std.testing.expectError(error.LengthMismatch, result);
}

test "batch evaluation empty" {
    const allocator = std.testing.allocator;
    var evaluator = Evaluator.init(allocator, .{});

    const hypotheses = [_][]const u8{};
    const references = [_][]const u8{};

    const result = evaluator.evaluateBatch(&hypotheses, &references);
    try std.testing.expectError(error.EmptyInput, result);
}

test {
    _ = bleu;
    _ = rouge;
    _ = perplexity;
    _ = metrics;
    _ = tokenizer;
}

test {
    std.testing.refAllDecls(@This());
}
