//! Stub implementation for evaluation framework when AI features are disabled.

const std = @import("std");

/// Stub token metrics.
pub const TokenMetrics = struct {
    true_positives: usize = 0,
    false_positives: usize = 0,
    false_negatives: usize = 0,
    precision: f64 = 0,
    recall: f64 = 0,
    f1: f64 = 0,
};

/// Stub text statistics.
pub const TextStatistics = struct {
    char_count: usize = 0,
    word_count: usize = 0,
    sentence_count: usize = 0,
    avg_word_length: f64 = 0,
    unique_words: usize = 0,
    type_token_ratio: f64 = 0,
};

/// Stub BLEU score.
pub const BleuScore = struct {
    score: f64 = 0,
    precisions: [4]f64 = .{ 0, 0, 0, 0 },
    brevity_penalty: f64 = 0,
    hypothesis_length: usize = 0,
    reference_length: usize = 0,

    pub fn format(
        self: BleuScore,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = self;
        _ = fmt;
        _ = options;
        try writer.writeAll("BLEU: disabled");
    }
};

/// Stub ROUGE type.
pub const RougeType = enum {
    rouge_1,
    rouge_2,
    rouge_3,
    rouge_l,
};

/// Stub ROUGE score.
pub const RougeScore = struct {
    rouge_type: RougeType = .rouge_1,
    precision: f64 = 0,
    recall: f64 = 0,
    f1: f64 = 0,

    pub fn format(
        self: RougeScore,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = self;
        _ = fmt;
        _ = options;
        try writer.writeAll("ROUGE: disabled");
    }
};

/// Stub perplexity result.
pub const PerplexityResult = struct {
    perplexity: f64 = 0,
    avg_log_prob: f64 = 0,
    cross_entropy: f64 = 0,
    num_tokens: usize = 0,

    pub fn format(
        self: PerplexityResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = self;
        _ = fmt;
        _ = options;
        try writer.writeAll("Perplexity: disabled");
    }
};

/// Stub smoothing method.
const SmoothingMethod = enum {
    none,
    method1,
    method2,
    method3,
};

/// Stub evaluation config.
pub const EvalConfig = struct {
    max_ngram: u32 = 4,
    bleu_smoothing: SmoothingMethod = .method1,
    rouge_types: []const RougeType = &[_]RougeType{ .rouge_1, .rouge_2, .rouge_l },
    case_insensitive: bool = true,
    compute_token_metrics: bool = true,
};

/// Stub evaluation result.
pub const EvaluationResult = struct {
    bleu: ?BleuScore = null,
    rouge: ?[]RougeScore = null,
    perplexity: ?PerplexityResult = null,
    exact_match: f64 = 0,
    f1: f64 = 0,
    token_metrics: ?TokenMetrics = null,
    hypothesis_stats: TextStatistics = .{},
    reference_stats: TextStatistics = .{},

    pub fn deinit(self: *EvaluationResult, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.* = undefined;
    }
};

/// Stub evaluation report.
pub const EvaluationReport = struct {
    num_samples: usize = 0,
    avg_bleu: f64 = 0,
    avg_rouge1_f1: f64 = 0,
    avg_rouge2_f1: f64 = 0,
    avg_rougeL_f1: f64 = 0,
    avg_perplexity: f64 = 0,
    exact_match_ratio: f64 = 0,
    avg_f1: f64 = 0,
    std_bleu: f64 = 0,
    min_bleu: f64 = 0,
    max_bleu: f64 = 0,
};

/// Stub evaluator.
pub const Evaluator = struct {
    allocator: std.mem.Allocator,
    config: EvalConfig,

    pub fn init(allocator: std.mem.Allocator, config: EvalConfig) Evaluator {
        return .{
            .allocator = allocator,
            .config = config,
        };
    }

    pub fn evaluate(
        self: *Evaluator,
        hypothesis: []const u8,
        reference: []const u8,
    ) !EvaluationResult {
        _ = self;
        _ = hypothesis;
        _ = reference;
        return error.EvalDisabled;
    }

    pub fn evaluateBatch(
        self: *Evaluator,
        hypotheses: []const []const u8,
        references: []const []const u8,
    ) !EvaluationReport {
        _ = self;
        _ = hypotheses;
        _ = references;
        return error.EvalDisabled;
    }
};

/// Stub BLEU computation.
pub fn computeBleu(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
    max_ngram: u32,
    smoothing: SmoothingMethod,
) !BleuScore {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    _ = max_ngram;
    _ = smoothing;
    return error.EvalDisabled;
}

/// Stub BLEU multi-reference computation.
pub fn computeBleuMultiRef(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    references: []const []const u8,
    max_ngram: u32,
    smoothing: SmoothingMethod,
) !BleuScore {
    _ = allocator;
    _ = hypothesis;
    _ = references;
    _ = max_ngram;
    _ = smoothing;
    return error.EvalDisabled;
}

/// Stub ROUGE computation.
pub fn computeRouge(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
    rouge_type: RougeType,
) !RougeScore {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    _ = rouge_type;
    return error.EvalDisabled;
}

/// Stub ROUGE-N computation.
pub fn computeRougeN(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
    n: usize,
) !RougeScore {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    _ = n;
    return error.EvalDisabled;
}

/// Stub ROUGE-L computation.
pub fn computeRougeL(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !RougeScore {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub perplexity computation.
pub fn computePerplexity(log_probs: []const f64) PerplexityResult {
    _ = log_probs;
    return .{};
}

/// Stub F1 computation.
pub fn computeF1(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub exact match computation.
pub fn computeExactMatch(hypothesis: []const u8, reference: []const u8) f64 {
    _ = hypothesis;
    _ = reference;
    return 0;
}

/// Stub CER computation.
pub fn computeCER(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub WER computation.
pub fn computeWER(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub normalized exact match computation.
pub fn computeNormalizedExactMatch(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub Levenshtein distance computation.
pub fn levenshteinDistance(
    allocator: std.mem.Allocator,
    a: []const u8,
    b: []const u8,
) !usize {
    _ = allocator;
    _ = a;
    _ = b;
    return error.EvalDisabled;
}

/// Stub token metrics computation.
pub fn computeTokenMetrics(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !TokenMetrics {
    _ = allocator;
    _ = hypothesis;
    _ = reference;
    return error.EvalDisabled;
}

/// Stub text statistics computation.
pub fn computeTextStatistics(text: []const u8) TextStatistics {
    _ = text;
    return .{};
}

/// Stub windowed perplexity computation.
pub fn computeWindowedPerplexity(
    allocator: std.mem.Allocator,
    log_probs: []const f64,
    window_size: usize,
) ![]PerplexityResult {
    _ = allocator;
    _ = log_probs;
    _ = window_size;
    return error.EvalDisabled;
}

/// Stub perplexity from cross-entropy.
pub fn perplexityFromCrossEntropy(cross_entropy: f64) f64 {
    _ = cross_entropy;
    return 0;
}

/// Stub perplexity from BPC.
pub fn perplexityFromBpc(bpc: f64) f64 {
    _ = bpc;
    return 0;
}

/// Stub perplexity to BPC.
pub fn perplexityToBpc(perplexity_val: f64) f64 {
    _ = perplexity_val;
    return 0;
}

/// Stub aggregate perplexity.
pub fn aggregatePerplexity(results: []const PerplexityResult) PerplexityResult {
    _ = results;
    return .{};
}

/// Stub perplexity from probs.
pub fn computePerplexityFromProbs(probs: []const f64) PerplexityResult {
    _ = probs;
    return .{};
}
