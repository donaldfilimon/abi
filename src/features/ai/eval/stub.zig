//! Evaluation framework stub — disabled at compile time.

const std = @import("std");

// --- Types ---

pub const TokenMetrics = struct {
    true_positives: usize = 0,
    false_positives: usize = 0,
    false_negatives: usize = 0,
    precision: f64 = 0,
    recall: f64 = 0,
    f1: f64 = 0,
};

pub const TextStatistics = struct {
    char_count: usize = 0,
    word_count: usize = 0,
    sentence_count: usize = 0,
    avg_word_length: f64 = 0,
    unique_words: usize = 0,
    type_token_ratio: f64 = 0,
};

pub const BleuScore = struct {
    score: f64 = 0,
    precisions: [4]f64 = .{ 0, 0, 0, 0 },
    brevity_penalty: f64 = 0,
    hypothesis_length: usize = 0,
    reference_length: usize = 0,
    pub fn format(self: BleuScore, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = self;
        _ = fmt;
        _ = options;
        try writer.writeAll("BLEU: disabled");
    }
};

pub const RougeType = enum { rouge_1, rouge_2, rouge_3, rouge_l };

pub const RougeScore = struct {
    rouge_type: RougeType = .rouge_1,
    precision: f64 = 0,
    recall: f64 = 0,
    f1: f64 = 0,
    pub fn format(self: RougeScore, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = self;
        _ = fmt;
        _ = options;
        try writer.writeAll("ROUGE: disabled");
    }
};

pub const PerplexityResult = struct {
    perplexity: f64 = 0,
    avg_log_prob: f64 = 0,
    cross_entropy: f64 = 0,
    num_tokens: usize = 0,
    pub fn format(self: PerplexityResult, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = self;
        _ = fmt;
        _ = options;
        try writer.writeAll("Perplexity: disabled");
    }
};

const SmoothingMethod = enum { none, method1, method2, method3 };

pub const EvalConfig = struct {
    max_ngram: u32 = 4,
    bleu_smoothing: SmoothingMethod = .method1,
    rouge_types: []const RougeType = &[_]RougeType{ .rouge_1, .rouge_2, .rouge_l },
    case_insensitive: bool = true,
    compute_token_metrics: bool = true,
};

pub const EvaluationResult = struct {
    bleu: ?BleuScore = null,
    rouge: ?[]RougeScore = null,
    perplexity: ?PerplexityResult = null,
    exact_match: f64 = 0,
    f1: f64 = 0,
    token_metrics: ?TokenMetrics = null,
    hypothesis_stats: TextStatistics = .{},
    reference_stats: TextStatistics = .{},
    pub fn deinit(self: *EvaluationResult, _: std.mem.Allocator) void {
        self.* = undefined;
    }
};

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

// --- Evaluator ---

pub const Evaluator = struct {
    allocator: std.mem.Allocator,
    config: EvalConfig,
    pub fn init(allocator: std.mem.Allocator, config: EvalConfig) Evaluator {
        return .{ .allocator = allocator, .config = config };
    }
    pub fn evaluate(_: *Evaluator, _: []const u8, _: []const u8) !EvaluationResult {
        return error.FeatureDisabled;
    }
    pub fn evaluateBatch(_: *Evaluator, _: []const []const u8, _: []const []const u8) !EvaluationReport {
        return error.FeatureDisabled;
    }
};

// --- Metric Functions ---

pub fn computeBleu(_: std.mem.Allocator, _: []const u8, _: []const u8, _: u32, _: SmoothingMethod) !BleuScore {
    return error.FeatureDisabled;
}
pub fn computeBleuMultiRef(_: std.mem.Allocator, _: []const u8, _: []const []const u8, _: u32, _: SmoothingMethod) !BleuScore {
    return error.FeatureDisabled;
}
pub fn computeRouge(_: std.mem.Allocator, _: []const u8, _: []const u8, _: RougeType) !RougeScore {
    return error.FeatureDisabled;
}
pub fn computeRougeN(_: std.mem.Allocator, _: []const u8, _: []const u8, _: usize) !RougeScore {
    return error.FeatureDisabled;
}
pub fn computeRougeL(_: std.mem.Allocator, _: []const u8, _: []const u8) !RougeScore {
    return error.FeatureDisabled;
}
pub fn computePerplexity(_: []const f64) PerplexityResult {
    return .{};
}
pub fn computeF1(_: std.mem.Allocator, _: []const u8, _: []const u8) !f64 {
    return error.FeatureDisabled;
}
pub fn computeExactMatch(_: []const u8, _: []const u8) f64 {
    return 0;
}
pub fn computeCER(_: std.mem.Allocator, _: []const u8, _: []const u8) !f64 {
    return error.FeatureDisabled;
}
pub fn computeWER(_: std.mem.Allocator, _: []const u8, _: []const u8) !f64 {
    return error.FeatureDisabled;
}
pub fn computeNormalizedExactMatch(_: std.mem.Allocator, _: []const u8, _: []const u8) !f64 {
    return error.FeatureDisabled;
}
pub fn levenshteinDistance(_: std.mem.Allocator, _: []const u8, _: []const u8) !usize {
    return error.FeatureDisabled;
}
pub fn computeTokenMetrics(_: std.mem.Allocator, _: []const u8, _: []const u8) !TokenMetrics {
    return error.FeatureDisabled;
}
pub fn computeTextStatistics(_: []const u8) TextStatistics {
    return .{};
}
pub fn computeWindowedPerplexity(_: std.mem.Allocator, _: []const f64, _: usize) ![]PerplexityResult {
    return error.FeatureDisabled;
}
pub fn perplexityFromCrossEntropy(_: f64) f64 {
    return 0;
}
pub fn perplexityFromBpc(_: f64) f64 {
    return 0;
}
pub fn perplexityToBpc(_: f64) f64 {
    return 0;
}
pub fn aggregatePerplexity(_: []const PerplexityResult) PerplexityResult {
    return .{};
}
pub fn computePerplexityFromProbs(_: []const f64) PerplexityResult {
    return .{};
}
