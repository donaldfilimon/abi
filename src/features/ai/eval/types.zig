//! Evaluation framework stub types — extracted from stub.zig.

const std = @import("std");

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

pub const SmoothingMethod = enum { none, method1, method2, method3 };

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
