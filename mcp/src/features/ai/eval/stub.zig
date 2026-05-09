//! Evaluation framework stub — disabled at compile time.

const std = @import("std");
const types = @import("types.zig");

// Re-export types
pub const TokenMetrics = types.TokenMetrics;
pub const TextStatistics = types.TextStatistics;
pub const BleuScore = types.BleuScore;
pub const RougeType = types.RougeType;
pub const RougeScore = types.RougeScore;
pub const PerplexityResult = types.PerplexityResult;
pub const SmoothingMethod = types.SmoothingMethod;
pub const EvalConfig = types.EvalConfig;
pub const EvaluationResult = types.EvaluationResult;
pub const EvaluationReport = types.EvaluationReport;

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

// Shared tokenizer stub
pub const tokenizer = struct {
    pub fn tokenize(_: std.mem.Allocator, _: []const u8) ![]const []const u8 {
        return error.FeatureDisabled;
    }
    pub fn countTokens(_: []const u8) usize {
        return 0;
    }
};
pub fn tokenize(_: std.mem.Allocator, _: []const u8) ![]const []const u8 {
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
