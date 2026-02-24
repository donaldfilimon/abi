//! Perplexity calculation for language model evaluation.
//!
//! Provides functions to compute perplexity from log probabilities
//! and cross-entropy loss.

const std = @import("std");

/// Perplexity result.
pub const PerplexityResult = struct {
    /// Perplexity value.
    perplexity: f64,
    /// Average log probability per token.
    avg_log_prob: f64,
    /// Cross-entropy loss.
    cross_entropy: f64,
    /// Number of tokens evaluated.
    num_tokens: usize,

    pub fn format(
        self: PerplexityResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Perplexity: {d:.4} (CE: {d:.4}, Tokens: {d})", .{
            self.perplexity,
            self.cross_entropy,
            self.num_tokens,
        });
    }
};

/// Compute perplexity from log probabilities.
/// Log probabilities should be natural log (base e).
pub fn computePerplexity(log_probs: []const f64) PerplexityResult {
    if (log_probs.len == 0) {
        return .{
            .perplexity = std.math.inf(f64),
            .avg_log_prob = 0,
            .cross_entropy = std.math.inf(f64),
            .num_tokens = 0,
        };
    }

    var sum: f64 = 0;
    for (log_probs) |lp| {
        sum += lp;
    }

    const n = @as(f64, @floatFromInt(log_probs.len));
    const avg_log_prob = sum / n;
    const cross_entropy = -avg_log_prob;
    const perplexity = @exp(cross_entropy);

    return .{
        .perplexity = perplexity,
        .avg_log_prob = avg_log_prob,
        .cross_entropy = cross_entropy,
        .num_tokens = log_probs.len,
    };
}

/// Compute perplexity from cross-entropy loss.
pub fn perplexityFromCrossEntropy(cross_entropy: f64) f64 {
    return @exp(cross_entropy);
}

/// Compute perplexity from bits per character/word.
pub fn perplexityFromBpc(bpc: f64) f64 {
    return std.math.pow(f64, 2, bpc);
}

/// Convert perplexity to bits per character/word.
pub fn perplexityToBpc(perplexity: f64) f64 {
    return @log2(perplexity);
}

/// Compute perplexity for a sequence with model probabilities.
/// Probabilities should be actual probabilities (0-1), not log probs.
pub fn computePerplexityFromProbs(probs: []const f64) PerplexityResult {
    if (probs.len == 0) {
        return .{
            .perplexity = std.math.inf(f64),
            .avg_log_prob = 0,
            .cross_entropy = std.math.inf(f64),
            .num_tokens = 0,
        };
    }

    // Compute directly without allocation - sum log probs inline
    var sum: f64 = 0;
    for (probs) |p| {
        // Clamp to avoid log(0)
        const clamped = @max(p, 1e-10);
        sum += @log(clamped);
    }

    const n = @as(f64, @floatFromInt(probs.len));
    const avg_log_prob = sum / n;
    const cross_entropy = -avg_log_prob;
    const perplexity = @exp(cross_entropy);

    return .{
        .perplexity = perplexity,
        .avg_log_prob = avg_log_prob,
        .cross_entropy = cross_entropy,
        .num_tokens = probs.len,
    };
}

/// Aggregate perplexity results from multiple sequences.
pub fn aggregatePerplexity(results: []const PerplexityResult) PerplexityResult {
    if (results.len == 0) {
        return .{
            .perplexity = std.math.inf(f64),
            .avg_log_prob = 0,
            .cross_entropy = std.math.inf(f64),
            .num_tokens = 0,
        };
    }

    var total_log_prob: f64 = 0;
    var total_tokens: usize = 0;

    for (results) |r| {
        total_log_prob += r.avg_log_prob * @as(f64, @floatFromInt(r.num_tokens));
        total_tokens += r.num_tokens;
    }

    if (total_tokens == 0) {
        return .{
            .perplexity = std.math.inf(f64),
            .avg_log_prob = 0,
            .cross_entropy = std.math.inf(f64),
            .num_tokens = 0,
        };
    }

    const avg_log_prob = total_log_prob / @as(f64, @floatFromInt(total_tokens));
    const cross_entropy = -avg_log_prob;
    const perplexity = @exp(cross_entropy);

    return .{
        .perplexity = perplexity,
        .avg_log_prob = avg_log_prob,
        .cross_entropy = cross_entropy,
        .num_tokens = total_tokens,
    };
}

/// Compute windowed perplexity for detecting anomalies.
pub fn computeWindowedPerplexity(
    allocator: std.mem.Allocator,
    log_probs: []const f64,
    window_size: usize,
) ![]PerplexityResult {
    if (log_probs.len < window_size or window_size == 0) {
        return &.{};
    }

    const num_windows = log_probs.len - window_size + 1;
    var results = try allocator.alloc(PerplexityResult, num_windows);
    errdefer allocator.free(results);

    for (0..num_windows) |i| {
        results[i] = computePerplexity(log_probs[i .. i + window_size]);
    }

    return results;
}

test "perplexity from uniform distribution" {
    // For a uniform distribution over V vocabulary items,
    // perplexity should be V
    const vocab_size: f64 = 10;
    const log_prob = @log(1.0 / vocab_size);
    const log_probs = [_]f64{ log_prob, log_prob, log_prob };

    const result = computePerplexity(&log_probs);

    try std.testing.expectApproxEqAbs(vocab_size, result.perplexity, 0.0001);
}

test "perplexity from certain prediction" {
    // If model always predicts correctly with probability 1,
    // perplexity should be 1
    const log_probs = [_]f64{ @log(1.0), @log(1.0), @log(1.0) };
    const result = computePerplexity(&log_probs);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.perplexity, 0.0001);
}

test "perplexity bpc conversion" {
    const ppl: f64 = 100;
    const bpc = perplexityToBpc(ppl);
    const recovered = perplexityFromBpc(bpc);

    try std.testing.expectApproxEqAbs(ppl, recovered, 0.0001);
}

test "aggregate perplexity" {
    const r1 = PerplexityResult{
        .perplexity = 10,
        .avg_log_prob = -@log(10.0),
        .cross_entropy = @log(10.0),
        .num_tokens = 100,
    };
    const r2 = PerplexityResult{
        .perplexity = 20,
        .avg_log_prob = -@log(20.0),
        .cross_entropy = @log(20.0),
        .num_tokens = 100,
    };

    const results = [_]PerplexityResult{ r1, r2 };
    const aggregated = aggregatePerplexity(&results);

    // Aggregate perplexity should be geometric mean weighted by token count
    try std.testing.expect(aggregated.perplexity > 10);
    try std.testing.expect(aggregated.perplexity < 20);
    try std.testing.expectEqual(@as(usize, 200), aggregated.num_tokens);
}

test "perplexity from probs long sequence" {
    // Create sequence longer than 1024
    var probs: [2000]f64 = undefined;
    for (&probs) |*p| {
        p.* = 0.1; // 10% probability each
    }

    const result = computePerplexityFromProbs(&probs);

    // Should process all 2000 tokens, not just 1024
    try std.testing.expectEqual(@as(usize, 2000), result.num_tokens);

    // Perplexity of uniform 0.1 = 1/0.1 = 10
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result.perplexity, 0.01);
}

test {
    std.testing.refAllDecls(@This());
}
