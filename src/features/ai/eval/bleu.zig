//! BLEU (Bilingual Evaluation Understudy) score calculation.
//!
//! Implements BLEU score with various smoothing methods for
//! evaluating machine translation and text generation quality.

const std = @import("std");
const tokenizer = @import("tokenizer.zig");

/// Smoothing method for BLEU calculation.
pub const SmoothingMethod = enum {
    /// No smoothing (original BLEU).
    none,
    /// Add 1 to numerator for zero counts.
    method1,
    /// Add 1 to both numerator and denominator.
    method2,
    /// Exponential decay smoothing.
    method3,
};

/// BLEU score result.
pub const BleuScore = struct {
    /// Overall BLEU score (0-1).
    score: f64,
    /// Individual n-gram precisions.
    precisions: [4]f64,
    /// Brevity penalty.
    brevity_penalty: f64,
    /// Hypothesis length.
    hypothesis_length: usize,
    /// Reference length.
    reference_length: usize,

    pub fn format(
        self: BleuScore,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("BLEU: {d:.4} (BP={d:.4}, P1={d:.4}, P2={d:.4}, P3={d:.4}, P4={d:.4})", .{
            self.score,
            self.brevity_penalty,
            self.precisions[0],
            self.precisions[1],
            self.precisions[2],
            self.precisions[3],
        });
    }
};

/// Compute BLEU score between hypothesis and reference.
pub fn computeBleu(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
    max_ngram: u32,
    smoothing: SmoothingMethod,
) !BleuScore {
    const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);
    defer allocator.free(hyp_tokens);

    const ref_tokens = try tokenizer.tokenize(allocator, reference);
    defer allocator.free(ref_tokens);

    return computeBleuFromTokens(allocator, hyp_tokens, &[_][]const []const u8{ref_tokens}, max_ngram, smoothing);
}

/// Compute BLEU with multiple references.
pub fn computeBleuMultiRef(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    references: []const []const u8,
    max_ngram: u32,
    smoothing: SmoothingMethod,
) !BleuScore {
    const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);
    defer allocator.free(hyp_tokens);

    var ref_tokens_list = std.ArrayListUnmanaged([]const []const u8){};
    defer {
        for (ref_tokens_list.items) |tokens| {
            allocator.free(tokens);
        }
        ref_tokens_list.deinit(allocator);
    }

    for (references) |ref| {
        const tokens = try tokenizer.tokenize(allocator, ref);
        errdefer allocator.free(tokens);
        try ref_tokens_list.append(allocator, tokens);
    }

    return computeBleuFromTokens(allocator, hyp_tokens, ref_tokens_list.items, max_ngram, smoothing);
}

fn computeBleuFromTokens(
    allocator: std.mem.Allocator,
    hyp_tokens: []const []const u8,
    ref_tokens_list: []const []const []const u8,
    max_ngram: u32,
    smoothing: SmoothingMethod,
) !BleuScore {
    var precisions: [4]f64 = .{ 0, 0, 0, 0 };
    const n = @min(max_ngram, 4);

    for (1..n + 1) |ngram_size| {
        var matches: f64 = 0;
        var total: f64 = 0;

        // Get hypothesis n-grams
        const hyp_ngrams = try getNgrams(allocator, hyp_tokens, ngram_size);
        defer {
            var iter = hyp_ngrams.iterator();
            while (iter.next()) |entry| {
                allocator.free(entry.key_ptr.*);
            }
            @constCast(&hyp_ngrams).deinit(allocator);
        }

        // Count matches against references
        var iter = hyp_ngrams.iterator();
        while (iter.next()) |entry| {
            const hyp_count = entry.value_ptr.*;
            total += @as(f64, @floatFromInt(hyp_count));

            // Find max count in any reference
            var max_ref_count: u32 = 0;
            for (ref_tokens_list) |ref_tokens| {
                const ref_ngrams = try getNgrams(allocator, ref_tokens, ngram_size);
                defer {
                    var ref_iter = ref_ngrams.iterator();
                    while (ref_iter.next()) |ref_entry| {
                        allocator.free(ref_entry.key_ptr.*);
                    }
                    @constCast(&ref_ngrams).deinit(allocator);
                }

                if (ref_ngrams.get(entry.key_ptr.*)) |ref_count| {
                    max_ref_count = @max(max_ref_count, ref_count);
                }
            }

            matches += @as(f64, @floatFromInt(@min(hyp_count, max_ref_count)));
        }

        // Apply smoothing
        const precision = switch (smoothing) {
            .none => if (total > 0) matches / total else 0,
            .method1 => if (total > 0) @max(matches, 1.0) / total else 0,
            .method2 => (matches + 1) / (total + 1),
            .method3 => blk: {
                if (total > 0 and matches > 0) {
                    break :blk matches / total;
                } else if (total > 0) {
                    // Exponential decay for zero counts
                    break :blk 1.0 / std.math.pow(f64, 2, @as(f64, @floatFromInt(ngram_size)));
                }
                break :blk 0;
            },
        };

        precisions[ngram_size - 1] = precision;
    }

    // Compute geometric mean of precisions
    var log_sum: f64 = 0;
    var valid_n: usize = 0;
    for (0..n) |i| {
        if (precisions[i] > 0) {
            log_sum += @log(precisions[i]);
            valid_n += 1;
        }
    }

    const geo_mean = if (valid_n > 0)
        @exp(log_sum / @as(f64, @floatFromInt(valid_n)))
    else
        0;

    // Compute brevity penalty
    const hyp_len = hyp_tokens.len;
    const ref_len = closestRefLength(ref_tokens_list, hyp_len);
    const brevity_penalty = if (hyp_len >= ref_len)
        1.0
    else
        @exp(1.0 - @as(f64, @floatFromInt(ref_len)) / @as(f64, @floatFromInt(@max(hyp_len, 1))));

    return .{
        .score = brevity_penalty * geo_mean,
        .precisions = precisions,
        .brevity_penalty = brevity_penalty,
        .hypothesis_length = hyp_len,
        .reference_length = ref_len,
    };
}

fn getNgrams(
    allocator: std.mem.Allocator,
    tokens: []const []const u8,
    n: usize,
) !std.StringHashMapUnmanaged(u32) {
    var ngrams = std.StringHashMapUnmanaged(u32){};
    errdefer ngrams.deinit(allocator);

    if (tokens.len < n) return ngrams;

    for (0..tokens.len - n + 1) |i| {
        // Create n-gram key
        var key_parts = std.ArrayListUnmanaged(u8){};
        defer key_parts.deinit(allocator);

        for (0..n) |j| {
            if (j > 0) try key_parts.append(allocator, ' ');
            try key_parts.appendSlice(allocator, tokens[i + j]);
        }

        const key = try allocator.dupe(u8, key_parts.items);

        if (ngrams.getPtr(key)) |count| {
            count.* += 1;
            allocator.free(key);
        } else {
            try ngrams.put(allocator, key, 1);
        }
    }

    return ngrams;
}

fn closestRefLength(ref_tokens_list: []const []const []const u8, hyp_len: usize) usize {
    if (ref_tokens_list.len == 0) return hyp_len;

    var closest = ref_tokens_list[0].len;
    var min_diff = if (hyp_len > closest) hyp_len - closest else closest - hyp_len;

    for (ref_tokens_list[1..]) |ref| {
        const diff = if (hyp_len > ref.len) hyp_len - ref.len else ref.len - hyp_len;
        if (diff < min_diff) {
            min_diff = diff;
            closest = ref.len;
        }
    }

    return closest;
}

test "bleu perfect match" {
    const allocator = std.testing.allocator;
    const result = try computeBleu(
        allocator,
        "the cat sat on the mat",
        "the cat sat on the mat",
        4,
        .none,
    );

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.score, 0.0001);
}

test "bleu partial match" {
    const allocator = std.testing.allocator;
    const result = try computeBleu(
        allocator,
        "the cat sat",
        "the cat sat on the mat",
        4,
        .method1,
    );

    // Should be less than 1 due to brevity penalty
    try std.testing.expect(result.score < 1.0);
    try std.testing.expect(result.score > 0.0);
}

test "bleu no match" {
    const allocator = std.testing.allocator;
    const result = try computeBleu(
        allocator,
        "foo bar baz",
        "the cat sat on the mat",
        4,
        .method1,
    );

    try std.testing.expect(result.score < 0.1);
}
