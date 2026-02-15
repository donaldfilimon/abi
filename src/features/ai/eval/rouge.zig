//! ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score calculation.
//!
//! Implements ROUGE-N and ROUGE-L metrics for evaluating summarization
//! and text generation quality.

const std = @import("std");
const tokenizer = @import("tokenizer.zig");

/// ROUGE metric type.
pub const RougeType = enum {
    rouge_1,
    rouge_2,
    rouge_3,
    rouge_l,
};

/// ROUGE score result.
pub const RougeScore = struct {
    /// ROUGE type used.
    rouge_type: RougeType,
    /// Precision score (0-1).
    precision: f64,
    /// Recall score (0-1).
    recall: f64,
    /// F1 score (0-1).
    f1: f64,

    pub fn format(
        self: RougeScore,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        const type_str = switch (self.rouge_type) {
            .rouge_1 => "ROUGE-1",
            .rouge_2 => "ROUGE-2",
            .rouge_3 => "ROUGE-3",
            .rouge_l => "ROUGE-L",
        };
        try writer.print("{s}: P={d:.4} R={d:.4} F1={d:.4}", .{
            type_str,
            self.precision,
            self.recall,
            self.f1,
        });
    }
};

/// Compute ROUGE score.
pub fn computeRouge(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
    rouge_type: RougeType,
) !RougeScore {
    return switch (rouge_type) {
        .rouge_1 => computeRougeN(allocator, hypothesis, reference, 1),
        .rouge_2 => computeRougeN(allocator, hypothesis, reference, 2),
        .rouge_3 => computeRougeN(allocator, hypothesis, reference, 3),
        .rouge_l => computeRougeL(allocator, hypothesis, reference),
    };
}

/// Compute ROUGE-N score.
pub fn computeRougeN(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
    n: usize,
) !RougeScore {
    const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);
    defer allocator.free(hyp_tokens);

    const ref_tokens = try tokenizer.tokenize(allocator, reference);
    defer allocator.free(ref_tokens);

    if (hyp_tokens.len < n or ref_tokens.len < n) {
        return .{
            .rouge_type = switch (n) {
                1 => .rouge_1,
                2 => .rouge_2,
                3 => .rouge_3,
                else => .rouge_1,
            },
            .precision = 0,
            .recall = 0,
            .f1 = 0,
        };
    }

    // Get n-grams
    const hyp_ngrams = try getNgrams(allocator, hyp_tokens, n);
    defer {
        var iter = hyp_ngrams.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        @constCast(&hyp_ngrams).deinit(allocator);
    }

    const ref_ngrams = try getNgrams(allocator, ref_tokens, n);
    defer {
        var iter = ref_ngrams.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        @constCast(&ref_ngrams).deinit(allocator);
    }

    // Count overlapping n-grams
    var overlap: f64 = 0;
    var hyp_total: f64 = 0;
    var ref_total: f64 = 0;

    var hyp_iter = hyp_ngrams.iterator();
    while (hyp_iter.next()) |entry| {
        const hyp_count = @as(f64, @floatFromInt(entry.value_ptr.*));
        hyp_total += hyp_count;

        if (ref_ngrams.get(entry.key_ptr.*)) |ref_count| {
            overlap += @min(hyp_count, @as(f64, @floatFromInt(ref_count)));
        }
    }

    var ref_iter = ref_ngrams.iterator();
    while (ref_iter.next()) |entry| {
        ref_total += @as(f64, @floatFromInt(entry.value_ptr.*));
    }

    const precision = if (hyp_total > 0) overlap / hyp_total else 0;
    const recall = if (ref_total > 0) overlap / ref_total else 0;
    const f1 = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0;

    return .{
        .rouge_type = switch (n) {
            1 => .rouge_1,
            2 => .rouge_2,
            3 => .rouge_3,
            else => .rouge_1,
        },
        .precision = precision,
        .recall = recall,
        .f1 = f1,
    };
}

/// Compute ROUGE-L score using Longest Common Subsequence.
pub fn computeRougeL(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !RougeScore {
    const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);
    defer allocator.free(hyp_tokens);

    const ref_tokens = try tokenizer.tokenize(allocator, reference);
    defer allocator.free(ref_tokens);

    if (hyp_tokens.len == 0 or ref_tokens.len == 0) {
        return .{
            .rouge_type = .rouge_l,
            .precision = 0,
            .recall = 0,
            .f1 = 0,
        };
    }

    // Compute LCS length
    const lcs_length = try computeLcsLength(allocator, hyp_tokens, ref_tokens);

    const hyp_len = @as(f64, @floatFromInt(hyp_tokens.len));
    const ref_len = @as(f64, @floatFromInt(ref_tokens.len));
    const lcs_len = @as(f64, @floatFromInt(lcs_length));

    const precision = lcs_len / hyp_len;
    const recall = lcs_len / ref_len;
    const f1 = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0;

    return .{
        .rouge_type = .rouge_l,
        .precision = precision,
        .recall = recall,
        .f1 = f1,
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
        var key_parts = std.ArrayListUnmanaged(u8).empty;
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

fn computeLcsLength(
    allocator: std.mem.Allocator,
    a: []const []const u8,
    b: []const []const u8,
) !usize {
    const m = a.len;
    const n = b.len;

    // Use two rows instead of full matrix for memory efficiency
    var prev_row = try allocator.alloc(usize, n + 1);
    defer allocator.free(prev_row);
    @memset(prev_row, 0);

    var curr_row = try allocator.alloc(usize, n + 1);
    defer allocator.free(curr_row);

    for (0..m) |i| {
        @memset(curr_row, 0);

        for (0..n) |j| {
            if (std.mem.eql(u8, a[i], b[j])) {
                curr_row[j + 1] = prev_row[j] + 1;
            } else {
                curr_row[j + 1] = @max(curr_row[j], prev_row[j + 1]);
            }
        }

        // Swap rows
        const tmp = prev_row;
        prev_row = curr_row;
        curr_row = tmp;
    }

    return prev_row[n];
}

test "rouge-1 perfect match" {
    const allocator = std.testing.allocator;
    const result = try computeRougeN(
        allocator,
        "the cat sat on the mat",
        "the cat sat on the mat",
        1,
    );

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.f1, 0.0001);
}

test "rouge-2 partial match" {
    const allocator = std.testing.allocator;
    const result = try computeRougeN(
        allocator,
        "the cat sat",
        "the cat sat on the mat",
        2,
    );

    // Should have perfect precision but lower recall
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.precision, 0.0001);
    try std.testing.expect(result.recall < 1.0);
}

test "rouge-l computation" {
    const allocator = std.testing.allocator;
    const result = try computeRougeL(
        allocator,
        "the cat sat on the mat",
        "the cat sat on the mat",
    );

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.f1, 0.0001);
}

test "rouge-l partial match" {
    const allocator = std.testing.allocator;
    const result = try computeRougeL(
        allocator,
        "the cat mat",
        "the cat sat on the mat",
    );

    // LCS = "the cat mat" = 3 words
    // Precision = 3/3 = 1.0
    // Recall = 3/6 = 0.5
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result.precision, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), result.recall, 0.0001);
}
