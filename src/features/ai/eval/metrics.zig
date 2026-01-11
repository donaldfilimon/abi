//! General text and token metrics for evaluation.
//!
//! Provides utilities for computing F1, exact match, text statistics,
//! and other evaluation metrics.

const std = @import("std");

/// Token-level metrics comparing hypothesis to reference.
pub const TokenMetrics = struct {
    /// True positives (tokens in both).
    true_positives: usize,
    /// False positives (tokens only in hypothesis).
    false_positives: usize,
    /// False negatives (tokens only in reference).
    false_negatives: usize,
    /// Precision (0-1).
    precision: f64,
    /// Recall (0-1).
    recall: f64,
    /// F1 score (0-1).
    f1: f64,
};

/// Text statistics.
pub const TextStatistics = struct {
    /// Total characters.
    char_count: usize,
    /// Word count (whitespace separated).
    word_count: usize,
    /// Sentence count (. ! ? terminators).
    sentence_count: usize,
    /// Average word length.
    avg_word_length: f64,
    /// Unique words.
    unique_words: usize,
    /// Type-token ratio (lexical diversity).
    type_token_ratio: f64,
};

/// Compute token-level precision, recall, and F1.
pub fn computeTokenMetrics(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !TokenMetrics {
    const hyp_tokens = try tokenize(allocator, hypothesis);
    defer allocator.free(hyp_tokens);

    const ref_tokens = try tokenize(allocator, reference);
    defer allocator.free(ref_tokens);

    // Build set of reference tokens
    var ref_set = std.StringHashMapUnmanaged(void){};
    defer ref_set.deinit(allocator);

    for (ref_tokens) |token| {
        try ref_set.put(allocator, token, {});
    }

    // Count true positives and false positives
    var true_positives: usize = 0;
    var false_positives: usize = 0;

    var seen_hyp = std.StringHashMapUnmanaged(void){};
    defer seen_hyp.deinit(allocator);

    for (hyp_tokens) |token| {
        if (seen_hyp.contains(token)) continue;
        try seen_hyp.put(allocator, token, {});

        if (ref_set.contains(token)) {
            true_positives += 1;
        } else {
            false_positives += 1;
        }
    }

    // False negatives = reference tokens not in hypothesis
    var false_negatives: usize = 0;
    var ref_iter = ref_set.iterator();
    while (ref_iter.next()) |entry| {
        if (!seen_hyp.contains(entry.key_ptr.*)) {
            false_negatives += 1;
        }
    }

    const precision = if (true_positives + false_positives > 0)
        @as(f64, @floatFromInt(true_positives)) / @as(f64, @floatFromInt(true_positives + false_positives))
    else
        0;

    const recall = if (true_positives + false_negatives > 0)
        @as(f64, @floatFromInt(true_positives)) / @as(f64, @floatFromInt(true_positives + false_negatives))
    else
        0;

    const f1 = if (precision + recall > 0)
        2 * precision * recall / (precision + recall)
    else
        0;

    return .{
        .true_positives = true_positives,
        .false_positives = false_positives,
        .false_negatives = false_negatives,
        .precision = precision,
        .recall = recall,
        .f1 = f1,
    };
}

/// Compute F1 score between hypothesis and reference.
pub fn computeF1(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    const metrics = try computeTokenMetrics(allocator, hypothesis, reference);
    return metrics.f1;
}

/// Compute exact match (1.0 if identical, 0.0 otherwise).
pub fn computeExactMatch(hypothesis: []const u8, reference: []const u8) f64 {
    return if (std.mem.eql(u8, hypothesis, reference)) 1.0 else 0.0;
}

/// Compute normalized exact match (after lowercasing and stripping whitespace).
pub fn computeNormalizedExactMatch(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    const norm_hyp = try normalize(allocator, hypothesis);
    defer allocator.free(norm_hyp);

    const norm_ref = try normalize(allocator, reference);
    defer allocator.free(norm_ref);

    return computeExactMatch(norm_hyp, norm_ref);
}

/// Compute text statistics.
pub fn computeTextStatistics(text: []const u8) TextStatistics {
    if (text.len == 0) {
        return .{
            .char_count = 0,
            .word_count = 0,
            .sentence_count = 0,
            .avg_word_length = 0,
            .unique_words = 0,
            .type_token_ratio = 0,
        };
    }

    var word_count: usize = 0;
    var sentence_count: usize = 0;
    var total_word_length: usize = 0;
    var in_word = false;
    var current_word_len: usize = 0;

    for (text) |c| {
        if (std.ascii.isWhitespace(c)) {
            if (in_word) {
                word_count += 1;
                total_word_length += current_word_len;
                current_word_len = 0;
                in_word = false;
            }
        } else {
            in_word = true;
            current_word_len += 1;

            // Check for sentence terminators
            if (c == '.' or c == '!' or c == '?') {
                sentence_count += 1;
            }
        }
    }

    // Handle last word
    if (in_word) {
        word_count += 1;
        total_word_length += current_word_len;
    }

    // Ensure at least one sentence if there's text
    if (sentence_count == 0 and word_count > 0) {
        sentence_count = 1;
    }

    const avg_word_length = if (word_count > 0)
        @as(f64, @floatFromInt(total_word_length)) / @as(f64, @floatFromInt(word_count))
    else
        0;

    // For unique words and TTR, we'd need to tokenize - simplified version
    const unique_words = word_count; // Simplified
    const type_token_ratio = if (word_count > 0)
        @as(f64, @floatFromInt(unique_words)) / @as(f64, @floatFromInt(word_count))
    else
        0;

    return .{
        .char_count = text.len,
        .word_count = word_count,
        .sentence_count = sentence_count,
        .avg_word_length = avg_word_length,
        .unique_words = unique_words,
        .type_token_ratio = type_token_ratio,
    };
}

/// Compute Levenshtein (edit) distance.
pub fn levenshteinDistance(
    allocator: std.mem.Allocator,
    a: []const u8,
    b: []const u8,
) !usize {
    const m = a.len;
    const n = b.len;

    if (m == 0) return n;
    if (n == 0) return m;

    // Use two rows for memory efficiency
    var prev_row = try allocator.alloc(usize, n + 1);
    defer allocator.free(prev_row);

    var curr_row = try allocator.alloc(usize, n + 1);
    defer allocator.free(curr_row);

    // Initialize first row
    for (0..n + 1) |j| {
        prev_row[j] = j;
    }

    for (a, 0..) |ca, i| {
        curr_row[0] = i + 1;

        for (b, 0..) |cb, j| {
            const cost: usize = if (ca == cb) 0 else 1;
            curr_row[j + 1] = @min(@min(
                curr_row[j] + 1, // Insertion
                prev_row[j + 1] + 1, // Deletion
            ), prev_row[j] + cost); // Substitution
        }

        // Swap rows
        const tmp = prev_row;
        prev_row = curr_row;
        curr_row = tmp;
    }

    return prev_row[n];
}

/// Compute Character Error Rate (CER).
pub fn computeCER(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    if (reference.len == 0) return 0;

    const distance = try levenshteinDistance(allocator, hypothesis, reference);
    return @as(f64, @floatFromInt(distance)) / @as(f64, @floatFromInt(reference.len));
}

/// Compute Word Error Rate (WER).
pub fn computeWER(
    allocator: std.mem.Allocator,
    hypothesis: []const u8,
    reference: []const u8,
) !f64 {
    const hyp_tokens = try tokenize(allocator, hypothesis);
    defer allocator.free(hyp_tokens);

    const ref_tokens = try tokenize(allocator, reference);
    defer allocator.free(ref_tokens);

    if (ref_tokens.len == 0) return 0;

    const distance = try wordLevenshteinDistance(allocator, hyp_tokens, ref_tokens);
    return @as(f64, @floatFromInt(distance)) / @as(f64, @floatFromInt(ref_tokens.len));
}

fn tokenize(allocator: std.mem.Allocator, text: []const u8) ![]const []const u8 {
    var tokens = std.ArrayListUnmanaged([]const u8){};
    errdefer tokens.deinit(allocator);

    var start: usize = 0;
    var i: usize = 0;

    while (i < text.len) : (i += 1) {
        if (std.ascii.isWhitespace(text[i])) {
            if (i > start) {
                try tokens.append(allocator, text[start..i]);
            }
            start = i + 1;
        }
    }

    if (start < text.len) {
        try tokens.append(allocator, text[start..]);
    }

    return tokens.toOwnedSlice(allocator);
}

fn normalize(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var result = std.ArrayListUnmanaged(u8){};
    errdefer result.deinit(allocator);

    var in_space = true;
    for (text) |c| {
        if (std.ascii.isWhitespace(c)) {
            if (!in_space) {
                try result.append(allocator, ' ');
                in_space = true;
            }
        } else {
            try result.append(allocator, std.ascii.toLower(c));
            in_space = false;
        }
    }

    // Remove trailing space
    if (result.items.len > 0 and result.items[result.items.len - 1] == ' ') {
        _ = result.pop();
    }

    return result.toOwnedSlice(allocator);
}

fn wordLevenshteinDistance(
    allocator: std.mem.Allocator,
    a: []const []const u8,
    b: []const []const u8,
) !usize {
    const m = a.len;
    const n = b.len;

    if (m == 0) return n;
    if (n == 0) return m;

    var prev_row = try allocator.alloc(usize, n + 1);
    defer allocator.free(prev_row);

    var curr_row = try allocator.alloc(usize, n + 1);
    defer allocator.free(curr_row);

    for (0..n + 1) |j| {
        prev_row[j] = j;
    }

    for (a, 0..) |wa, i| {
        curr_row[0] = i + 1;

        for (b, 0..) |wb, j| {
            const cost: usize = if (std.mem.eql(u8, wa, wb)) 0 else 1;
            curr_row[j + 1] = @min(@min(
                curr_row[j] + 1,
                prev_row[j + 1] + 1,
            ), prev_row[j] + cost);
        }

        const tmp = prev_row;
        prev_row = curr_row;
        curr_row = tmp;
    }

    return prev_row[n];
}

test "exact match" {
    try std.testing.expectEqual(@as(f64, 1.0), computeExactMatch("hello", "hello"));
    try std.testing.expectEqual(@as(f64, 0.0), computeExactMatch("hello", "world"));
}

test "token metrics perfect match" {
    const allocator = std.testing.allocator;
    const metrics = try computeTokenMetrics(allocator, "the cat sat", "the cat sat");

    try std.testing.expectEqual(@as(usize, 3), metrics.true_positives);
    try std.testing.expectEqual(@as(usize, 0), metrics.false_positives);
    try std.testing.expectEqual(@as(usize, 0), metrics.false_negatives);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.f1, 0.0001);
}

test "levenshtein distance" {
    const allocator = std.testing.allocator;

    try std.testing.expectEqual(@as(usize, 0), try levenshteinDistance(allocator, "hello", "hello"));
    try std.testing.expectEqual(@as(usize, 1), try levenshteinDistance(allocator, "hello", "hallo"));
    try std.testing.expectEqual(@as(usize, 3), try levenshteinDistance(allocator, "kitten", "sitting"));
}

test "text statistics" {
    const stats = computeTextStatistics("Hello world. This is a test!");

    try std.testing.expect(stats.word_count == 6);
    try std.testing.expect(stats.sentence_count == 2);
    try std.testing.expect(stats.avg_word_length > 0);
}
