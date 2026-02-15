//! General text and token metrics for evaluation.
//!
//! Provides utilities for computing F1, exact match, text statistics,
//! and other evaluation metrics.

const std = @import("std");
const tokenizer = @import("tokenizer.zig");

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
    const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);
    defer allocator.free(hyp_tokens);

    const ref_tokens = try tokenizer.tokenize(allocator, reference);
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

/// FNV-1a hash for case-insensitive word hashing.
fn fnv1aHashWord(word: []const u8) u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    var hash: u64 = FNV_OFFSET_BASIS;
    for (word) |c| {
        // Case-insensitive: convert to lowercase
        const lower = std.ascii.toLower(c);
        hash ^= @as(u64, lower);
        hash *%= FNV_PRIME;
    }
    return hash;
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

    // Fixed-size hash table for tracking unique words (1024 entries)
    const HASH_TABLE_SIZE: usize = 1024;
    var hash_table: [HASH_TABLE_SIZE]u64 = [_]u64{0} ** HASH_TABLE_SIZE;
    var hash_occupied: [HASH_TABLE_SIZE]bool = [_]bool{false} ** HASH_TABLE_SIZE;

    var word_count: usize = 0;
    var unique_words: usize = 0;
    var sentence_count: usize = 0;
    var total_word_length: usize = 0;
    var word_start: usize = 0;
    var in_word = false;

    for (text, 0..) |c, i| {
        if (std.ascii.isWhitespace(c)) {
            if (in_word) {
                const word = text[word_start..i];
                word_count += 1;
                total_word_length += word.len;

                // Check if this word is unique using hash table
                const hash = fnv1aHashWord(word);
                var slot = hash % HASH_TABLE_SIZE;
                var found = false;

                // Linear probing to handle collisions
                var probes: usize = 0;
                while (probes < HASH_TABLE_SIZE) : (probes += 1) {
                    if (!hash_occupied[slot]) {
                        // Empty slot - word is unique
                        hash_table[slot] = hash;
                        hash_occupied[slot] = true;
                        unique_words += 1;
                        found = true;
                        break;
                    } else if (hash_table[slot] == hash) {
                        // Same hash - assume same word (collision possible but acceptable)
                        found = true;
                        break;
                    }
                    // Collision with different hash - probe next slot
                    slot = (slot + 1) % HASH_TABLE_SIZE;
                }

                // If we exhausted all probes, count as unique (table full edge case)
                if (!found) {
                    unique_words += 1;
                }

                in_word = false;
            }
        } else {
            if (!in_word) {
                word_start = i;
                in_word = true;
            }

            // Check for sentence terminators
            if (c == '.' or c == '!' or c == '?') {
                sentence_count += 1;
            }
        }
    }

    // Handle last word
    if (in_word) {
        const word = text[word_start..];
        word_count += 1;
        total_word_length += word.len;

        // Check if this word is unique
        const hash = fnv1aHashWord(word);
        var slot = hash % HASH_TABLE_SIZE;
        var found = false;

        var probes: usize = 0;
        while (probes < HASH_TABLE_SIZE) : (probes += 1) {
            if (!hash_occupied[slot]) {
                unique_words += 1;
                found = true;
                break;
            } else if (hash_table[slot] == hash) {
                found = true;
                break;
            }
            slot = (slot + 1) % HASH_TABLE_SIZE;
        }

        if (!found) {
            unique_words += 1;
        }
    }

    // Ensure at least one sentence if there's text
    if (sentence_count == 0 and word_count > 0) {
        sentence_count = 1;
    }

    const avg_word_length = if (word_count > 0)
        @as(f64, @floatFromInt(total_word_length)) / @as(f64, @floatFromInt(word_count))
    else
        0;

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
    const hyp_tokens = try tokenizer.tokenize(allocator, hypothesis);
    defer allocator.free(hyp_tokens);

    const ref_tokens = try tokenizer.tokenize(allocator, reference);
    defer allocator.free(ref_tokens);

    if (ref_tokens.len == 0) return 0;

    const distance = try wordLevenshteinDistance(allocator, hyp_tokens, ref_tokens);
    return @as(f64, @floatFromInt(distance)) / @as(f64, @floatFromInt(ref_tokens.len));
}

fn normalize(allocator: std.mem.Allocator, text: []const u8) ![]u8 {
    var result = std.ArrayListUnmanaged(u8).empty;
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

test "text statistics unique words" {
    const stats = computeTextStatistics("the cat sat on the mat");
    // "the" appears twice, so unique_words should be 5, not 6
    try std.testing.expectEqual(@as(usize, 6), stats.word_count);
    try std.testing.expectEqual(@as(usize, 5), stats.unique_words);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8333), stats.type_token_ratio, 0.01);
}

test "text statistics all unique" {
    const stats = computeTextStatistics("one two three four");
    try std.testing.expectEqual(@as(usize, 4), stats.word_count);
    try std.testing.expectEqual(@as(usize, 4), stats.unique_words);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), stats.type_token_ratio, 0.0001);
}

test "text statistics all same" {
    const stats = computeTextStatistics("word word word word");
    try std.testing.expectEqual(@as(usize, 4), stats.word_count);
    try std.testing.expectEqual(@as(usize, 1), stats.unique_words);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), stats.type_token_ratio, 0.0001);
}
