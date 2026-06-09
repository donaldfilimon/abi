const std = @import("std");

pub fn countNonEmptyLines(data: []const u8) usize {
    var records: usize = 0;
    var lines = std.mem.splitScalar(u8, data, '\n');
    while (lines.next()) |line| {
        if (std.mem.trim(u8, line, " \t\r").len > 0) records += 1;
    }
    return records;
}

/// Embedding dimensionality. Wider than a toy hash so cosine similarity carries
/// real lexical signal; ≤ HNSW padding width so the index stores it directly.
pub const EMBED_DIM: usize = 32;

/// Deterministic text embedding via the signed feature-hashing trick over
/// character n-grams (unigram/bigram/trigram). Each n-gram hashes to a bucket
/// with a ± sign, so strings that share n-grams land on overlapping buckets with
/// consistent signs and score high cosine similarity — a genuine (if classical,
/// non-learned) semantic signal, unlike a per-position character-frequency hash.
/// L2-normalized; an empty input maps to a fixed unit vector.
pub fn textEmbedding(input: []const u8) [EMBED_DIM]f32 {
    var out: [EMBED_DIM]f32 = @splat(0);
    if (input.len == 0) {
        out[0] = 1.0;
        return out;
    }

    var window: [3]u8 = undefined;
    inline for (.{ 1, 2, 3 }) |n| {
        var i: usize = 0;
        while (i + n <= input.len) : (i += 1) {
            for (0..n) |k| window[k] = std.ascii.toLower(input[i + k]);
            // Seed by n so unigram/bigram/trigram feature spaces stay distinct.
            const h = std.hash.Wyhash.hash(n, window[0..n]);
            const bucket = h % EMBED_DIM;
            const sign: f32 = if ((h >> 63) & 1 == 0) 1.0 else -1.0;
            out[bucket] += sign;
        }
    }

    var norm: f32 = 0;
    for (out) |v| norm += v * v;
    if (norm == 0) {
        out[0] = 1.0;
        return out;
    }
    const scale = @sqrt(norm);
    for (&out) |*v| v.* /= scale;
    return out;
}

/// A response embedding derived from the query embedding: a small deterministic
/// per-dimension perturbation so the stored response vector is distinct from but
/// related to the query vector.
pub fn responseEmbedding(query: [EMBED_DIM]f32) [EMBED_DIM]f32 {
    var out: [EMBED_DIM]f32 = undefined;
    for (query, 0..) |v, i| out[i] = v * (if (i % 2 == 0) @as(f32, 0.99) else @as(f32, 1.01));
    return out;
}

test {
    std.testing.refAllDecls(@This());
}

test "countNonEmptyLines" {
    try std.testing.expectEqual(@as(usize, 2), countNonEmptyLines("hello\nworld\n"));
    try std.testing.expectEqual(@as(usize, 0), countNonEmptyLines(""));
}

test "textEmbedding returns a normalized EMBED_DIM vector" {
    const v = textEmbedding("test");
    try std.testing.expect(v.len == EMBED_DIM);
    var norm: f32 = 0;
    for (v) |val| norm += val * val;
    try std.testing.expect(@abs(norm - 1.0) < 0.001);
}

fn cosine(a: [EMBED_DIM]f32, b: [EMBED_DIM]f32) f32 {
    var dot: f32 = 0;
    for (a, b) |x, y| dot += x * y;
    return dot; // both inputs are unit vectors
}

test "textEmbedding: shared n-grams yield higher similarity than unrelated text" {
    const base = textEmbedding("hello world cognitive runtime");
    const similar = textEmbedding("hello world cognitive engine");
    const unrelated = textEmbedding("zzzz qqqq xkcd vbnm");
    // Lexical overlap must score strictly higher than an unrelated string —
    // the property a per-position character hash cannot provide.
    try std.testing.expect(cosine(base, similar) > cosine(base, unrelated));
}

test "responseEmbedding perturbs each dimension deterministically" {
    const q: [EMBED_DIM]f32 = @splat(1.0);
    const r = responseEmbedding(q);
    try std.testing.expect(r[0] < q[0]);
    try std.testing.expect(r[1] > q[1]);
}
