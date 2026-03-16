//! Hybrid search combining vector and full-text search.
//!
//! Provides fusion of dense vector similarity and sparse keyword
//! matching for improved retrieval quality.

const std = @import("std");
const fulltext = @import("fulltext.zig");

/// Safely convert rank (usize) to u32 with saturation to prevent overflow.
fn safeRank(rank: usize) u32 {
    const rank_plus_one = rank +| 1; // Saturating add
    return @intCast(@min(rank_plus_one, std.math.maxInt(u32)));
}

/// Fusion method for combining search results.
pub const FusionMethod = enum {
    /// Reciprocal Rank Fusion (RRF).
    reciprocal_rank,
    /// Linear combination of scores.
    linear_combination,
    /// Weighted sum based on score distribution.
    distribution_weighted,
    /// Take maximum score.
    max_score,
    /// Cascading: text filter, then vector rank.
    cascade,
};

/// Hybrid search configuration.
pub const HybridSearchConfig = struct {
    /// Fusion method.
    fusion_method: FusionMethod = .reciprocal_rank,
    /// Weight for vector similarity (0.0 to 1.0).
    vector_weight: f32 = 0.5,
    /// Weight for text relevance (0.0 to 1.0).
    text_weight: f32 = 0.5,
    /// RRF constant (higher = more uniform blending).
    rrf_k: f32 = 60.0,
    /// Minimum text score to include in results.
    min_text_score: f32 = 0.0,
    /// Minimum vector score to include in results.
    min_vector_score: f32 = 0.0,
    /// Fetch multiplier for initial retrieval.
    fetch_multiplier: usize = 3,
};

/// Vector search result.
pub const VectorResult = struct {
    doc_id: u64,
    score: f32,
    vector: ?[]const f32 = null,
};

/// Text search result.
pub const TextResult = struct {
    doc_id: u64,
    score: f32,
    matched_terms: []const []const u8 = &.{},
};

/// Combined hybrid search result.
pub const HybridResult = struct {
    doc_id: u64,
    combined_score: f32,
    vector_score: ?f32,
    text_score: ?f32,
    vector_rank: ?u32,
    text_rank: ?u32,

    pub fn lessThan(_: void, a: HybridResult, b: HybridResult) bool {
        return a.combined_score > b.combined_score;
    }
};

/// Hybrid search engine.
pub const HybridSearchEngine = struct {
    allocator: std.mem.Allocator,
    config: HybridSearchConfig,
    text_index: ?*fulltext.InvertedIndex,

    /// Initialize hybrid search engine.
    pub fn init(allocator: std.mem.Allocator, config: HybridSearchConfig) HybridSearchEngine {
        return .{
            .allocator = allocator,
            .config = config,
            .text_index = null,
        };
    }

    /// Set text index for keyword search.
    pub fn setTextIndex(self: *HybridSearchEngine, index: *fulltext.InvertedIndex) void {
        self.text_index = index;
    }

    /// Perform hybrid search by combining vector and text results.
    pub fn search(
        self: *HybridSearchEngine,
        vector_results: []const VectorResult,
        text_results: []const TextResult,
        top_k: usize,
    ) ![]HybridResult {
        return switch (self.config.fusion_method) {
            .reciprocal_rank => try self.rrfFusion(vector_results, text_results, top_k),
            .linear_combination => try self.linearFusion(vector_results, text_results, top_k),
            .distribution_weighted => try self.distributionFusion(vector_results, text_results, top_k),
            .max_score => try self.maxScoreFusion(vector_results, text_results, top_k),
            .cascade => try self.cascadeFusion(vector_results, text_results, top_k),
        };
    }

    /// Reciprocal Rank Fusion.
    fn rrfFusion(
        self: *HybridSearchEngine,
        vector_results: []const VectorResult,
        text_results: []const TextResult,
        top_k: usize,
    ) ![]HybridResult {
        var scores = std.AutoHashMapUnmanaged(u64, RrfAccum){};
        defer scores.deinit(self.allocator);

        const k = self.config.rrf_k;

        // Add vector results
        for (vector_results, 0..) |result, rank| {
            const rrf_score = 1.0 / (k + @as(f32, @floatFromInt(rank + 1)));
            const entry = try scores.getOrPut(self.allocator, result.doc_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = RrfAccum{};
            }
            entry.value_ptr.rrf_score += rrf_score * self.config.vector_weight;
            entry.value_ptr.vector_score = result.score;
            entry.value_ptr.vector_rank = safeRank(rank);
        }

        // Add text results
        for (text_results, 0..) |result, rank| {
            const rrf_score = 1.0 / (k + @as(f32, @floatFromInt(rank + 1)));
            const entry = try scores.getOrPut(self.allocator, result.doc_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = RrfAccum{};
            }
            entry.value_ptr.rrf_score += rrf_score * self.config.text_weight;
            entry.value_ptr.text_score = result.score;
            entry.value_ptr.text_rank = safeRank(rank);
        }

        return try self.collectResults(&scores, top_k);
    }

    /// Linear combination of normalized scores.
    fn linearFusion(
        self: *HybridSearchEngine,
        vector_results: []const VectorResult,
        text_results: []const TextResult,
        top_k: usize,
    ) ![]HybridResult {
        var scores = std.AutoHashMapUnmanaged(u64, RrfAccum){};
        defer scores.deinit(self.allocator);

        // Normalize and add vector scores
        const v_max = if (vector_results.len > 0) vector_results[0].score else 1.0;
        for (vector_results, 0..) |result, rank| {
            const norm_score = if (v_max > 0) result.score / v_max else 0;
            const entry = try scores.getOrPut(self.allocator, result.doc_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = RrfAccum{};
            }
            entry.value_ptr.rrf_score += norm_score * self.config.vector_weight;
            entry.value_ptr.vector_score = result.score;
            entry.value_ptr.vector_rank = safeRank(rank);
        }

        // Normalize and add text scores
        const t_max = if (text_results.len > 0) text_results[0].score else 1.0;
        for (text_results, 0..) |result, rank| {
            const norm_score = if (t_max > 0) result.score / t_max else 0;
            const entry = try scores.getOrPut(self.allocator, result.doc_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = RrfAccum{};
            }
            entry.value_ptr.rrf_score += norm_score * self.config.text_weight;
            entry.value_ptr.text_score = result.score;
            entry.value_ptr.text_rank = safeRank(rank);
        }

        return try self.collectResults(&scores, top_k);
    }

    /// Distribution-weighted fusion.
    fn distributionFusion(
        self: *HybridSearchEngine,
        vector_results: []const VectorResult,
        text_results: []const TextResult,
        top_k: usize,
    ) ![]HybridResult {
        // Calculate score statistics for dynamic weighting
        var v_mean: f32 = 0;
        var t_mean: f32 = 0;

        for (vector_results) |r| v_mean += r.score;
        for (text_results) |r| t_mean += r.score;

        if (vector_results.len > 0) v_mean /= @floatFromInt(vector_results.len);
        if (text_results.len > 0) t_mean /= @floatFromInt(text_results.len);

        // Adjust weights based on score distribution
        const total_mean = v_mean + t_mean;
        const adjusted_v_weight = if (total_mean > 0) v_mean / total_mean else 0.5;
        const adjusted_t_weight = if (total_mean > 0) t_mean / total_mean else 0.5;

        var scores = std.AutoHashMapUnmanaged(u64, RrfAccum){};
        defer scores.deinit(self.allocator);

        for (vector_results, 0..) |result, rank| {
            const entry = try scores.getOrPut(self.allocator, result.doc_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = RrfAccum{};
            }
            entry.value_ptr.rrf_score += result.score * adjusted_v_weight;
            entry.value_ptr.vector_score = result.score;
            entry.value_ptr.vector_rank = safeRank(rank);
        }

        for (text_results, 0..) |result, rank| {
            const entry = try scores.getOrPut(self.allocator, result.doc_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = RrfAccum{};
            }
            entry.value_ptr.rrf_score += result.score * adjusted_t_weight;
            entry.value_ptr.text_score = result.score;
            entry.value_ptr.text_rank = safeRank(rank);
        }

        return try self.collectResults(&scores, top_k);
    }

    /// Max score fusion.
    fn maxScoreFusion(
        self: *HybridSearchEngine,
        vector_results: []const VectorResult,
        text_results: []const TextResult,
        top_k: usize,
    ) ![]HybridResult {
        var scores = std.AutoHashMapUnmanaged(u64, RrfAccum){};
        defer scores.deinit(self.allocator);

        for (vector_results, 0..) |result, rank| {
            const entry = try scores.getOrPut(self.allocator, result.doc_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = RrfAccum{};
            }
            entry.value_ptr.rrf_score = @max(entry.value_ptr.rrf_score, result.score * self.config.vector_weight);
            entry.value_ptr.vector_score = result.score;
            entry.value_ptr.vector_rank = safeRank(rank);
        }

        for (text_results, 0..) |result, rank| {
            const entry = try scores.getOrPut(self.allocator, result.doc_id);
            if (!entry.found_existing) {
                entry.value_ptr.* = RrfAccum{};
            }
            entry.value_ptr.rrf_score = @max(entry.value_ptr.rrf_score, result.score * self.config.text_weight);
            entry.value_ptr.text_score = result.score;
            entry.value_ptr.text_rank = safeRank(rank);
        }

        return try self.collectResults(&scores, top_k);
    }

    /// Cascade fusion: filter by text, rank by vector.
    fn cascadeFusion(
        self: *HybridSearchEngine,
        vector_results: []const VectorResult,
        text_results: []const TextResult,
        top_k: usize,
    ) ![]HybridResult {
        // Build set of text-matching doc IDs
        var text_docs = std.AutoHashMapUnmanaged(u64, TextResult){};
        defer text_docs.deinit(self.allocator);

        for (text_results) |result| {
            if (result.score >= self.config.min_text_score) {
                try text_docs.put(self.allocator, result.doc_id, result);
            }
        }

        // Collect vector results that match text filter
        var results = std.ArrayListUnmanaged(HybridResult).empty;
        defer results.deinit(self.allocator);

        for (vector_results, 0..) |v_result, rank| {
            if (text_docs.get(v_result.doc_id)) |t_result| {
                try results.append(self.allocator, .{
                    .doc_id = v_result.doc_id,
                    .combined_score = v_result.score,
                    .vector_score = v_result.score,
                    .text_score = t_result.score,
                    .vector_rank = safeRank(rank),
                    .text_rank = null,
                });
            }
        }

        // Sort by vector score
        std.mem.sort(HybridResult, results.items, {}, HybridResult.lessThan);

        // Return top_k
        const result_count = @min(top_k, results.items.len);
        const final_results = try self.allocator.alloc(HybridResult, result_count);
        @memcpy(final_results, results.items[0..result_count]);

        return final_results;
    }

    const RrfAccum = struct {
        rrf_score: f32 = 0,
        vector_score: ?f32 = null,
        text_score: ?f32 = null,
        vector_rank: ?u32 = null,
        text_rank: ?u32 = null,
    };

    fn collectResults(
        self: *HybridSearchEngine,
        scores: *std.AutoHashMapUnmanaged(u64, RrfAccum),
        top_k: usize,
    ) ![]HybridResult {
        var results = std.ArrayListUnmanaged(HybridResult).empty;
        defer results.deinit(self.allocator);

        var iter = scores.iterator();
        while (iter.next()) |entry| {
            try results.append(self.allocator, .{
                .doc_id = entry.key_ptr.*,
                .combined_score = entry.value_ptr.rrf_score,
                .vector_score = entry.value_ptr.vector_score,
                .text_score = entry.value_ptr.text_score,
                .vector_rank = entry.value_ptr.vector_rank,
                .text_rank = entry.value_ptr.text_rank,
            });
        }

        std.mem.sort(HybridResult, results.items, {}, HybridResult.lessThan);

        const result_count = @min(top_k, results.items.len);
        const final_results = try self.allocator.alloc(HybridResult, result_count);
        @memcpy(final_results, results.items[0..result_count]);

        return final_results;
    }
};

/// Convenience function for RRF fusion.
pub fn reciprocalRankFusion(
    allocator: std.mem.Allocator,
    vector_results: []const VectorResult,
    text_results: []const TextResult,
    top_k: usize,
) ![]HybridResult {
    var engine = HybridSearchEngine.init(allocator, .{ .fusion_method = .reciprocal_rank });
    return try engine.search(vector_results, text_results, top_k);
}

test "hybrid search rrf" {
    const allocator = std.testing.allocator;
    var engine = HybridSearchEngine.init(allocator, .{});

    const vector_results = [_]VectorResult{
        .{ .doc_id = 1, .score = 0.9 },
        .{ .doc_id = 2, .score = 0.8 },
        .{ .doc_id = 3, .score = 0.7 },
    };

    const text_results = [_]TextResult{
        .{ .doc_id = 2, .score = 5.0 },
        .{ .doc_id = 1, .score = 3.0 },
        .{ .doc_id = 4, .score = 2.0 },
    };

    const results = try engine.search(&vector_results, &text_results, 5);
    defer allocator.free(results);

    // Doc 2 should be top (appears in both with good ranks)
    try std.testing.expect(results.len > 0);
    // Results should be sorted by combined score
    for (1..results.len) |i| {
        try std.testing.expect(results[i - 1].combined_score >= results[i].combined_score);
    }
}

test "hybrid search linear" {
    const allocator = std.testing.allocator;
    var engine = HybridSearchEngine.init(allocator, .{
        .fusion_method = .linear_combination,
        .vector_weight = 0.6,
        .text_weight = 0.4,
    });

    const vector_results = [_]VectorResult{
        .{ .doc_id = 1, .score = 1.0 },
        .{ .doc_id = 2, .score = 0.5 },
    };

    const text_results = [_]TextResult{
        .{ .doc_id = 2, .score = 10.0 },
        .{ .doc_id = 1, .score = 1.0 },
    };

    const results = try engine.search(&vector_results, &text_results, 5);
    defer allocator.free(results);

    try std.testing.expect(results.len == 2);
}

test "hybrid search cascade" {
    const allocator = std.testing.allocator;
    var engine = HybridSearchEngine.init(allocator, .{
        .fusion_method = .cascade,
        .min_text_score = 1.0,
    });

    const vector_results = [_]VectorResult{
        .{ .doc_id = 1, .score = 0.9 },
        .{ .doc_id = 2, .score = 0.8 },
        .{ .doc_id = 3, .score = 0.7 },
    };

    const text_results = [_]TextResult{
        .{ .doc_id = 1, .score = 2.0 },
        .{ .doc_id = 3, .score = 1.5 },
        // doc_id 2 not in text results
    };

    const results = try engine.search(&vector_results, &text_results, 5);
    defer allocator.free(results);

    // Only docs 1 and 3 should be in results (2 filtered out by cascade)
    try std.testing.expect(results.len == 2);
}
