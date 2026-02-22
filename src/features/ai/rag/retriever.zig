//! Retrieval component for RAG pipeline.
//!
//! Provides vector similarity-based retrieval of document chunks.
//! Integrates with the embeddings module for semantic search.
//!
//! ## Example
//!
//! ```zig
//! const retriever = @import("retriever.zig");
//! const embeddings = @import("../embeddings/mod.zig");
//!
//! // Create embeddings context with OpenAI backend
//! var embed_ctx = try embeddings.Context.init(allocator, .{});
//! defer embed_ctx.deinit();
//! try embed_ctx.useOpenAIFromEnv();
//!
//! // Create retriever with embeddings integration
//! var ret = retriever.Retriever.init(allocator, .{});
//! defer ret.deinit();
//! ret.setEmbeddingsContext(embed_ctx);
//!
//! // Now computeEmbedding uses real embeddings
//! const emb = try ret.computeEmbedding("search query");
//! ```

const std = @import("std");
const chunker = @import("chunker.zig");
const Chunk = chunker.Chunk;
const simd = @import("../../../services/shared/simd/mod.zig");
const embeddings = @import("../embeddings/mod.zig");

/// Common AI processing errors
pub const AIError = std.mem.Allocator.Error || error{
    EmbeddingFailed,
    ModelNotLoaded,
    InvalidInput,
    ProcessingTimeout,
};

/// Retriever configuration.
pub const RetrieverConfig = struct {
    /// Number of results to return.
    top_k: usize = 5,
    /// Minimum similarity score threshold.
    min_score: f32 = 0.3,
    /// Embedding dimension.
    embedding_dim: usize = 384,
    /// Custom embedding function.
    embed_fn: ?*const fn ([]const u8, std.mem.Allocator) AIError![]f32 = null,
    /// Similarity metric.
    similarity_metric: SimilarityMetric = .cosine,
};

/// Similarity metric type.
pub const SimilarityMetric = enum {
    cosine,
    euclidean,
    dot_product,
};

/// Retrieval result.
pub const RetrievalResult = struct {
    /// Retrieved chunk.
    chunk: Chunk,
    /// Source document ID.
    doc_id: []const u8,
    /// Similarity score.
    score: f32,
    /// Rank in results.
    rank: usize,
};

/// Document retriever.
pub const Retriever = struct {
    allocator: std.mem.Allocator,
    config: RetrieverConfig,
    /// Optional embeddings context for semantic search.
    embeddings_ctx: ?*embeddings.Context = null,

    /// Initialize retriever.
    pub fn init(allocator: std.mem.Allocator, config: RetrieverConfig) Retriever {
        return .{
            .allocator = allocator,
            .config = config,
            .embeddings_ctx = null,
        };
    }

    /// Deinitialize retriever.
    pub fn deinit(self: *Retriever) void {
        // Note: embeddings_ctx is not owned by retriever, don't deinit it
        self.embeddings_ctx = null;
        self.* = undefined;
    }

    /// Set the embeddings context for semantic search.
    /// The context is not owned by the retriever.
    pub fn setEmbeddingsContext(self: *Retriever, ctx: *embeddings.Context) void {
        self.embeddings_ctx = ctx;
        // Update dimension if embeddings context has different dimension
        if (ctx.config.dimension != self.config.embedding_dim) {
            self.config.embedding_dim = ctx.config.dimension;
        }
    }

    /// Check if embeddings context is configured.
    pub fn hasEmbeddingsContext(self: *const Retriever) bool {
        return self.embeddings_ctx != null;
    }

    /// Compute embedding for text.
    /// Uses embeddings context if available, otherwise falls back to custom fn or default.
    pub fn computeEmbedding(self: *Retriever, text: []const u8) !?[]f32 {
        // Priority 1: Use embeddings context if available
        if (self.embeddings_ctx) |ctx| {
            return ctx.embed(text) catch |err| {
                // Convert embeddings error to AIError
                return switch (err) {
                    embeddings.Error.EmbeddingFailed => AIError.EmbeddingFailed,
                    embeddings.Error.EmbeddingsDisabled => AIError.ModelNotLoaded,
                    else => AIError.EmbeddingFailed,
                };
            };
        }

        // Priority 2: Use custom embedding function if provided
        if (self.config.embed_fn) |embed_fn| {
            return try embed_fn(text, self.allocator);
        }

        // Priority 3: Return default hash-based embedding
        return try defaultEmbedding(self.allocator, text, self.config.embedding_dim);
    }

    /// Compute embeddings for multiple texts in batch.
    pub fn computeEmbeddingBatch(self: *Retriever, texts: []const []const u8) ![][]f32 {
        // Use embeddings context if available for efficient batching
        if (self.embeddings_ctx) |ctx| {
            return ctx.embedBatch(texts) catch |err| {
                return switch (err) {
                    embeddings.Error.EmbeddingFailed => AIError.EmbeddingFailed,
                    embeddings.Error.EmbeddingsDisabled => AIError.ModelNotLoaded,
                    else => AIError.EmbeddingFailed,
                };
            };
        }

        // Fallback: compute individually
        var results = try self.allocator.alloc([]f32, texts.len);
        var completed: usize = 0;
        errdefer {
            for (results[0..completed]) |emb| {
                self.allocator.free(emb);
            }
            self.allocator.free(results);
        }

        for (texts) |text| {
            const emb = try self.computeEmbedding(text);
            results[completed] = emb orelse return AIError.EmbeddingFailed;
            completed += 1;
        }

        return results;
    }

    /// Compute similarity between two embeddings.
    pub fn computeSimilarity(self: *const Retriever, a: []const f32, b: []const f32) f32 {
        return switch (self.config.similarity_metric) {
            .cosine => cosineSimilarity(a, b),
            .euclidean => euclideanSimilarity(a, b),
            .dot_product => dotProduct(a, b),
        };
    }

    /// Compute text similarity without embeddings.
    pub fn textSimilarity(_: *const Retriever, query: []const u8, text: []const u8) f32 {
        return jaccardSimilarity(query, text);
    }

    /// Re-rank results using a scoring function.
    pub fn rerank(
        self: *Retriever,
        results: []RetrievalResult,
        query: []const u8,
        score_fn: ?*const fn ([]const u8, Chunk) f32,
    ) void {
        const scorer = score_fn orelse struct {
            fn default(q: []const u8, c: Chunk) f32 {
                return jaccardSimilarity(q, c.content);
            }
        }.default;

        for (results) |*result| {
            const rerank_score = scorer(query, result.chunk);
            // Combine original score with rerank score
            result.score = result.score * 0.7 + rerank_score * 0.3;
        }

        // Re-sort by new scores
        std.mem.sort(RetrievalResult, results, {}, struct {
            fn lessThan(_: void, a: RetrievalResult, b: RetrievalResult) bool {
                return a.score > b.score;
            }
        }.lessThan);

        // Update ranks
        for (results, 0..) |*result, i| {
            result.rank = i + 1;
        }

        _ = self;
    }
};

/// Compute cosine similarity (SIMD-optimized via shared module).
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    return simd.cosineSimilarity(a, b);
}

/// Compute Euclidean similarity (1 / (1 + distance)).
pub fn euclideanSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len or a.len == 0) return 0;
    return 1.0 / (1.0 + simd.l2Distance(a, b));
}

/// Compute dot product (SIMD-optimized via shared module).
pub fn dotProduct(a: []const f32, b: []const f32) f32 {
    return simd.vectorDot(a, b);
}

/// Compute Jaccard similarity between texts.
pub fn jaccardSimilarity(a: []const u8, b: []const u8) f32 {
    if (a.len == 0 and b.len == 0) return 1.0;
    if (a.len == 0 or b.len == 0) return 0;

    // Simple word-level Jaccard
    var a_words: usize = 0;
    var b_words: usize = 0;
    var common: usize = 0;

    // Count words in a
    var i: usize = 0;
    while (i < a.len) {
        if (!std.ascii.isWhitespace(a[i])) {
            a_words += 1;
            // Skip word
            while (i < a.len and !std.ascii.isWhitespace(a[i])) i += 1;
        } else {
            i += 1;
        }
    }

    // Count words in b and check for common words
    i = 0;
    while (i < b.len) {
        if (!std.ascii.isWhitespace(b[i])) {
            b_words += 1;
            // Find word end
            var end = i;
            while (end < b.len and !std.ascii.isWhitespace(b[end])) end += 1;
            const word = b[i..end];

            // Check if word exists in a
            if (std.mem.indexOf(u8, a, word) != null) {
                common += 1;
            }
            i = end;
        } else {
            i += 1;
        }
    }

    const union_size = a_words + b_words - common;
    if (union_size == 0) return 0;
    return @as(f32, @floatFromInt(common)) / @as(f32, @floatFromInt(union_size));
}

/// Default pseudo-embedding based on character hash.
fn defaultEmbedding(
    allocator: std.mem.Allocator,
    text: []const u8,
    dim: usize,
) ![]f32 {
    var embedding = try allocator.alloc(f32, dim);
    @memset(embedding, 0);

    // Simple bag-of-characters with position weighting
    for (text, 0..) |c, i| {
        const idx = @as(usize, c) % dim;
        const pos_weight: f32 = 1.0 / @as(f32, @floatFromInt(i + 1));
        embedding[idx] += pos_weight;
    }

    // Normalize
    var norm: f32 = 0;
    for (embedding) |v| {
        norm += v * v;
    }
    norm = @sqrt(norm);
    if (norm > 0) {
        for (embedding) |*v| {
            v.* /= norm;
        }
    }

    return embedding;
}

test "cosine similarity" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0 };
    const c = [_]f32{ 0.0, 1.0, 0.0 };

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineSimilarity(&a, &b), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&a, &c), 0.001);
}

test "euclidean similarity" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 0.0, 0.0 };

    // Same point = max similarity
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), euclideanSimilarity(&a, &b), 0.001);
}

test "jaccard similarity" {
    const sim1 = jaccardSimilarity("hello world", "hello world");
    try std.testing.expect(sim1 > 0.5);

    const sim2 = jaccardSimilarity("hello", "goodbye");
    try std.testing.expect(sim2 < sim1);
}

test "retriever embedding" {
    const allocator = std.testing.allocator;
    var ret = Retriever.init(allocator, .{ .embedding_dim = 64 });
    defer ret.deinit();

    const emb = try ret.computeEmbedding("test text");
    defer if (emb) |e| allocator.free(e);

    try std.testing.expect(emb != null);
    try std.testing.expectEqual(@as(usize, 64), emb.?.len);
}
