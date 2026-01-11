//! Batch embeddings API for efficient vector generation.
//!
//! Provides batch processing, caching, and similarity search capabilities
//! for text embeddings using transformer models.

const std = @import("std");
const batch = @import("batch.zig");
const cache = @import("cache.zig");

pub const BatchProcessor = batch.BatchProcessor;
pub const BatchConfig = batch.BatchConfig;
pub const BatchResult = batch.BatchResult;
pub const EmbeddingCache = cache.EmbeddingCache;
pub const CacheConfig = cache.CacheConfig;

/// Embedding model configuration.
pub const EmbedderConfig = struct {
    /// Embedding dimension (output size).
    dimension: u32 = 384,
    /// Maximum sequence length.
    max_seq_length: u32 = 512,
    /// Whether to normalize embeddings to unit length.
    normalize: bool = true,
    /// Model identifier for cache key generation.
    model_id: []const u8 = "default",
    /// Enable embedding cache.
    enable_cache: bool = true,
    /// Maximum cache size in entries.
    cache_max_entries: usize = 10000,
    /// Batch size for processing.
    batch_size: usize = 32,
};

/// A single embedding result.
pub const EmbeddingResult = struct {
    /// The embedding vector.
    vector: []f32,
    /// Input text hash for deduplication.
    text_hash: u64,
    /// Whether result came from cache.
    from_cache: bool,

    pub fn deinit(self: *EmbeddingResult, allocator: std.mem.Allocator) void {
        allocator.free(self.vector);
        self.* = undefined;
    }
};

/// Batch embedding response.
pub const BatchEmbeddingResponse = struct {
    /// Embeddings in same order as input texts.
    embeddings: []EmbeddingResult,
    /// Total texts processed.
    total_processed: usize,
    /// Number of cache hits.
    cache_hits: usize,
    /// Processing time in nanoseconds.
    processing_time_ns: u64,

    pub fn deinit(self: *BatchEmbeddingResponse, allocator: std.mem.Allocator) void {
        for (self.embeddings) |*emb| {
            emb.deinit(allocator);
        }
        allocator.free(self.embeddings);
        self.* = undefined;
    }
};

/// Similarity search result.
pub const SimilarityResult = struct {
    /// Index in the corpus.
    index: usize,
    /// Similarity score (cosine similarity).
    score: f32,
    /// Original text (if available).
    text: ?[]const u8,
};

/// Main embedder interface.
pub const Embedder = struct {
    allocator: std.mem.Allocator,
    config: EmbedderConfig,
    cache: ?EmbeddingCache,
    batch_processor: BatchProcessor,

    pub fn init(allocator: std.mem.Allocator, config: EmbedderConfig) !Embedder {
        var embedder = Embedder{
            .allocator = allocator,
            .config = config,
            .cache = null,
            .batch_processor = BatchProcessor.init(allocator, .{
                .batch_size = config.batch_size,
                .dimension = config.dimension,
                .normalize = config.normalize,
            }),
        };

        if (config.enable_cache) {
            embedder.cache = EmbeddingCache.init(allocator, .{
                .max_entries = config.cache_max_entries,
                .dimension = config.dimension,
            });
        }

        return embedder;
    }

    pub fn deinit(self: *Embedder) void {
        if (self.cache) |*c| {
            c.deinit();
        }
        self.batch_processor.deinit();
        self.* = undefined;
    }

    /// Embed a single text.
    pub fn embed(self: *Embedder, text: []const u8) !EmbeddingResult {
        const results = try self.embedBatch(&[_][]const u8{text});
        defer self.allocator.free(results.embeddings);

        // Take ownership of the first embedding
        if (results.embeddings.len > 0) {
            const result = results.embeddings[0];
            // Don't free the vector, we're returning it
            return result;
        }

        return error.EmbeddingFailed;
    }

    /// Embed multiple texts in batch.
    pub fn embedBatch(self: *Embedder, texts: []const []const u8) !BatchEmbeddingResponse {
        var timer = std.time.Timer.start() catch return error.TimerFailed;
        var embeddings = std.ArrayListUnmanaged(EmbeddingResult){};
        errdefer {
            for (embeddings.items) |*emb| {
                emb.deinit(self.allocator);
            }
            embeddings.deinit(self.allocator);
        }

        var cache_hits: usize = 0;
        var to_process = std.ArrayListUnmanaged(usize){};
        defer to_process.deinit(self.allocator);

        // Check cache first
        for (texts, 0..) |text, i| {
            const hash = computeTextHash(text);

            if (self.cache) |*c| {
                if (c.get(hash)) |cached_vector| {
                    try embeddings.append(self.allocator, .{
                        .vector = try self.allocator.dupe(f32, cached_vector),
                        .text_hash = hash,
                        .from_cache = true,
                    });
                    cache_hits += 1;
                    continue;
                }
            }

            try to_process.append(self.allocator, i);
            // Add placeholder that will be filled later
            try embeddings.append(self.allocator, .{
                .vector = &.{},
                .text_hash = hash,
                .from_cache = false,
            });
        }

        // Process uncached texts in batches
        if (to_process.items.len > 0) {
            var batch_texts = std.ArrayListUnmanaged([]const u8){};
            defer batch_texts.deinit(self.allocator);

            for (to_process.items) |idx| {
                try batch_texts.append(self.allocator, texts[idx]);
            }

            const batch_results = try self.batch_processor.process(batch_texts.items);
            defer {
                for (batch_results) |result| {
                    self.allocator.free(result);
                }
                self.allocator.free(batch_results);
            }

            // Fill in results and update cache
            for (to_process.items, 0..) |original_idx, result_idx| {
                const vector = try self.allocator.dupe(f32, batch_results[result_idx]);
                embeddings.items[original_idx].vector = vector;

                // Update cache
                if (self.cache) |*c| {
                    try c.put(embeddings.items[original_idx].text_hash, vector);
                }
            }
        }

        return .{
            .embeddings = try embeddings.toOwnedSlice(self.allocator),
            .total_processed = texts.len,
            .cache_hits = cache_hits,
            .processing_time_ns = timer.read(),
        };
    }

    /// Find most similar texts in a corpus.
    pub fn findSimilar(
        self: *Embedder,
        query: []const u8,
        corpus: []const []const u8,
        top_k: usize,
    ) ![]SimilarityResult {
        // Embed query
        var query_result = try self.embed(query);
        defer query_result.deinit(self.allocator);

        // Embed corpus
        var corpus_response = try self.embedBatch(corpus);
        defer corpus_response.deinit(self.allocator);

        // Compute similarities
        var scores = std.ArrayListUnmanaged(SimilarityResult){};
        errdefer scores.deinit(self.allocator);

        for (corpus_response.embeddings, 0..) |emb, i| {
            const score = cosineSimilarity(query_result.vector, emb.vector);
            try scores.append(self.allocator, .{
                .index = i,
                .score = score,
                .text = corpus[i],
            });
        }

        // Sort by score descending
        std.mem.sort(SimilarityResult, scores.items, {}, struct {
            fn lessThan(_: void, a: SimilarityResult, b: SimilarityResult) bool {
                return a.score > b.score;
            }
        }.lessThan);

        // Return top_k
        const k = @min(top_k, scores.items.len);
        const result = try self.allocator.alloc(SimilarityResult, k);
        @memcpy(result, scores.items[0..k]);

        scores.deinit(self.allocator);
        return result;
    }

    /// Clear the embedding cache.
    pub fn clearCache(self: *Embedder) void {
        if (self.cache) |*c| {
            c.clear();
        }
    }

    /// Get cache statistics.
    pub fn getCacheStats(self: *const Embedder) ?cache.CacheStats {
        if (self.cache) |c| {
            return c.getStats();
        }
        return null;
    }
};

/// Compute hash for cache key.
pub fn computeTextHash(text: []const u8) u64 {
    return std.hash.Wyhash.hash(0, text);
}

/// Compute cosine similarity between two vectors.
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len or a.len == 0) return 0;

    var dot: f32 = 0;
    var norm_a: f32 = 0;
    var norm_b: f32 = 0;

    for (a, b) |av, bv| {
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
    }

    const denom = @sqrt(norm_a) * @sqrt(norm_b);
    if (denom == 0) return 0;

    return dot / denom;
}

/// Compute euclidean distance between two vectors.
pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) return std.math.floatMax(f32);

    var sum: f32 = 0;
    for (a, b) |av, bv| {
        const diff = av - bv;
        sum += diff * diff;
    }

    return @sqrt(sum);
}

/// Normalize a vector to unit length.
pub fn normalize(vector: []f32) void {
    var norm: f32 = 0;
    for (vector) |v| {
        norm += v * v;
    }
    norm = @sqrt(norm);

    if (norm > 0) {
        for (vector) |*v| {
            v.* /= norm;
        }
    }
}

test "embedder initialization" {
    const allocator = std.testing.allocator;
    var embedder = try Embedder.init(allocator, .{});
    defer embedder.deinit();

    try std.testing.expect(embedder.cache != null);
}

test "cosine similarity" {
    const a = [_]f32{ 1, 0, 0 };
    const b = [_]f32{ 1, 0, 0 };
    const c = [_]f32{ 0, 1, 0 };

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), cosineSimilarity(&a, &b), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), cosineSimilarity(&a, &c), 0.0001);
}

test "normalize vector" {
    var vec = [_]f32{ 3, 4 };
    normalize(&vec);

    // Should be unit length
    const norm = @sqrt(vec[0] * vec[0] + vec[1] * vec[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 0.0001);
}

test "text hash computation" {
    const hash1 = computeTextHash("hello world");
    const hash2 = computeTextHash("hello world");
    const hash3 = computeTextHash("different text");

    try std.testing.expectEqual(hash1, hash2);
    try std.testing.expect(hash1 != hash3);
}
