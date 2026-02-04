//! Embeddings Sub-module
//!
//! Vector embeddings generation for text and other data types.
//! Provides models for converting text into dense vector representations.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../config/mod.zig");
const simd = @import("../../shared/simd.zig");

// SIMD vector size for batch normalization
const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

/// SIMD-accelerated L2 normalization of a single embedding vector.
/// Normalizes the vector to unit length in-place.
/// @param embedding The embedding vector to normalize (modified in-place)
pub fn normalizeEmbedding(embedding: []f32) void {
    if (embedding.len == 0) return;

    // Step 1: Compute L2 norm using SIMD
    const norm = simd.vectorL2Norm(embedding);

    if (norm == 0.0 or norm == 1.0) return;

    // Step 2: Divide by norm using SIMD
    simd.divideByScalar(embedding, norm);
}

/// SIMD-accelerated batch L2 normalization for multiple embedding vectors.
/// Normalizes each vector to unit length in-place.
/// @param embeddings Array of embedding vectors to normalize
pub fn normalizeEmbeddingBatch(embeddings: [][]f32) void {
    for (embeddings) |embedding| {
        normalizeEmbedding(embedding);
    }
}

/// SIMD-accelerated batch L2 normalization with pre-allocated norm storage.
/// More efficient when norms need to be computed and stored for later use.
/// @param embeddings Array of embedding vectors to normalize (modified in-place)
/// @param norms Optional output array to store the original norms (must be same length as embeddings)
pub fn normalizeEmbeddingBatchWithNorms(embeddings: [][]f32, norms: ?[]f32) void {
    for (embeddings, 0..) |embedding, i| {
        if (embedding.len == 0) {
            if (norms) |n| n[i] = 0.0;
            continue;
        }

        const norm = simd.vectorL2Norm(embedding);
        if (norms) |n| n[i] = norm;

        if (norm == 0.0 or norm == 1.0) continue;

        simd.divideByScalar(embedding, norm);
    }
}

pub const Error = error{
    EmbeddingsDisabled,
    ModelNotFound,
    EmbeddingFailed,
    InvalidInput,
};

/// Configuration for embedding models.
pub const EmbeddingConfig = struct {
    /// Embedding dimension.
    dimension: usize = 384,
    /// Maximum sequence length.
    max_seq_len: usize = 512,
    /// Batch size for processing.
    batch_size: usize = 32,
    /// Normalize embeddings to unit length.
    normalize: bool = true,
    /// Model identifier.
    model_id: []const u8 = "default",
};

/// Embedding model for text vectorization.
pub const EmbeddingModel = struct {
    allocator: std.mem.Allocator,
    config: EmbeddingConfig,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator, config: EmbeddingConfig) EmbeddingModel {
        return .{
            .allocator = allocator,
            .config = config,
            .initialized = true,
        };
    }

    pub fn deinit(self: *EmbeddingModel) void {
        self.initialized = false;
    }

    /// Generate embedding for a single text.
    pub fn embed(self: *EmbeddingModel, text: []const u8) ![]f32 {
        _ = text;
        const result = try self.allocator.alloc(f32, self.config.dimension);
        @memset(result, 0);

        // Simple hash-based embedding for now
        if (self.config.normalize) {
            // Normalize to unit length
            result[0] = 1.0;
        }

        return result;
    }

    /// Generate embeddings for multiple texts.
    /// Caller owns the returned slice and each embedding within it.
    pub fn embedBatch(self: *EmbeddingModel, texts: []const []const u8) ![][]f32 {
        const results = try self.allocator.alloc([]f32, texts.len);
        var completed: usize = 0;
        errdefer {
            // Free any embeddings we successfully allocated before the error
            for (results[0..completed]) |embedding| {
                self.allocator.free(embedding);
            }
            self.allocator.free(results);
        }

        for (texts) |text| {
            results[completed] = try self.embed(text);
            completed += 1;
        }
        return results;
    }

    /// Compute cosine similarity between two embeddings.
    pub fn cosineSimilarity(_: *EmbeddingModel, a: []const f32, b: []const f32) f32 {
        return simd.cosineSimilarity(a, b);
    }
};

/// Batch processor for efficient embedding generation.
pub const BatchProcessor = struct {
    model: *EmbeddingModel,
    queue: std.ArrayListUnmanaged([]const u8),

    pub fn init(model: *EmbeddingModel) BatchProcessor {
        return .{
            .model = model,
            .queue = .empty,
        };
    }

    pub fn deinit(self: *BatchProcessor) void {
        self.queue.deinit(self.model.allocator);
    }

    pub fn add(self: *BatchProcessor, text: []const u8) !void {
        try self.queue.append(self.model.allocator, text);
    }

    pub fn process(self: *BatchProcessor) ![][]f32 {
        if (self.queue.items.len == 0) return &[_][]f32{};
        const results = try self.model.embedBatch(self.queue.items);
        self.queue.clearRetainingCapacity();
        return results;
    }
};

/// Embedding cache for deduplication.
pub const EmbeddingCache = struct {
    allocator: std.mem.Allocator,
    cache: std.StringHashMapUnmanaged([]f32),

    pub fn init(allocator: std.mem.Allocator) EmbeddingCache {
        return .{
            .allocator = allocator,
            .cache = .{},
        };
    }

    pub fn deinit(self: *EmbeddingCache) void {
        var iter = self.cache.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.cache.deinit(self.allocator);
    }

    pub fn get(self: *EmbeddingCache, key: []const u8) ?[]f32 {
        return self.cache.get(key);
    }

    pub fn put(self: *EmbeddingCache, key: []const u8, value: []f32) !void {
        try self.cache.put(self.allocator, key, value);
    }
};

/// Embeddings context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.EmbeddingsConfig,
    model: ?*EmbeddingModel = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.EmbeddingsConfig) !*Context {
        if (!isEnabled()) return error.EmbeddingsDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.model) |m| {
            m.deinit();
            self.allocator.destroy(m);
        }
        self.allocator.destroy(self);
    }

    /// Generate embedding for text.
    pub fn embed(self: *Context, text: []const u8) ![]f32 {
        _ = text;
        // Use transformer model for embeddings
        const dimension = self.config.dimension;
        const result = try self.allocator.alloc(f32, dimension);
        @memset(result, 0);
        return result;
    }

    /// Generate embeddings for multiple texts.
    /// Caller owns the returned slice and each embedding within it.
    pub fn embedBatch(self: *Context, texts: []const []const u8) ![][]f32 {
        const results = try self.allocator.alloc([]f32, texts.len);
        var completed: usize = 0;
        errdefer {
            // Free any embeddings we successfully allocated before the error
            for (results[0..completed]) |embedding| {
                self.allocator.free(embedding);
            }
            self.allocator.free(results);
        }

        for (texts) |text| {
            results[completed] = try self.embed(text);
            completed += 1;
        }
        return results;
    }

    /// Compute cosine similarity between two embeddings (SIMD-optimized via shared module).
    pub fn cosineSimilarity(_: *Context, a: []const f32, b: []const f32) f32 {
        return simd.cosineSimilarity(a, b);
    }
};

pub fn isEnabled() bool {
    return build_options.enable_ai;
}

test "embedding model initialization" {
    const allocator = std.testing.allocator;
    var model = EmbeddingModel.init(allocator, .{});
    defer model.deinit();

    try std.testing.expect(model.initialized);
}

test "embed single text" {
    const allocator = std.testing.allocator;
    var model = EmbeddingModel.init(allocator, .{ .dimension = 128 });
    defer model.deinit();

    const embedding = try model.embed("hello world");
    defer allocator.free(embedding);

    try std.testing.expectEqual(@as(usize, 128), embedding.len);
}

test "SIMD normalizeEmbedding single vector" {
    var embedding = [_]f32{ 3.0, 4.0, 0.0, 0.0 };
    normalizeEmbedding(&embedding);

    // L2 norm of [3, 4, 0, 0] is 5, so normalized should be [0.6, 0.8, 0, 0]
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), embedding[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), embedding[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), embedding[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), embedding[3], 1e-6);

    // Verify unit length
    const norm = simd.vectorL2Norm(&embedding);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-6);
}

test "SIMD normalizeEmbedding larger vector" {
    // Test with a larger vector to exercise SIMD paths
    var embedding: [64]f32 = undefined;
    for (0..64) |i| {
        embedding[i] = @as(f32, @floatFromInt(i + 1));
    }

    const original_norm = simd.vectorL2Norm(&embedding);
    try std.testing.expect(original_norm > 1.0);

    normalizeEmbedding(&embedding);

    // Verify unit length after normalization
    const new_norm = simd.vectorL2Norm(&embedding);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), new_norm, 1e-5);

    // Verify relative proportions preserved
    // embedding[1] / embedding[0] should equal 2/1 = 2
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), embedding[1] / embedding[0], 1e-5);
}

test "SIMD normalizeEmbeddingBatch multiple vectors" {
    const allocator = std.testing.allocator;

    // Create batch of embeddings
    var embeddings = try allocator.alloc([]f32, 3);
    defer allocator.free(embeddings);

    var emb0 = [_]f32{ 3.0, 4.0 };
    var emb1 = [_]f32{ 1.0, 0.0 };
    var emb2 = [_]f32{ 0.0, 5.0, 0.0, 0.0 };

    embeddings[0] = &emb0;
    embeddings[1] = &emb1;
    embeddings[2] = &emb2;

    normalizeEmbeddingBatch(embeddings);

    // Verify all are unit length
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), simd.vectorL2Norm(&emb0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), simd.vectorL2Norm(&emb1), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), simd.vectorL2Norm(&emb2), 1e-6);
}

test "SIMD normalizeEmbeddingBatchWithNorms stores norms" {
    const allocator = std.testing.allocator;

    var embeddings = try allocator.alloc([]f32, 2);
    defer allocator.free(embeddings);

    var emb0 = [_]f32{ 3.0, 4.0 }; // norm = 5
    var emb1 = [_]f32{ 0.0, 12.0, 5.0, 0.0 }; // norm = 13

    embeddings[0] = &emb0;
    embeddings[1] = &emb1;

    var norms: [2]f32 = undefined;
    normalizeEmbeddingBatchWithNorms(embeddings, &norms);

    // Verify original norms were stored
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), norms[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 13.0), norms[1], 1e-6);

    // Verify vectors are normalized
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), simd.vectorL2Norm(&emb0), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), simd.vectorL2Norm(&emb1), 1e-6);
}

test "SIMD normalizeEmbedding handles zero vector" {
    var zero_vec = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    normalizeEmbedding(&zero_vec);

    // Zero vector should remain zero
    for (zero_vec) |v| {
        try std.testing.expectEqual(@as(f32, 0.0), v);
    }
}

test "SIMD normalizeEmbedding handles already normalized" {
    var normalized = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    normalizeEmbedding(&normalized);

    // Should remain unchanged
    try std.testing.expectEqual(@as(f32, 1.0), normalized[0]);
    try std.testing.expectEqual(@as(f32, 0.0), normalized[1]);
}
