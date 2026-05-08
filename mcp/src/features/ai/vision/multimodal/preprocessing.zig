//! Multi-Modal Preprocessing and Embedding Space
//!
//! Configuration types for multi-modal fusion, and unified embedding space
//! for storing and retrieving embeddings across modalities (image, text,
//! audio, video, document).

const std = @import("std");
const vit = @import("../vit.zig");
const simd = @import("../../../../foundation/mod.zig").simd;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for multi-modal fusion
pub const MultiModalConfig = struct {
    /// Vision encoder configuration
    vision_config: vit.ViTConfig = vit.ViTConfig.base(224, 16),

    /// Text encoder hidden size
    text_hidden_size: u32 = 512,

    /// Text encoder number of layers
    text_num_layers: u32 = 12,

    /// Text encoder number of attention heads
    text_num_heads: u32 = 8,

    /// Projection dimension for contrastive learning
    projection_dim: u32 = 512,

    /// Temperature for contrastive loss
    temperature: f32 = 0.07,

    /// Whether to use learned temperature
    learnable_temperature: bool = true,

    /// Maximum text sequence length
    max_text_length: u32 = 77,

    /// Vocabulary size for text encoder
    vocab_size: u32 = 49408,

    /// Dropout probability
    dropout: f32 = 0.0,

    /// Use cross-attention fusion
    use_cross_attention: bool = true,

    /// Number of cross-attention layers
    cross_attention_layers: u32 = 6,
};

// ============================================================================
// Unified Multi-Modal Embeddings
// ============================================================================

/// Unified embedding space for multiple modalities
pub const UnifiedEmbeddingSpace = struct {
    allocator: std.mem.Allocator,
    embedding_dim: u32,

    /// Store embeddings with their modality type
    embeddings: std.ArrayListUnmanaged(EmbeddingEntry),

    /// Data modalities for unified embedding; models can process and generate all types.
    pub const Modality = enum {
        image,
        text,
        document,
        audio,
        video,
        /// Arbitrary / other data (any type passed in)
        other,
    };

    pub const EmbeddingEntry = struct {
        id: u64,
        modality: Modality,
        embedding: []f32,
        metadata: ?[]const u8,
    };

    pub fn init(allocator: std.mem.Allocator, embedding_dim: u32) UnifiedEmbeddingSpace {
        return .{
            .allocator = allocator,
            .embedding_dim = embedding_dim,
            .embeddings = .empty,
        };
    }

    pub fn deinit(self: *UnifiedEmbeddingSpace) void {
        for (self.embeddings.items) |entry| {
            self.allocator.free(entry.embedding);
            if (entry.metadata) |meta| self.allocator.free(meta);
        }
        self.embeddings.deinit(self.allocator);
    }

    /// Add an embedding to the space
    pub fn addEmbedding(
        self: *UnifiedEmbeddingSpace,
        id: u64,
        modality: Modality,
        embedding: []const f32,
        metadata: ?[]const u8,
    ) !void {
        if (embedding.len != self.embedding_dim) return error.DimensionMismatch;

        const embed_copy = try self.allocator.dupe(f32, embedding);
        errdefer self.allocator.free(embed_copy);

        const meta_copy = if (metadata) |m| try self.allocator.dupe(u8, m) else null;

        try self.embeddings.append(self.allocator, .{
            .id = id,
            .modality = modality,
            .embedding = embed_copy,
            .metadata = meta_copy,
        });
    }

    /// Find nearest neighbors across all modalities
    pub fn findNearest(
        self: *const UnifiedEmbeddingSpace,
        query: []const f32,
        k: usize,
        filter_modality: ?Modality,
    ) ![]const EmbeddingEntry {
        if (query.len != self.embedding_dim) return error.DimensionMismatch;

        const max_results = @min(k, self.embeddings.items.len);
        if (max_results == 0) return &[_]EmbeddingEntry{};

        // Compute similarities
        const Scored = struct {
            idx: usize,
            score: f32,
        };

        var scores = std.ArrayListUnmanaged(Scored).empty;
        defer scores.deinit(self.allocator);

        for (self.embeddings.items, 0..) |entry, idx| {
            if (filter_modality) |mod| {
                if (entry.modality != mod) continue;
            }

            const sim = simd.cosineSimilarity(query, entry.embedding);
            try scores.append(self.allocator, .{ .idx = idx, .score = sim });
        }

        // Sort by similarity (descending)
        std.mem.sort(Scored, scores.items, {}, struct {
            fn lessThan(_: void, a: Scored, b: Scored) bool {
                return a.score > b.score;
            }
        }.lessThan);

        // Return top-k
        const results = try self.allocator.alloc(EmbeddingEntry, @min(max_results, scores.items.len));
        for (0..results.len) |i| {
            results[i] = self.embeddings.items[scores.items[i].idx];
        }

        return results;
    }

    /// Cross-modal retrieval: find items of target modality similar to query
    pub fn crossModalRetrieve(
        self: *const UnifiedEmbeddingSpace,
        query: []const f32,
        target_modality: Modality,
        k: usize,
    ) ![]const EmbeddingEntry {
        return self.findNearest(query, k, target_modality);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "UnifiedEmbeddingSpace operations" {
    const allocator = std.testing.allocator;

    var space = UnifiedEmbeddingSpace.init(allocator, 64);
    defer space.deinit();

    const embed1 = try allocator.alloc(f32, 64);
    defer allocator.free(embed1);
    for (embed1) |*v| v.* = 0.5;

    try space.addEmbedding(1, .image, embed1, "test image");
    try space.addEmbedding(2, .text, embed1, "test text");

    try std.testing.expectEqual(@as(usize, 2), space.embeddings.items.len);

    // Find nearest
    const results = try space.findNearest(embed1, 2, null);
    defer allocator.free(results);
    try std.testing.expect(results.len > 0);
}

test "MultiModalConfig defaults" {
    const config = MultiModalConfig{};

    try std.testing.expectEqual(@as(u32, 512), config.projection_dim);
    try std.testing.expectEqual(@as(f32, 0.07), config.temperature);
    try std.testing.expect(config.learnable_temperature);
}

test {
    std.testing.refAllDecls(@This());
}
