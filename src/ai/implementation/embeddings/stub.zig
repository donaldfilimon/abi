//! Stub implementation for embeddings when AI features are disabled.

const std = @import("std");

/// Stub configuration for embedding models.
pub const EmbeddingConfig = struct {
    dimension: usize = 384,
    max_seq_len: usize = 512,
    batch_size: usize = 32,
    normalize: bool = true,
    model_id: []const u8 = "default",
};

/// Stub embedding model.
pub const EmbeddingModel = struct {
    allocator: std.mem.Allocator,
    config: EmbeddingConfig,
    initialized: bool = false,

    pub fn init(allocator: std.mem.Allocator, config: EmbeddingConfig) EmbeddingModel {
        return .{
            .allocator = allocator,
            .config = config,
            .initialized = false,
        };
    }

    pub fn deinit(self: *EmbeddingModel) void {
        _ = self;
    }

    pub fn embed(self: *EmbeddingModel, text: []const u8) ![]f32 {
        _ = self;
        _ = text;
        return error.EmbeddingsDisabled;
    }

    pub fn embedBatch(self: *EmbeddingModel, texts: []const []const u8) ![][]f32 {
        _ = self;
        _ = texts;
        return error.EmbeddingsDisabled;
    }

    pub fn cosineSimilarity(_: *EmbeddingModel, a: []const f32, b: []const f32) f32 {
        _ = a;
        _ = b;
        return 0;
    }
};

/// Stub batch processor.
pub const BatchProcessor = struct {
    model: *EmbeddingModel,
    queue: std.ArrayListUnmanaged([]const u8),

    pub fn init(model: *EmbeddingModel) BatchProcessor {
        return .{
            .model = model,
            .queue = .{},
        };
    }

    pub fn deinit(self: *BatchProcessor) void {
        _ = self;
    }

    pub fn add(self: *BatchProcessor, text: []const u8) !void {
        _ = self;
        _ = text;
        return error.EmbeddingsDisabled;
    }

    pub fn process(self: *BatchProcessor) ![][]f32 {
        _ = self;
        return error.EmbeddingsDisabled;
    }
};

/// Stub embedding cache.
pub const EmbeddingCache = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) EmbeddingCache {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *EmbeddingCache) void {
        _ = self;
    }

    pub fn get(self: *EmbeddingCache, key: []const u8) ?[]f32 {
        _ = self;
        _ = key;
        return null;
    }

    pub fn put(self: *EmbeddingCache, key: []const u8, value: []f32) !void {
        _ = self;
        _ = key;
        _ = value;
        return error.EmbeddingsDisabled;
    }
};
