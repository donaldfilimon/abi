//! Embeddings Stub Module
//!
//! Stub implementation when Embeddings is disabled at compile time.

const std = @import("std");
const config_module = @import("../../config.zig");

pub const Error = error{
    EmbeddingsDisabled,
    ModelNotFound,
    EmbeddingFailed,
    InvalidInput,
};

/// Configuration for embedding models.
pub const EmbeddingConfig = struct {
    dimension: usize = 384,
    max_seq_len: usize = 512,
    batch_size: usize = 32,
    normalize: bool = true,
    model_id: []const u8 = "default",
};

/// Embedding model stub.
pub const EmbeddingModel = struct {
    pub fn init(_: std.mem.Allocator, _: EmbeddingConfig) EmbeddingModel {
        return .{};
    }

    pub fn deinit(_: *EmbeddingModel) void {}

    pub fn embed(_: *EmbeddingModel, _: []const u8) Error![]f32 {
        return error.EmbeddingsDisabled;
    }

    pub fn embedBatch(_: *EmbeddingModel, _: []const []const u8) Error![][]f32 {
        return error.EmbeddingsDisabled;
    }

    pub fn cosineSimilarity(_: *EmbeddingModel, _: []const f32, _: []const f32) f32 {
        return 0;
    }
};

/// Batch processor stub.
pub const BatchProcessor = struct {
    pub fn init(_: *EmbeddingModel) BatchProcessor {
        return .{};
    }

    pub fn deinit(_: *BatchProcessor) void {}

    pub fn add(_: *BatchProcessor, _: []const u8) Error!void {
        return error.EmbeddingsDisabled;
    }

    pub fn process(_: *BatchProcessor) Error![][]f32 {
        return error.EmbeddingsDisabled;
    }
};

/// Embedding cache stub.
pub const EmbeddingCache = struct {
    pub fn init(_: std.mem.Allocator) EmbeddingCache {
        return .{};
    }

    pub fn deinit(_: *EmbeddingCache) void {}

    pub fn get(_: *EmbeddingCache, _: []const u8) ?[]f32 {
        return null;
    }

    pub fn put(_: *EmbeddingCache, _: []const u8, _: []f32) Error!void {
        return error.EmbeddingsDisabled;
    }
};

/// Public API Context wrapper
pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.EmbeddingsConfig) Error!*Context {
        return error.EmbeddingsDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn embed(_: *Context, _: []const u8) Error![]f32 {
        return error.EmbeddingsDisabled;
    }

    pub fn embedBatch(_: *Context, _: []const []const u8) Error![][]f32 {
        return error.EmbeddingsDisabled;
    }

    pub fn cosineSimilarity(_: *Context, _: []const f32, _: []const f32) f32 {
        return 0;
    }
};

pub fn isEnabled() bool {
    return false;
}
