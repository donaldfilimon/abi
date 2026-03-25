//! Embeddings Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");

// Shared types (canonical definitions from types.zig).
pub const types = @import("types.zig");
pub const BackendType = types.BackendType;
pub const BackendError = types.BackendError;
pub const BackendConfig = types.BackendConfig;
pub const EmbeddingConfig = types.EmbeddingConfig;

pub const Error = error{ FeatureDisabled, ModelNotFound, EmbeddingFailed, InvalidInput };
pub const EmbeddingBackend = struct {
    backend_type: BackendType = .local,
    pub fn embed(_: EmbeddingBackend, _: std.mem.Allocator, _: []const u8, _: usize) BackendError![]f32 {
        return BackendError.BackendNotAvailable;
    }
    pub fn embedBatch(_: EmbeddingBackend, _: std.mem.Allocator, _: []const []const u8, _: usize) BackendError![][]f32 {
        return BackendError.BackendNotAvailable;
    }
    pub fn deinit(_: EmbeddingBackend) void {}
};

pub const backend = struct {
    pub const EmbeddingBackend_ = EmbeddingBackend;
    pub const BackendError_ = BackendError;
    pub const BackendType_ = BackendType;
    pub const BackendConfig_ = BackendConfig;
};

pub const backends = struct {};

pub fn normalizeEmbedding(_: []f32) void {}
pub fn normalizeEmbeddingBatch(_: [][]f32) void {}
pub fn normalizeEmbeddingBatchWithNorms(_: [][]f32, _: ?[]f32) void {}

// EmbeddingConfig is re-exported from types.zig above.

pub const EmbeddingModel = struct {
    pub fn init(_: std.mem.Allocator, _: EmbeddingConfig) EmbeddingModel {
        return .{};
    }
    pub fn deinit(_: *EmbeddingModel) void {}
    pub fn embed(_: *EmbeddingModel, _: []const u8) Error![]f32 {
        return error.FeatureDisabled;
    }
    pub fn embedBatch(_: *EmbeddingModel, _: []const []const u8) Error![][]f32 {
        return error.FeatureDisabled;
    }
    pub fn cosineSimilarity(_: *EmbeddingModel, _: []const f32, _: []const f32) f32 {
        return 0;
    }
};

pub const BatchProcessor = struct {
    pub fn init(_: *EmbeddingModel) BatchProcessor {
        return .{};
    }
    pub fn deinit(_: *BatchProcessor) void {}
    pub fn add(_: *BatchProcessor, _: []const u8) Error!void {
        return error.FeatureDisabled;
    }
    pub fn process(_: *BatchProcessor) Error![][]f32 {
        return error.FeatureDisabled;
    }
};

pub const EmbeddingCache = struct {
    pub fn init(_: std.mem.Allocator) EmbeddingCache {
        return .{};
    }
    pub fn deinit(_: *EmbeddingCache) void {}
    pub fn get(_: *EmbeddingCache, _: []const u8) ?[]f32 {
        return null;
    }
    pub fn put(_: *EmbeddingCache, _: []const u8, _: []f32) Error!void {
        return error.FeatureDisabled;
    }
};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.EmbeddingsConfig) Error!*Context {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn embed(_: *Context, _: []const u8) Error![]f32 {
        return error.FeatureDisabled;
    }
    pub fn embedBatch(_: *Context, _: []const []const u8) Error![][]f32 {
        return error.FeatureDisabled;
    }
    pub fn cosineSimilarity(_: *Context, _: []const f32, _: []const f32) f32 {
        return 0;
    }
};

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
