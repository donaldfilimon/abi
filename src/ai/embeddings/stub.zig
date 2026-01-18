//! Embeddings Stub Module
//!
//! Stub implementation when Embeddings is disabled at compile time.
//! Re-exports from implementation stub and adds Context wrapper.

const std = @import("std");
const config_module = @import("../../config.zig");

// Re-export all types from implementation stub
const impl_stub = @import("../implementation/embeddings/stub.zig");

pub const EmbeddingConfig = impl_stub.EmbeddingConfig;
pub const EmbeddingModel = impl_stub.EmbeddingModel;
pub const BatchProcessor = impl_stub.BatchProcessor;
pub const EmbeddingCache = impl_stub.EmbeddingCache;

pub const Error = error{
    EmbeddingsDisabled,
    ModelNotFound,
    EmbeddingFailed,
    InvalidInput,
};

/// Public API Context wrapper (specific to this stub)
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
