//! Embeddings Stub Module

const std = @import("std");
const config_module = @import("../../config.zig");

pub const Error = error{
    EmbeddingsDisabled,
    ModelNotFound,
    EmbeddingFailed,
    InvalidInput,
};

pub const EmbeddingModel = struct {};
pub const EmbeddingConfig = struct {};

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
