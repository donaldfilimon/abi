//! Embeddings Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../core/config/mod.zig");

// Shared types (canonical definitions from types.zig).
pub const types = @import("types.zig");
pub const BackendType = types.BackendType;
pub const BackendError = types.BackendError;
pub const BackendConfig = types.BackendConfig;
pub const EmbeddingConfig = types.EmbeddingConfig;

pub const Error = error{ FeatureDisabled, ModelNotFound, EmbeddingFailed, InvalidInput };

pub const EmbedFn = *const fn (
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    text: []const u8,
    dimensions: usize,
) BackendError![]f32;

pub const EmbedBatchFn = *const fn (
    ctx: *anyopaque,
    allocator: std.mem.Allocator,
    texts: []const []const u8,
    dimensions: usize,
) BackendError![][]f32;

pub const DeinitFn = *const fn (ctx: *anyopaque) void;

pub const EmbeddingBackend = struct {
    ptr: *anyopaque = undefined,
    embedFn: EmbedFn = undefined,
    embedBatchFn: EmbedBatchFn = undefined,
    deinitFn: ?DeinitFn = null,
    backend_type: BackendType = .local,
    name: []const u8 = "stub",
    model: []const u8 = "",
    default_dimensions: usize = 384,

    pub fn embed(_: EmbeddingBackend, _: std.mem.Allocator, _: []const u8, _: usize) BackendError![]f32 {
        return BackendError.BackendNotAvailable;
    }
    pub fn embedBatch(_: EmbeddingBackend, _: std.mem.Allocator, _: []const []const u8, _: usize) BackendError![][]f32 {
        return BackendError.BackendNotAvailable;
    }
    pub fn deinit(_: EmbeddingBackend) void {}
};

pub const backend = struct {
    pub const EmbeddingBackend = @import("stub.zig").EmbeddingBackend;
    pub const BackendError = @import("stub.zig").BackendError;
    pub const BackendType = @import("stub.zig").BackendType;
    pub const BackendConfig = @import("stub.zig").BackendConfig;
    pub const EmbedFn = @import("stub.zig").EmbedFn;
    pub const EmbedBatchFn = @import("stub.zig").EmbedBatchFn;
    pub const DeinitFn = @import("stub.zig").DeinitFn;
};

pub const backends = struct {
    pub const openai = struct {
        pub const OpenAIBackend = struct {
            pub fn init(_: std.mem.Allocator, _: []const u8, _: []const u8) error{FeatureDisabled}!*openai.OpenAIBackend {
                return error.FeatureDisabled;
            }
            pub fn initFromEnv(_: std.mem.Allocator) error{FeatureDisabled}!*openai.OpenAIBackend {
                return error.FeatureDisabled;
            }
            pub fn deinit(_: *openai.OpenAIBackend) void {}
            pub fn asBackend(_: *openai.OpenAIBackend) EmbeddingBackend {
                return .{};
            }
        };
        pub const Model = enum { @"text-embedding-3-small", @"text-embedding-3-large", @"text-embedding-ada-002" };
    };
};

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
    pub fn setBackend(_: *EmbeddingModel, _: EmbeddingBackend) void {}
    pub fn hasBackend(_: *const EmbeddingModel) bool {
        return false;
    }
    pub fn getBackendType(_: *const EmbeddingModel) ?BackendType {
        return null;
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
    pub fn useOpenAI(_: *Context, _: []const u8, _: []const u8) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn useOpenAIFromEnv(_: *Context) error{FeatureDisabled}!void {
        return error.FeatureDisabled;
    }
    pub fn hasBackend(_: *const Context) bool {
        return false;
    }
    pub fn getBackendType(_: *const Context) ?BackendType {
        return null;
    }
};

pub fn isEnabled() bool {
    return false;
}

test {
    std.testing.refAllDecls(@This());
}
