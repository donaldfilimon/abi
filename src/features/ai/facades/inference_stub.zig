//! AI Inference Stub Module — disabled when AI inference is off.

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");

pub const Error = error{ LlmDisabled, EmbeddingsDisabled, InferenceFailed, InvalidConfig };

// Sub-module stubs
pub const llm = @import("../llm/stub.zig");
pub const embeddings = @import("../embeddings/stub.zig");
pub const vision = @import("../vision/stub.zig");
pub const streaming = @import("../streaming/stub.zig");
pub const transformer = @import("../transformer/stub.zig");
pub const personas = @import("../personas/stub.zig");

// Re-exports
pub const LlmEngine = llm.Engine;
pub const LlmModel = llm.Model;
pub const LlmConfig = llm.InferenceConfig;
pub const GgufFile = llm.GgufFile;
pub const BpeTokenizer = llm.BpeTokenizer;
pub const TransformerConfig = transformer.TransformerConfig;
pub const TransformerModel = transformer.TransformerModel;
pub const StreamingGenerator = streaming.StreamingGenerator;
pub const StreamToken = streaming.StreamToken;
pub const StreamState = streaming.StreamState;
pub const GenerationConfig = streaming.GenerationConfig;
pub const ServerConfig = streaming.ServerConfig;
pub const StreamingServer = streaming.StreamingServer;
pub const StreamingServerError = streaming.StreamingServerError;
pub const BackendType = streaming.BackendType;

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AiConfig,
    llm_ctx: ?*llm.Context = null,
    embeddings_ctx: ?*embeddings.Context = null,
    personas_ctx: ?*personas.Context = null,

    pub fn init(_: std.mem.Allocator, _: config_module.AiConfig) !*Context {
        return error.LlmDisabled;
    }
    pub fn deinit(_: *Context) void {}
    fn deinitSubFeatures(_: *Context) void {}
    pub fn getLlm(_: *Context) Error!*llm.Context {
        return error.LlmDisabled;
    }
    pub fn getEmbeddings(_: *Context) Error!*embeddings.Context {
        return error.LlmDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}
