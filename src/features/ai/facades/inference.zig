//! AI Inference Module — LLM, Embeddings, Vision, Streaming
//!
//! This module provides all inference-time AI capabilities: local LLM execution,
//! embedding generation, vision processing, streaming output, and the transformer
//! engine. Also includes the personas system (which depends on embeddings).
//!
//! Gated by `-Denable-llm`.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../../core/config/mod.zig");

// ============================================================================
// Sub-module re-exports (from features/ai/)
// ============================================================================

pub const llm = if (build_options.enable_llm)
    @import("../llm/mod.zig")
else
    @import("../llm/stub.zig");

pub const embeddings = if (build_options.enable_ai)
    @import("../embeddings/mod.zig")
else
    @import("../embeddings/stub.zig");

pub const vision = if (build_options.enable_vision)
    @import("../vision/mod.zig")
else
    @import("../vision/stub.zig");

pub const streaming = @import("../streaming/mod.zig");
pub const transformer = @import("../transformer/mod.zig");

pub const personas = if (build_options.enable_ai)
    @import("../personas/mod.zig")
else
    @import("../personas/stub.zig");

// ============================================================================
// Convenience type re-exports
// ============================================================================

// LLM Engine
pub const LlmEngine = llm.Engine;
pub const LlmModel = llm.Model;
pub const LlmConfig = llm.InferenceConfig;
pub const GgufFile = llm.GgufFile;
pub const BpeTokenizer = llm.BpeTokenizer;

// Transformer
pub const TransformerConfig = transformer.TransformerConfig;
pub const TransformerModel = transformer.TransformerModel;

// Streaming
pub const StreamingGenerator = streaming.StreamingGenerator;
pub const StreamToken = streaming.StreamToken;
pub const StreamState = streaming.StreamState;
pub const GenerationConfig = streaming.GenerationConfig;
pub const ServerConfig = streaming.ServerConfig;
pub const StreamingServer = streaming.StreamingServer;
pub const StreamingServerError = streaming.StreamingServerError;
pub const BackendType = streaming.BackendType;

// ============================================================================
// Error
// ============================================================================

pub const Error = error{
    LlmDisabled,
    EmbeddingsDisabled,
    InferenceFailed,
    InvalidConfig,
};

// ============================================================================
// Context
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AiConfig,
    llm_ctx: ?*llm.Context = null,
    embeddings_ctx: ?*embeddings.Context = null,
    personas_ctx: ?*personas.Context = null,

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: config_module.AiConfig,
    ) !*Context {
        if (!isEnabled()) return error.LlmDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };

        errdefer ctx.deinitSubFeatures();

        if (cfg.llm) |llm_cfg| {
            ctx.llm_ctx = try llm.Context.init(allocator, llm_cfg);
        }

        if (cfg.embeddings) |emb_cfg| {
            ctx.embeddings_ctx = try embeddings.Context.init(
                allocator,
                emb_cfg,
            );
        }

        if (cfg.personas) |personas_cfg| {
            ctx.personas_ctx = try personas.Context.init(
                allocator,
                personas_cfg,
            );
        }

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        self.deinitSubFeatures();
        self.allocator.destroy(self);
    }

    fn deinitSubFeatures(self: *Context) void {
        if (self.personas_ctx) |p| p.deinit();
        if (self.embeddings_ctx) |e| e.deinit();
        if (self.llm_ctx) |l| l.deinit();
    }

    pub const SubFeature = enum { llm, embeddings, personas };

    pub fn SubFeatureContext(comptime feature: SubFeature) type {
        return switch (feature) {
            .llm => llm.Context,
            .embeddings => embeddings.Context,
            .personas => personas.Context,
        };
    }

    pub fn get(self: *Context, comptime feature: SubFeature) Error!*SubFeatureContext(feature) {
        return @field(self, @tagName(feature) ++ "_ctx") orelse switch (feature) {
            .llm => error.LlmDisabled,
            .embeddings => error.EmbeddingsDisabled,
            .personas => error.EmbeddingsDisabled,
        };
    }
};

// ============================================================================
// Module-level functions
// ============================================================================

pub fn isEnabled() bool {
    return build_options.enable_llm;
}

// ============================================================================
// Tests
// ============================================================================

test "ai_inference module loads" {
    try std.testing.expect(@TypeOf(StreamingGenerator) != void);
    try std.testing.expect(@TypeOf(TransformerConfig) != void);
}

test "ai_inference isEnabled reflects build flag" {
    try std.testing.expectEqual(build_options.enable_llm, isEnabled());
}

test "ai_inference type sizes are reasonable" {
    try std.testing.expect(@sizeOf(TransformerConfig) > 0);
    try std.testing.expect(@sizeOf(GenerationConfig) > 0);
    try std.testing.expect(@sizeOf(StreamToken) > 0);
}
