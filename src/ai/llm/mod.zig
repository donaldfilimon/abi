//! LLM Sub-module
//!
//! Local LLM inference supporting GGUF models and transformer architectures.
//!
//! ## Features
//! - GGUF model loading
//! - BPE tokenization
//! - Streaming generation
//! - GPU acceleration (when available)

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../config.zig");

// Re-export from existing LLM module
const features_llm = @import("../implementation/llm/mod.zig");

pub const Engine = features_llm.Engine;
pub const Model = features_llm.Model;
pub const InferenceConfig = features_llm.InferenceConfig;
pub const GgufFile = features_llm.GgufFile;
pub const BpeTokenizer = features_llm.BpeTokenizer;

// Sub-modules
pub const io = features_llm.io;
pub const model = features_llm.model;
pub const tensor = features_llm.tensor;
pub const tokenizer = features_llm.tokenizer;
pub const ops = features_llm.ops;
pub const cache = features_llm.cache;
pub const generation = features_llm.generation;

pub const Error = error{
    LlmDisabled,
    ModelNotFound,
    ModelLoadFailed,
    InferenceFailed,
    TokenizationFailed,
    InvalidConfig,
};

/// LLM context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.LlmConfig,
    engine: ?*Engine = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.LlmConfig) !*Context {
        if (!isEnabled()) return error.LlmDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.engine) |e| {
            e.deinit();
        }
        self.allocator.destroy(self);
    }

    /// Get or initialize the LLM engine.
    pub fn getEngine(self: *Context) !*Engine {
        if (self.engine) |e| return e;

        const engine_ptr = try self.allocator.create(Engine);
        engine_ptr.* = try Engine.init(self.allocator, .{
            .context_size = self.config.context_size,
            .batch_size = self.config.batch_size,
        });
        self.engine = engine_ptr;
        return engine_ptr;
    }

    /// Generate text from prompt.
    pub fn generate(self: *Context, prompt: []const u8) ![]u8 {
        const engine = try self.getEngine();
        return engine.generate(self.allocator, prompt);
    }

    /// Tokenize text.
    pub fn tokenize(self: *Context, text: []const u8) ![]u32 {
        const engine = try self.getEngine();
        return engine.tokenize(self.allocator, text);
    }

    /// Detokenize tokens.
    pub fn detokenize(self: *Context, tokens: []const u32) ![]u8 {
        const engine = try self.getEngine();
        return engine.detokenize(self.allocator, tokens);
    }
};

pub fn isEnabled() bool {
    return build_options.enable_llm;
}
