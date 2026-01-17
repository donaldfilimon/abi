//! LLM Stub Module
//!
//! Stub implementation when LLM is disabled at compile time.

const std = @import("std");
const config_module = @import("../../config.zig");

pub const Error = error{
    LlmDisabled,
    ModelNotFound,
    ModelLoadFailed,
    InferenceFailed,
    TokenizationFailed,
    InvalidConfig,
};

pub const Engine = struct {};
pub const Model = struct {};
pub const InferenceConfig = struct {};
pub const GgufFile = struct {};
pub const BpeTokenizer = struct {};

// Sub-module stubs (for API parity with mod.zig)
pub const io = struct {};
pub const model = struct {};
pub const tensor = struct {};
pub const tokenizer = struct {};
pub const ops = struct {};
pub const cache = struct {};
pub const generation = struct {};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.LlmConfig) error{LlmDisabled}!*Context {
        return error.LlmDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn getEngine(_: *Context) Error!*Engine {
        return error.LlmDisabled;
    }

    pub fn generate(_: *Context, _: []const u8) Error![]u8 {
        return error.LlmDisabled;
    }

    pub fn tokenize(_: *Context, _: []const u8) Error![]u32 {
        return error.LlmDisabled;
    }

    pub fn detokenize(_: *Context, _: []const u32) Error![]u8 {
        return error.LlmDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}
