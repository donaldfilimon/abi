//! Transformer stub — active when AI feature is disabled.

const std = @import("std");

pub const TransformerError = error{
    InvalidConfiguration,
    EmptyInput,
    ContextTooLarge,
    OutOfMemory,
    FeatureDisabled,
};

pub const TransformerConfig = struct {
    layers: u16 = 4,
    hidden_size: u16 = 256,
    intermediate_size: u16 = 1024,
    vocab_size: u32 = 8192,
    max_tokens: u32 = 128,
    num_heads: u16 = 8,
    seed: u64 = 0x2a9d_7d3c_b1e5_4f03,
    temperature: f32 = 0.8,
    top_p: f32 = 0.9,
    top_k: u32 = 40,

    pub fn validate(_: TransformerConfig) TransformerError!void {
        return TransformerError.FeatureDisabled;
    }
};

pub const TransformerModel = struct {
    allocator: std.mem.Allocator = undefined,
    config: TransformerConfig = .{},
    embedding_weights: []f32 = &.{},
    layer_query_weights: [][]f32 = &.{},
    layer_key_weights: [][]f32 = &.{},
    layer_value_weights: [][]f32 = &.{},
    layer_output_weights: [][]f32 = &.{},
    layer_ff_0_weights: [][]f32 = &.{},
    layer_ff_1_weights: [][]f32 = &.{},
    lm_head_weights: []f32 = &.{},
    rng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),

    pub fn init(_: std.mem.Allocator, _: TransformerConfig) TransformerError!TransformerModel {
        return TransformerError.FeatureDisabled;
    }

    pub fn deinit(_: *TransformerModel) void {}

    pub fn embed(_: *TransformerModel, _: std.mem.Allocator, _: []const u8) ![]f32 {
        return error.FeatureDisabled;
    }

    pub fn infer(_: *TransformerModel, _: std.mem.Allocator, _: []const u8) ![]u8 {
        return error.FeatureDisabled;
    }

    pub fn encode(_: *const TransformerModel, _: std.mem.Allocator, _: []const u8) ![]u32 {
        return error.FeatureDisabled;
    }

    pub fn decode(_: *const TransformerModel, _: std.mem.Allocator, _: []const u32) ![]u8 {
        return error.FeatureDisabled;
    }

    pub fn generate(_: *TransformerModel, _: std.mem.Allocator, _: []const u32, _: u32) ![]u32 {
        return error.FeatureDisabled;
    }
};
