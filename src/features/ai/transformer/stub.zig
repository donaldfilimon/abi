//! Transformer stub — active when AI feature is disabled.

const std = @import("std");
const types = @import("types.zig");

pub const TransformerError = types.TransformerError || error{FeatureDisabled};

pub const TransformerConfig = types.TransformerConfig;

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

test {
    std.testing.refAllDecls(@This());
}
