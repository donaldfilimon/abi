//! nn stub -- disabled at compile time.
//!
//! Mirrors the public API surface of `mod.zig` so that code referencing
//! `abi.features.nn` keeps compiling when the feature is off. Unlike the hash
//! stub, the trainer cannot produce meaningful output while disabled, so the
//! producing/consuming entry points return `error.FeatureDisabled`. The
//! "disabled" signal is `isEnabled() == false`.

pub const types = @import("types.zig");

const std = @import("std");

pub const NnError = types.NnError;
pub const Error = types.Error;
pub const TrainConfig = types.TrainConfig;
pub const TrainReport = types.TrainReport;

/// Disabled-path placeholder so consumers can still name `nn.Model`.
pub const Model = struct {};

pub fn isEnabled() bool {
    return false;
}

pub fn trainModel(allocator: std.mem.Allocator, text: []const u8, config: TrainConfig) !Model {
    _ = allocator;
    _ = text;
    _ = config;
    return error.FeatureDisabled;
}

pub fn trainOnText(allocator: std.mem.Allocator, text: []const u8, config: TrainConfig) !TrainReport {
    _ = allocator;
    _ = text;
    _ = config;
    return error.FeatureDisabled;
}

pub fn sample(allocator: std.mem.Allocator, model: *const Model, seed_char: u8, n: usize) ![]u8 {
    _ = allocator;
    _ = model;
    _ = seed_char;
    _ = n;
    return error.FeatureDisabled;
}

test {
    std.testing.refAllDecls(@This());
}

test "nn stub reports disabled" {
    try std.testing.expect(!isEnabled());
}

test "nn stub refuses to train or sample" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.FeatureDisabled, trainOnText(a, "hello world ", .{}));
    var model = Model{};
    try std.testing.expectError(error.FeatureDisabled, sample(a, &model, 'h', 4));
}
