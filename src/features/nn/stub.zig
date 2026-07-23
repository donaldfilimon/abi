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

/// Disabled-path placeholder so consumers can still name `nn.Model` and stay
/// type-compatible with the enabled CLI handler. The `report`/`deinit` members
/// are never reached while disabled (the producers above return
/// `error.FeatureDisabled`); they exist only so a build that drags the handler
/// in under `-Dfeat-nn=false` still compiles.
pub const Model = struct {
    report: TrainReport = std.mem.zeroes(TrainReport),

    pub fn deinit(self: *Model) void {
        self.* = undefined;
    }
};

pub const Scratch = struct {};
pub const Grads = struct {};

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

pub fn extractCorpusFromJsonl(allocator: std.mem.Allocator, bytes: []const u8, field: []const u8) ![]u8 {
    _ = allocator;
    _ = bytes;
    _ = field;
    return error.FeatureDisabled;
}

pub fn trainOnJsonl(allocator: std.mem.Allocator, path: []const u8, field: []const u8, config: TrainConfig) !TrainReport {
    _ = allocator;
    _ = path;
    _ = field;
    _ = config;
    return error.FeatureDisabled;
}

pub const DEFAULT_CHECKPOINT_PATH = "assets/nn/persona-checkpoint.bin";
pub const BUNDLED_CORPUS = "";
pub const MAX_OUTPUT_CHARS = @as(usize, 300);

pub fn saveModelAlloc(allocator: std.mem.Allocator, model: *const Model) ![]u8 {
    _ = allocator;
    _ = model;
    return error.FeatureDisabled;
}

pub fn loadModelBytes(allocator: std.mem.Allocator, bytes: []const u8) !Model {
    _ = allocator;
    _ = bytes;
    return error.FeatureDisabled;
}

pub fn saveModelPath(io: std.Io, allocator: std.mem.Allocator, model: *const Model, path: []const u8) !void {
    _ = io;
    _ = allocator;
    _ = model;
    _ = path;
    return error.FeatureDisabled;
}

pub fn loadModelPath(io: std.Io, allocator: std.mem.Allocator, path: []const u8) (NnError || error{OutOfMemory})!Model {
    _ = io;
    _ = allocator;
    _ = path;
    return error.FeatureDisabled;
}

pub fn trainBundled(allocator: std.mem.Allocator) !Model {
    _ = allocator;
    return error.FeatureDisabled;
}

pub fn sampleStreaming(
    allocator: std.mem.Allocator,
    model: *const Model,
    seed_char: u8,
    max_chars: usize,
    chunk: usize,
    on_chunk: *const fn (*anyopaque, []const u8) anyerror!void,
    ctx: *anyopaque,
) ![]u8 {
    _ = allocator;
    _ = model;
    _ = seed_char;
    _ = max_chars;
    _ = chunk;
    _ = on_chunk;
    _ = ctx;
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
