//! Checkpoint save/load for LLM training sessions.
//!
//! Delegates to `llm_checkpoint.zig` for the binary format;
//! this module provides the higher-level trainer-aware wrappers.

const std = @import("std");
const trainable_model = @import("../trainable_model.zig");
const llm_checkpoint = @import("../llm_checkpoint.zig");
const types = @import("types.zig");

/// Save a training checkpoint to disk.
pub fn saveCheckpoint(
    allocator: std.mem.Allocator,
    model: *trainable_model.TrainableModel,
    config: types.LlmTrainingConfig,
    optimizer_state: *const types.OptimizerState,
    stats: *const types.TrainingStats,
    checkpoints_saved: *u32,
) !void {
    if (config.checkpoint_path) |path| {
        const weights = try model.collectWeights();
        defer allocator.free(weights);

        const m = optimizer_state.m orelse return error.OutOfMemory;
        const v = optimizer_state.v orelse return error.OutOfMemory;

        var io_backend = std.Io.Threaded.init(allocator, .{ .environ = std.process.Environ.empty });
        defer io_backend.deinit();
        const io = io_backend.io();
        try std.Io.Dir.cwd().createDirPath(io, path);

        const filename = try std.fmt.allocPrint(allocator, "{s}/llm_step_{d}.ckpt", .{
            path,
            stats.global_step,
        });
        defer allocator.free(filename);

        try llm_checkpoint.saveLlmCheckpoint(allocator, filename, .{
            .step = stats.global_step,
            .epoch = stats.epoch,
            .tokens_processed = stats.tokens_processed,
            .weights = weights,
            .m = m,
            .v = v,
        });

        checkpoints_saved.* += 1;
        std.log.info("Checkpoint saved at step {d}", .{stats.global_step});
    }
}

/// Load a checkpoint and restore trainer state.
pub fn loadCheckpoint(
    allocator: std.mem.Allocator,
    model: *trainable_model.TrainableModel,
    optimizer_state: *types.OptimizerState,
    stats: *types.TrainingStats,
    path: []const u8,
) !void {
    var ckpt = try llm_checkpoint.loadLlmCheckpoint(allocator, path);
    defer ckpt.deinit(allocator);

    const expected = model.numParams();
    if (ckpt.weights.len != expected or ckpt.m.len != expected or ckpt.v.len != expected) {
        return error.ConfigMismatch;
    }

    try model.distributeWeights(ckpt.weights);

    if (optimizer_state.m) |m| {
        @memcpy(m, ckpt.m);
    }
    if (optimizer_state.v) |v| {
        @memcpy(v, ckpt.v);
    }

    stats.global_step = ckpt.step;
    stats.epoch = ckpt.epoch;
    stats.tokens_processed = ckpt.tokens_processed;

    std.log.info("Checkpoint loaded from {s} (step {d})", .{ path, ckpt.step });
}

test {
    std.testing.refAllDecls(@This());
}
