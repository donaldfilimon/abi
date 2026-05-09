//! Training metrics logging for LLM trainer.

const std = @import("std");
const time = @import("../../../../foundation/mod.zig").time;
const logging = @import("../logging.zig");
const training_bridge = @import("../../../gpu/training_bridge.zig");
const types = @import("types.zig");

/// Log training metrics if the interval has been reached.
pub fn maybeLog(
    config: types.LlmTrainingConfig,
    stats: *types.TrainingStats,
    logger: *?logging.TrainingLogger,
    gpu_bridge: *const ?training_bridge.GpuTrainingBridge,
    log_timer: *?time.Timer,
    last_log_time_ns: *u64,
    last_log_tokens: *u64,
) !void {
    if (logger.* == null) return;
    if (config.log_interval == 0) return;
    if (stats.global_step % config.log_interval != 0) return;

    if (log_timer.*) |*timer| {
        const now_ns = timer.read();
        const delta_ns = now_ns - last_log_time_ns.*;
        if (delta_ns > 0) {
            const tokens_delta = stats.tokens_processed - last_log_tokens.*;
            stats.throughput = @as(f32, @floatFromInt(tokens_delta)) /
                (@as(f32, @floatFromInt(delta_ns)) / 1e9);
        }
        last_log_time_ns.* = now_ns;
        last_log_tokens.* = stats.tokens_processed;
    }

    var lg = &logger.*.?;
    try lg.logScalar("train/loss", stats.loss, stats.global_step);
    try lg.logScalar("train/accuracy", stats.accuracy, stats.global_step);
    try lg.logScalar("train/perplexity", stats.perplexity, stats.global_step);
    try lg.logScalar("train/learning_rate", stats.learning_rate, stats.global_step);
    try lg.logScalar("train/grad_norm", stats.grad_norm, stats.global_step);
    try lg.logScalar("train/throughput", stats.throughput, stats.global_step);

    // Log GPU stats if bridge is active
    try logGpuStats(logger, gpu_bridge, stats.global_step);
}

fn logGpuStats(
    logger: *?logging.TrainingLogger,
    gpu_bridge: *const ?training_bridge.GpuTrainingBridge,
    global_step: u64,
) !void {
    var lg = &(logger.* orelse return);
    const gpu_stats = if (gpu_bridge.*) |bridge| bridge.getStats() else training_bridge.GpuTrainingStats{};
    if (gpu_stats.gpu_available) {
        try lg.logScalar("gpu/utilization", gpu_stats.utilization, global_step);
        try lg.logScalar("gpu/kernel_time_ms", gpu_stats.avgKernelTimeMs(), global_step);
        try lg.logScalar("gpu/ops_count", @floatFromInt(gpu_stats.total_gpu_ops), global_step);
        try lg.logScalar("gpu/fallback_count", @floatFromInt(gpu_stats.cpu_fallback_ops), global_step);
    }
}

/// Write final summary metrics to logger.
pub fn finalizeLogging(
    logger: *?logging.TrainingLogger,
    stats: *const types.TrainingStats,
    early_stopping: *const types.EarlyStoppingState,
    best_val_accuracy: f32,
) !void {
    if (logger.*) |*lg| {
        const metrics = [_]logging.Metric{
            .{ .key = "train/final_loss", .value = stats.loss },
            .{ .key = "train/final_accuracy", .value = stats.accuracy },
            .{ .key = "train/final_perplexity", .value = stats.perplexity },
            .{ .key = "val/best_loss", .value = early_stopping.best_metric },
            .{ .key = "val/best_accuracy", .value = best_val_accuracy },
            .{ .key = "train/total_steps", .value = @floatFromInt(stats.global_step) },
            .{ .key = "train/total_tokens", .value = @floatFromInt(stats.tokens_processed) },
        };
        try lg.writeSummary(&metrics);
    }
}

test {
    std.testing.refAllDecls(@This());
}
