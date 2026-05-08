//! Evaluation and early stopping for LLM training.

const std = @import("std");
const trainable_model = @import("../trainable_model.zig");
const loss_mod = @import("../loss.zig");
const logging = @import("../logging.zig");
const training_bridge = @import("../../../gpu/training_bridge.zig");
const types = @import("types.zig");

/// Evaluate model on validation data.
/// Runs in inference mode (no gradient computation).
pub fn evaluate(
    allocator: std.mem.Allocator,
    model: *trainable_model.TrainableModel,
    loss_fn: *loss_mod.CrossEntropyLoss,
    gpu_bridge: *?training_bridge.GpuTrainingBridge,
    config: types.LlmTrainingConfig,
    data: []const u32,
) !types.EvalResult {
    const tokens_per_sample = config.max_seq_len;
    const batch_tokens = config.batch_size * tokens_per_sample;

    var total_loss: f32 = 0;
    var total_correct: u64 = 0;
    var total_tokens: u64 = 0;
    var num_batches: u32 = 0;

    // Ensure model is prepared for inference
    if (model.activations == null) {
        try model.prepareForTraining(@intCast(batch_tokens - 1));
    }

    var offset: usize = 0;
    while (offset + batch_tokens < data.len) {
        const batch_data = data[offset .. offset + batch_tokens];

        const input_ids = batch_data[0 .. batch_tokens - 1];
        const labels = batch_data[1..batch_tokens];

        // Forward pass (inference only)
        const logits = try allocator.alloc(f32, (batch_tokens - 1) * model.config.vocab_size);
        defer allocator.free(logits);

        if (gpu_bridge.*) |*bridge| {
            try model.forwardGpu(input_ids, logits, bridge);
        } else {
            try model.forward(input_ids, logits);
        }

        // Compute loss
        const loss = try loss_fn.forward(logits, labels);
        total_loss += loss;

        // Compute accuracy (argmax prediction vs label)
        const vocab_size = model.config.vocab_size;
        for (0..batch_tokens - 1) |i| {
            const logit_start = i * vocab_size;
            var max_idx: u32 = 0;
            var max_val: f32 = logits[logit_start];
            for (1..vocab_size) |j| {
                if (logits[logit_start + j] > max_val) {
                    max_val = logits[logit_start + j];
                    max_idx = @intCast(j);
                }
            }
            if (max_idx == labels[i]) {
                total_correct += 1;
            }
            total_tokens += 1;
        }

        num_batches += 1;
        offset += batch_tokens;
    }

    if (num_batches == 0) {
        return types.EvalResult{
            .loss = 0,
            .perplexity = 1,
            .accuracy = 0,
            .num_batches = 0,
            .total_tokens = 0,
        };
    }

    const avg_loss = total_loss / @as(f32, @floatFromInt(num_batches));
    const accuracy = if (total_tokens > 0)
        @as(f32, @floatFromInt(total_correct)) / @as(f32, @floatFromInt(total_tokens))
    else
        0;

    return types.EvalResult{
        .loss = avg_loss,
        .perplexity = @exp(avg_loss),
        .accuracy = accuracy,
        .num_batches = num_batches,
        .total_tokens = total_tokens,
    };
}

/// Run validation if interval reached and data available.
/// Returns true if training should continue (not early stopped).
pub fn maybeValidate(
    allocator: std.mem.Allocator,
    model: *trainable_model.TrainableModel,
    loss_fn: *loss_mod.CrossEntropyLoss,
    gpu_bridge: *?training_bridge.GpuTrainingBridge,
    config: types.LlmTrainingConfig,
    stats: *const types.TrainingStats,
    val_data: ?[]const u32,
    early_stop_config: types.EarlyStoppingConfig,
    early_stopping: *types.EarlyStoppingState,
    best_val_accuracy: *f32,
    best_weights: *?[]f32,
    logger: *?logging.TrainingLogger,
) !bool {
    // Check if we should validate
    if (config.eval_interval == 0 or
        stats.global_step % config.eval_interval != 0)
    {
        return true;
    }

    const vd = val_data orelse return true;

    // Run evaluation
    const result = try evaluate(allocator, model, loss_fn, gpu_bridge, config, vd);
    std.log.info("Validation: loss={d:.4} ppl={d:.2} acc={d:.2}%", .{
        result.loss,
        result.perplexity,
        result.accuracy * 100,
    });

    if (logger.*) |*lg| {
        try lg.logScalar("val/loss", result.loss, stats.global_step);
        try lg.logScalar("val/perplexity", result.perplexity, stats.global_step);
        try lg.logScalar("val/accuracy", result.accuracy, stats.global_step);
    }

    // Check early stopping
    if (!early_stop_config.enabled) return true;

    const metric = if (early_stop_config.monitor_loss) result.loss else result.perplexity;

    if (metric < early_stopping.best_metric - early_stop_config.min_delta) {
        // Improvement found
        early_stopping.best_metric = metric;
        early_stopping.patience_counter = 0;
        best_val_accuracy.* = result.accuracy;

        // Save best weights
        if (best_weights.*) |bw| allocator.free(bw);
        best_weights.* = try model.collectWeights();

        std.log.info("New best model (metric={d:.4})", .{metric});
    } else {
        // No improvement
        early_stopping.patience_counter += 1;
        if (early_stopping.patience_counter >= early_stop_config.patience) {
            std.log.info("Early stopping triggered (patience={d})", .{early_stop_config.patience});
            early_stopping.stopped = true;

            // Restore best weights
            if (best_weights.*) |bw| {
                try model.distributeWeights(bw);
                std.log.info("Restored best weights", .{});
            }

            return false;
        }
    }

    return true;
}

/// Check if training was early stopped.
pub fn wasEarlyStopped(early_stopping: *const types.EarlyStoppingState) bool {
    return early_stopping.stopped;
}

test {
    std.testing.refAllDecls(@This());
}
