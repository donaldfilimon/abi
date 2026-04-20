//! Optimizer updates for LLM training (SGD, Adam, AdamW).

const std = @import("std");
const mod = @import("../mod.zig");
const trainable_model = @import("../trainable_model.zig");
const types = @import("types.zig");

/// Collect gradients into a flat array (same layout as collectWeights).
pub fn collectGradients(allocator: std.mem.Allocator, model: *trainable_model.TrainableModel) ![]f32 {
    const n = model.numParams();
    const grads = try allocator.alloc(f32, n);
    errdefer allocator.free(grads);

    var off: usize = 0;
    const w = &model.weights;

    @memcpy(grads[off..][0..w.d_token_embedding.len], w.d_token_embedding);
    off += w.d_token_embedding.len;

    for (w.layers) |layer| {
        inline for (.{
            layer.d_w_q,       layer.d_w_k,    layer.d_w_v,  layer.d_w_o,
            layer.d_attn_norm, layer.d_w_gate, layer.d_w_up, layer.d_w_down,
            layer.d_ffn_norm,
        }) |slice| {
            @memcpy(grads[off..][0..slice.len], slice);
            off += slice.len;
        }
    }

    @memcpy(grads[off..][0..w.d_final_norm.len], w.d_final_norm);
    off += w.d_final_norm.len;

    if (w.d_output_proj) |d_op| {
        @memcpy(grads[off..][0..d_op.len], d_op);
        off += d_op.len;
    }

    return grads;
}

/// Perform an optimizer step: compute LR, clip gradients, apply update, zero grads.
pub fn optimizerStep(
    allocator: std.mem.Allocator,
    model: *trainable_model.TrainableModel,
    config: types.LlmTrainingConfig,
    optimizer_state: *types.OptimizerState,
    stats: *types.TrainingStats,
    accum_loss: f32,
    accum_correct: u64,
    accum_tokens: u64,
    loss_history: *std.ArrayListUnmanaged(f32),
) !void {
    // Compute average loss
    const avg_loss = accum_loss / @as(f32, @floatFromInt(config.grad_accum_steps));
    const avg_accuracy = if (accum_tokens > 0)
        @as(f32, @floatFromInt(accum_correct)) / @as(f32, @floatFromInt(accum_tokens))
    else
        0;

    // Get current learning rate
    const lr = mod.calculateLearningRate(
        .{
            .learning_rate = config.learning_rate,
            .learning_rate_schedule = config.lr_schedule,
            .warmup_steps = config.warmup_steps,
            .decay_steps = config.decay_steps,
            .min_learning_rate = config.min_learning_rate,
        },
        stats.global_step,
        config.learning_rate,
    );

    // Apply gradient clipping
    const grad_norm = model.clipGradients(config.max_grad_norm);

    // Apply optimizer update based on type
    switch (config.optimizer) {
        .sgd => applySgdUpdate(allocator, model, lr),
        .adam, .adamw => applyAdamUpdate(
            allocator,
            model,
            optimizer_state,
            lr,
            config.optimizer == .adamw,
            config.weight_decay,
            stats.global_step,
        ),
    }

    // Zero gradients
    model.zeroGradients();

    // Update stats
    stats.global_step += 1;
    stats.loss = avg_loss;
    stats.accuracy = avg_accuracy;
    stats.perplexity = @exp(avg_loss);
    stats.learning_rate = lr;
    stats.grad_norm = grad_norm;

    // Record loss history
    try loss_history.append(allocator, avg_loss);
}

/// Apply SGD update: weight -= lr * gradient
fn applySgdUpdate(allocator: std.mem.Allocator, model: *trainable_model.TrainableModel, lr: f32) void {
    const weights = model.collectWeights() catch return;
    defer allocator.free(weights);
    const grads = collectGradients(allocator, model) catch return;
    defer allocator.free(grads);

    for (weights, grads) |*w, g| {
        w.* -= lr * g;
    }

    model.distributeWeights(weights) catch return;
}

/// Apply Adam/AdamW update.
fn applyAdamUpdate(
    allocator: std.mem.Allocator,
    model: *trainable_model.TrainableModel,
    optimizer_state: *types.OptimizerState,
    lr: f32,
    weight_decay_enabled: bool,
    wd: f32,
    global_step: u64,
) void {
    const beta1 = optimizer_state.beta1;
    const beta2 = optimizer_state.beta2;
    const eps = optimizer_state.epsilon;
    const t = @as(f32, @floatFromInt(global_step + 1));

    const bias_correction1 = 1.0 - std.math.pow(f32, beta1, t);
    const bias_correction2 = 1.0 - std.math.pow(f32, beta2, t);

    const m = optimizer_state.m orelse return;
    const v = optimizer_state.v orelse return;

    const weights = model.collectWeights() catch return;
    defer allocator.free(weights);
    const grads = collectGradients(allocator, model) catch return;
    defer allocator.free(grads);

    for (weights, grads, m, v) |*w, g, *mi, *vi| {
        // Update biased first moment estimate
        mi.* = beta1 * mi.* + (1.0 - beta1) * g;
        // Update biased second raw moment estimate
        vi.* = beta2 * vi.* + (1.0 - beta2) * g * g;
        // Bias-corrected estimates
        const m_hat = mi.* / bias_correction1;
        const v_hat = vi.* / bias_correction2;
        // AdamW: decoupled weight decay
        if (weight_decay_enabled) {
            w.* -= lr * wd * w.*;
        }
        // Parameter update
        w.* -= lr * m_hat / (@sqrt(v_hat) + eps);
    }

    model.distributeWeights(weights) catch return;
}

test {
    std.testing.refAllDecls(@This());
}
