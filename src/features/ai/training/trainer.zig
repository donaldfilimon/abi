//! Training Loop and Utilities
//!
//! Core training loop, learning rate scheduling, gradient clipping,
//! and model persistence.

const std = @import("std");
const build_options = @import("build_options");
const time_utils = @import("../../../foundation/mod.zig").utils;
const time = @import("../../../foundation/mod.zig").time;
const database = if (build_options.feat_database) @import("../../database/mod.zig") else @import("../../database/stub.zig");

const mod = @import("mod.zig");
const optimizer_mod = @import("optimizer.zig");
const checkpoint = @import("checkpoint.zig");
const gradient = @import("gradient.zig");

pub const TrainingConfig = mod.TrainingConfig;
pub const TrainingReport = mod.TrainingReport;
pub const ModelState = mod.ModelState;
pub const TrainingError = mod.TrainingError;
pub const TrainError = mod.TrainError;
pub const CheckpointStore = checkpoint.CheckpointStore;
pub const Optimizer = optimizer_mod.Optimizer;

pub const TrainingResult = struct {
    allocator: std.mem.Allocator,
    report: TrainingReport,
    model: ModelState,
    optimizer: Optimizer,
    checkpoints: CheckpointStore,
    loss_history: []f32,
    accuracy_history: []f32,

    pub fn deinit(self: *TrainingResult) void {
        self.allocator.free(self.accuracy_history);
        self.allocator.free(self.loss_history);
        self.checkpoints.deinit();
        self.optimizer.deinit(self.allocator);
        self.model.deinit();
        self.* = undefined;
    }
};

pub fn calculateLearningRate(config: TrainingConfig, step_val: u64, base_lr: f32) f32 {
    return switch (config.learning_rate_schedule) {
        .constant => base_lr,
        .linear => {
            const progress = @min(1.0, @as(f32, @floatFromInt(step_val)) / @as(f32, @floatFromInt(config.decay_steps)));
            return base_lr * (1.0 - progress) + config.min_learning_rate * progress;
        },
        .cosine => base_lr * 0.5 * (1 + @cos(@as(f32, @floatFromInt(step_val % config.decay_steps)) * 2 * std.math.pi / @as(f32, @floatFromInt(config.decay_steps)))),
        .warmup_cosine => {
            if (step_val < config.warmup_steps) {
                return base_lr * @as(f32, @floatFromInt(step_val)) / @as(f32, @floatFromInt(config.warmup_steps));
            }
            const adjusted_step = step_val - config.warmup_steps;
            const adjusted_decay = config.decay_steps - config.warmup_steps;
            const progress = @min(1.0, @as(f32, @floatFromInt(adjusted_step)) / @as(f32, @floatFromInt(adjusted_decay)));
            return config.min_learning_rate + (base_lr - config.min_learning_rate) * 0.5 * (1 + @cos(progress * std.math.pi));
        },
        .step => {
            const decay = @as(f32, @floatFromInt(step_val / config.decay_steps));
            return base_lr * std.math.pow(f32, 0.1, decay);
        },
        .polynomial => {
            const progress = @min(1.0, @as(f32, @floatFromInt(step_val)) / @as(f32, @floatFromInt(config.decay_steps)));
            return base_lr * std.math.pow(f32, 1 - progress, 0.9);
        },
        .cosine_warm_restarts => {
            const t_0 = @as(f32, @floatFromInt(config.decay_steps));
            const t_mult: f32 = 2.0;
            const min_lr = config.min_learning_rate;

            var cycle_start: f32 = 0;
            var cycle_length = t_0;
            const step_f = @as(f32, @floatFromInt(step_val));

            while (cycle_start + cycle_length <= step_f) {
                cycle_start += cycle_length;
                cycle_length *= t_mult;
            }

            const t_cur = step_f - cycle_start;
            const t_i = cycle_length;
            const progress = t_cur / t_i;

            return min_lr + (base_lr - min_lr) * 0.5 * (1 + @cos(progress * std.math.pi));
        },
    };
}

pub fn clipGradients(gradients: []f32, max_norm: f32) f32 {
    var norm: f32 = 0;
    for (gradients) |g| {
        norm += g * g;
    }
    norm = @sqrt(norm);

    if (norm > max_norm and norm > 0) {
        const scale = max_norm / norm;
        for (gradients) |*g| {
            g.* *= scale;
        }
    }

    return norm;
}

pub fn saveModelToWdbx(allocator: std.mem.Allocator, model: *const ModelState, path: []const u8) !void {
    var handle = try database.semantic_store.createDatabase(allocator, "model_checkpoint");
    defer database.semantic_store.closeDatabase(&handle);

    // Store weights as vector ID 0
    try database.semantic_store.insertVector(&handle, 0, model.weights, model.name);

    try database.semantic_store.backup(&handle, path);
}

pub fn train(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!void {
    var result = try trainWithResult(allocator, config);
    defer result.deinit();
}

pub fn trainAndReport(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!TrainingReport {
    var result = try trainWithResult(allocator, config);
    defer result.deinit();
    return result.report;
}

pub fn trainWithResult(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!TrainingResult {
    try config.validate();

    var model = try ModelState.init(allocator, config.model_size, "model");
    errdefer model.deinit();

    var opt = try Optimizer.init(allocator, &model, config);
    errdefer opt.deinit(allocator);

    var accumulator = try gradient.GradientAccumulator.init(allocator, model.gradients.len);
    defer accumulator.deinit();

    var checkpoints = CheckpointStore.init(allocator, config.max_checkpoints);
    errdefer checkpoints.deinit();

    const batches_per_epoch: u32 = @intCast((config.sample_count + config.batch_size - 1) / config.batch_size);
    const gradient_buffer = try allocator.alloc(f32, model.gradients.len);
    defer allocator.free(gradient_buffer);

    var loss_history = try allocator.alloc(f32, config.epochs);
    errdefer allocator.free(loss_history);
    var accuracy_history = try allocator.alloc(f32, config.epochs);
    errdefer allocator.free(accuracy_history);

    var best_loss: f32 = 1e10;
    var patience_counter: u32 = 0;
    var early_stopped: bool = false;
    var last_epoch: usize = 0;

    var training_timer = time.Timer.start() catch return error.InvalidConfiguration;

    for (0..config.epochs) |epoch| {
        last_epoch = epoch;
        var epoch_loss: f32 = 0;
        var epoch_accuracy: f32 = 0;
        var batch_count: u32 = 0;

        var batch: u32 = 0;
        while (batch < batches_per_epoch) : (batch += 1) {
            for (gradient_buffer) |*v| {
                v.* = config.learning_rate;
            }
            try accumulator.add(gradient_buffer);

            const is_last_batch = batch + 1 == batches_per_epoch;
            if (accumulator.count >= config.gradient_accumulation_steps or is_last_batch) {
                const avg = try accumulator.average(allocator);
                defer allocator.free(avg);
                for (model.gradients, avg) |*g, a| {
                    g.* = a;
                }
                accumulator.reset();

                const norm = clipGradients(model.gradients, config.gradient_clip_norm);
                _ = norm;

                const current_lr = calculateLearningRate(config, model.step + 1, config.learning_rate);
                opt.setLearningRate(current_lr);
                opt.step(&model, current_lr, model.step + 1);
                model.step += 1;

                const step_loss = calculateLoss(model.weights, gradient_buffer);
                epoch_loss += step_loss;

                const step_acc = calculateAccuracy(model.weights, gradient_buffer);
                epoch_accuracy += step_acc;

                batch_count += 1;

                if (config.checkpoint_interval > 0 and
                    model.step % config.checkpoint_interval == 0)
                {
                    try checkpoints.add(model.step, model.weights);
                    if (config.checkpoint_path) |base_path| {
                        var path_buf: [256]u8 = undefined;
                        const ckpt_path = std.fmt.bufPrint(
                            &path_buf,
                            "{s}/step_{d}.ckpt",
                            .{ base_path, model.step },
                        ) catch continue;
                        checkpoint.saveCheckpoint(allocator, ckpt_path, .{
                            .step = model.step,
                            .timestamp = @as(u64, @intCast(time_utils.unixSeconds())),
                            .weights = model.weights,
                        }) catch |err| {
                            std.log.warn("failed to save checkpoint: {t}", .{err});
                        };
                    }
                }
            }
        }

        if (batch_count > 0) {
            epoch_loss /= @as(f32, @floatFromInt(batch_count));
            epoch_accuracy /= @as(f32, @floatFromInt(batch_count));
        }

        loss_history[epoch] = epoch_loss;
        accuracy_history[epoch] = epoch_accuracy;

        if (epoch_loss < best_loss - config.early_stopping_threshold) {
            best_loss = epoch_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if (patience_counter >= config.early_stopping_patience) {
                early_stopped = true;
                break;
            }
        }
    }

    const elapsed_ns = training_timer.read();
    const total_time_ms = elapsed_ns / std.time.ns_per_ms;

    const final_lr = calculateLearningRate(config, model.step, config.learning_rate);

    return .{
        .allocator = allocator,
        .report = .{
            .epochs = @as(u32, @intCast(last_epoch + 1)),
            .batches = @as(u32, @intCast(batches_per_epoch)),
            .final_loss = loss_history[last_epoch],
            .final_accuracy = accuracy_history[last_epoch],
            .best_loss = best_loss,
            .learning_rate = final_lr,
            .gradient_updates = model.step,
            .checkpoints_saved = @as(u32, @intCast(checkpoints.count())),
            .early_stopped = early_stopped,
            .total_time_ms = total_time_ms,
        },
        .model = model,
        .optimizer = opt,
        .checkpoints = checkpoints,
        .loss_history = loss_history,
        .accuracy_history = accuracy_history,
    };
}

pub fn initializeXavierUniform(weights: []f32, size: usize) void {
    const limit = @sqrt(2.0 / @as(f32, @floatFromInt(size)));
    var rng = std.Random.DefaultPrng.init(12345 + size);
    for (weights) |*val| {
        val.* = rng.random().floatNorm(f32) * limit;
    }
}

fn calculateLoss(weights: []f32, gradients: []f32) f32 {
    var total_loss: f32 = 0;
    for (weights, gradients) |w, g| {
        total_loss += w * w * 0.001 + g * g * 0.5;
    }
    return total_loss / @as(f32, @floatFromInt(weights.len));
}

fn calculateAccuracy(weights: []f32, gradients: []f32) f32 {
    var correct: usize = 0;
    for (weights, gradients) |w, g| {
        const prediction = if (w * g > 0) @as(u32, 1) else @as(u32, 0);
        if (prediction == 1) correct += 1;
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(weights.len));
}
