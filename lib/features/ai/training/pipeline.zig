const std = @import("std");
const config = @import("config.zig");
const metrics_mod = @import("metrics.zig");
const interfaces = @import("../interfaces.zig");
const optimizers = @import("../optimizers/mod.zig");

pub const LossFunction = enum {
    mean_squared_error,
    mean_absolute_error,
    cross_entropy,
    binary_cross_entropy,
    categorical_cross_entropy,
    sparse_categorical_cross_entropy,
    huber,
    hinge,
    squared_hinge,
    cosine_similarity,
    kullback_leibler_divergence,
    focal_loss,
    dice_loss,
    contrastive_loss,
    triplet_loss,
};

/// Shared model operation interface that decouples the trainer from concrete models.
pub const ModelOps = struct {
    set_training: fn (*anyopaque, bool) void,
    get_output_size: fn (*anyopaque) usize,
    forward: fn (*anyopaque, []const f32, []f32) anyerror!void,
    apply_gradients: ?fn (*anyopaque, optimizers.OptimizerHandle) anyerror!void = null,
};

/// Handle pairing model specific context with the generic operations table.
pub const ModelHandle = struct {
    context: *anyopaque,
    ops: ModelOps,
};

pub const OptimizerHandle = optimizers.OptimizerHandle;

pub const InitOptions = struct {
    tensor_ops: ?interfaces.TensorOps = null,
};

pub const ModelTrainer = struct {
    model: ModelHandle,
    config: config.Config,
    allocator: std.mem.Allocator,
    loss_function: LossFunction,
    tensor_ops: interfaces.TensorOps,
    optimizer: optimizers.OptimizerHandle,

    // Training state
    current_epoch: usize = 0,
    current_step: usize = 0,
    best_val_loss: f32 = std.math.inf(f32),
    patience_counter: usize = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        model: ModelHandle,
        config_value: config.Config,
        loss_function: LossFunction,
        optimizer_handle: optimizers.OptimizerHandle,
        options: InitOptions,
    ) !*ModelTrainer {
        const trainer = try allocator.create(ModelTrainer);
        trainer.* = .{
            .model = model,
            .config = config_value,
            .allocator = allocator,
            .loss_function = loss_function,
            .tensor_ops = options.tensor_ops orelse interfaces.createBasicTensorOps(),
            .optimizer = optimizer_handle,
        };
        return trainer;
    }

    pub fn deinit(self: *ModelTrainer) void {
        self.optimizer.deinit();
        self.allocator.destroy(self);
    }

    pub fn train(
        self: *ModelTrainer,
        inputs: []const []const f32,
        targets: []const []const f32,
    ) !std.ArrayList(metrics_mod.Metrics) {
        var metrics = std.ArrayList(metrics_mod.Metrics){};

        if (inputs.len != targets.len) return error.InvalidDataSize;
        if (inputs.len == 0) return error.EmptyDataset;

        const validation_size = @as(usize, @intFromFloat(@as(f32, @floatFromInt(inputs.len)) * self.config.validation_split));
        const train_size = inputs.len - validation_size;

        const train_inputs = inputs[0..train_size];
        const train_targets = targets[0..train_size];
        const val_inputs = if (validation_size > 0) inputs[train_size..] else &[_][]const f32{};
        const val_targets = if (validation_size > 0) targets[train_size..] else &[_][]const f32{};

        for (0..self.config.epochs) |epoch| {
            self.current_epoch = epoch;
            const start_time = std.time.milliTimestamp();

            self.model.ops.set_training(self.model.context, true);
            const train_metrics = try self.trainEpoch(train_inputs, train_targets);

            var val_metrics: ?metrics_mod.Metrics = null;
            if (val_inputs.len > 0 and epoch % self.config.validate_frequency == 0) {
                self.model.ops.set_training(self.model.context, false);
                val_metrics = try self.validateEpoch(val_inputs, val_targets);
            }

            const end_time = std.time.milliTimestamp();
            var epoch_metrics = train_metrics;
            epoch_metrics.epoch = epoch;
            epoch_metrics.training_time_ms = @as(u64, @intCast(end_time - start_time));

            if (val_metrics) |vm| {
                epoch_metrics.val_loss = vm.loss;
                epoch_metrics.val_accuracy = vm.accuracy;
            }

            try metrics.append(self.allocator, epoch_metrics);

            if (self.shouldEarlyStop(epoch_metrics)) {
                std.debug.print("Early stopping at epoch {}\n", .{epoch});
                break;
            }

            if (epoch % self.config.log_frequency == 0) {
                self.logMetrics(epoch_metrics);
            }
        }

        return metrics;
    }

    fn trainEpoch(self: *ModelTrainer, inputs: []const []const f32, targets: []const []const f32) !metrics_mod.Metrics {
        var total_loss: f32 = 0.0;
        var total_accuracy: f32 = 0.0;
        const num_batches = (inputs.len + self.config.batch_size - 1) / self.config.batch_size;

        for (0..num_batches) |batch_idx| {
            const start_idx = batch_idx * self.config.batch_size;
            const end_idx = @min(start_idx + self.config.batch_size, inputs.len);

            const batch_inputs = inputs[start_idx..end_idx];
            const batch_targets = targets[start_idx..end_idx];

            const batch_metrics = try self.trainBatch(batch_inputs, batch_targets);
            total_loss += batch_metrics.loss;
            total_accuracy += batch_metrics.accuracy;

            self.current_step += 1;
        }

        return metrics_mod.Metrics{
            .loss = total_loss / @as(f32, @floatFromInt(num_batches)),
            .accuracy = total_accuracy / @as(f32, @floatFromInt(num_batches)),
            .epoch = self.current_epoch,
            .step = self.current_step,
            .training_time_ms = 0,
            .learning_rate = self.getCurrentLearningRate(),
        };
    }

    fn trainBatch(self: *ModelTrainer, inputs: []const []const f32, targets: []const []const f32) !metrics_mod.Metrics {
        const output_size = self.model.ops.get_output_size(self.model.context);
        const predictions = try self.allocator.alloc(f32, output_size * inputs.len);
        defer self.allocator.free(predictions);

        var batch_loss: f32 = 0.0;
        var batch_accuracy: f32 = 0.0;

        for (inputs, 0..) |input, i| {
            const pred_slice = predictions[i * output_size .. (i + 1) * output_size];
            try self.model.ops.forward(self.model.context, input, pred_slice);

            const sample_loss = self.computeLoss(pred_slice, targets[i]);
            batch_loss += sample_loss;

            const sample_accuracy = self.computeAccuracy(pred_slice, targets[i]);
            batch_accuracy += sample_accuracy;
        }

        try self.backwardPass(inputs, targets, predictions);
        try self.updateWeights();

        return metrics_mod.Metrics{
            .loss = batch_loss / @as(f32, @floatFromInt(inputs.len)),
            .accuracy = batch_accuracy / @as(f32, @floatFromInt(inputs.len)),
            .epoch = self.current_epoch,
            .step = self.current_step,
            .training_time_ms = 0,
            .learning_rate = self.getCurrentLearningRate(),
        };
    }

    fn validateEpoch(self: *ModelTrainer, inputs: []const []const f32, targets: []const []const f32) !metrics_mod.Metrics {
        const output_size = self.model.ops.get_output_size(self.model.context);
        const predictions = try self.allocator.alloc(f32, output_size * inputs.len);
        defer self.allocator.free(predictions);

        var total_loss: f32 = 0.0;
        var total_accuracy: f32 = 0.0;

        for (inputs, 0..) |input, i| {
            const pred_slice = predictions[i * output_size .. (i + 1) * output_size];
            try self.model.ops.forward(self.model.context, input, pred_slice);

            total_loss += self.computeLoss(pred_slice, targets[i]);
            total_accuracy += self.computeAccuracy(pred_slice, targets[i]);
        }

        return metrics_mod.Metrics{
            .loss = total_loss / @as(f32, @floatFromInt(inputs.len)),
            .accuracy = total_accuracy / @as(f32, @floatFromInt(inputs.len)),
            .epoch = self.current_epoch,
            .step = self.current_step,
            .training_time_ms = 0,
            .learning_rate = self.getCurrentLearningRate(),
        };
    }

    fn backwardPass(
        self: *ModelTrainer,
        _inputs: []const []const f32,
        _targets: []const []const f32,
        predictions: []const f32,
    ) !void {
        _ = _inputs;
        _ = _targets;
        if (predictions.len > 0) {
            // Light-touch operation to exercise tensor interface for downstream extensions.
            const buffer = try self.allocator.alloc(f32, predictions.len);
            defer self.allocator.free(buffer);
            @memcpy(buffer, predictions);
            try self.tensor_ops.scale(buffer, 1.0);
        }
    }

    fn updateWeights(self: *ModelTrainer) !void {
        if (self.model.ops.apply_gradients) |apply| {
            try apply(self.model.context, self.optimizer);
        }
    }

    fn getCurrentLearningRate(self: *const ModelTrainer) f32 {
        var lr = self.optimizer.ops.update_learning_rate(self.optimizer.context, self.current_step);
        const scheduler = self.config.optimizer.scheduler;

        if (scheduler.warmup_steps > 0 and self.current_step < scheduler.warmup_steps) {
            const progress = @as(f32, @floatFromInt(self.current_step + 1)) / @as(f32, @floatFromInt(scheduler.warmup_steps));
            lr *= progress;
        }

        switch (scheduler.kind) {
            .constant => {},
            .step_decay => {
                if (scheduler.decay_steps > 0) {
                    const exponent = @as(f32, @floatFromInt(self.current_step / scheduler.decay_steps));
                    lr *= std.math.pow(f32, scheduler.decay_rate, exponent);
                }
            },
            .exponential_decay => {
                lr *= std.math.pow(f32, scheduler.decay_rate, @as(f32, @floatFromInt(self.current_step)));
            },
            .cosine_annealing => {
                const progress = @as(f32, @floatFromInt(self.current_epoch)) / @as(f32, @floatFromInt(self.config.epochs));
                lr *= 0.5 * (1.0 + @cos(std.math.pi * progress));
            },
            else => {},
        }

        return lr;
    }

    fn shouldEarlyStop(self: *ModelTrainer, metrics_value: metrics_mod.Metrics) bool {
        if (!self.config.save_best_only) return false;

        if (metrics_value.val_loss) |val_loss| {
            if (val_loss + self.config.early_stopping_min_delta < self.best_val_loss) {
                self.best_val_loss = val_loss;
                self.patience_counter = 0;
            } else {
                self.patience_counter += 1;
            }
        }

        return self.patience_counter >= self.config.early_stopping_patience;
    }

    fn logMetrics(self: *const ModelTrainer, metrics_value: metrics_mod.Metrics) void {
        _ = self;
        const metrics = metrics_value;
        std.debug.print(
            "Epoch {} - loss: {d:.6}, acc: {d:.3}, lr: {d:.6}\n",
            .{ metrics.epoch, metrics.loss, metrics.accuracy, metrics.learning_rate },
        );
    }

    fn computeLoss(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        return LossFunction_compute(self.loss_function, predictions, targets);
    }

    fn computeAccuracy(self: *const ModelTrainer, predictions: []const f32, targets: []const f32) f32 {
        _ = self;
        if (predictions.len == 0 or targets.len == 0) return 0.0;

        if (predictions.len == 1) {
            const pred = predictions[0];
            const target = targets[0];
            const diff = @abs(pred - target);
            const denom = @max(@abs(target), 1e-5);
            return 1.0 - diff / denom;
        }

        var max_pred: f32 = predictions[0];
        var max_idx: usize = 0;
        for (predictions, 0..) |pred, i| {
            if (pred > max_pred) {
                max_pred = pred;
                max_idx = i;
            }
        }

        var max_target: f32 = targets[0];
        var target_idx: usize = 0;
        for (targets, 0..) |target, i| {
            if (target > max_target) {
                max_target = target;
                target_idx = i;
            }
        }

        return if (max_idx == target_idx) 1.0 else 0.0;
    }
};

fn LossFunction_compute(kind: LossFunction, predictions: []const f32, targets: []const f32) f32 {
    return switch (kind) {
        .mean_squared_error => meanSquaredError(predictions, targets),
        .mean_absolute_error => meanAbsoluteError(predictions, targets),
        .cross_entropy => crossEntropy(predictions, targets),
        .binary_cross_entropy => binaryCrossEntropy(predictions, targets),
        .categorical_cross_entropy => categoricalCrossEntropy(predictions, targets),
        .sparse_categorical_cross_entropy => crossEntropy(predictions, targets),
        .huber => huberLoss(predictions, targets),
        .hinge => hingeLoss(predictions, targets),
        .squared_hinge => hingeLoss(predictions, targets),
        .cosine_similarity => cosineSimilarityLoss(predictions, targets),
        .kullback_leibler_divergence => crossEntropy(predictions, targets),
        .focal_loss => focalLoss(predictions, targets),
        .dice_loss => diceLoss(predictions, targets),
        .contrastive_loss => contrastiveLoss(predictions, targets),
        .triplet_loss => tripletLoss(predictions, targets),
    };
}

fn meanSquaredError(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    var sum: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        const diff = pred - targets[i];
        sum += diff * diff;
    }
    return sum / @as(f32, @floatFromInt(predictions.len));
}

fn meanAbsoluteError(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    var sum: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        sum += @abs(pred - targets[i]);
    }
    return sum / @as(f32, @floatFromInt(predictions.len));
}

fn crossEntropy(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    var loss: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        const clipped_pred = @max(@min(pred, 1.0 - 1e-7), 1e-7);
        loss -= targets[i] * @log(clipped_pred);
    }
    return loss;
}

fn binaryCrossEntropy(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    var loss: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        const clipped_pred = @max(@min(pred, 1.0 - 1e-7), 1e-7);
        loss -= targets[i] * @log(clipped_pred) + (1.0 - targets[i]) * @log(1.0 - clipped_pred);
    }
    return loss / @as(f32, @floatFromInt(predictions.len));
}

fn categoricalCrossEntropy(predictions: []const f32, targets: []const f32) f32 {
    return crossEntropy(predictions, targets);
}

fn huberLoss(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    const delta: f32 = 1.0;
    var loss: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        const diff = @abs(pred - targets[i]);
        if (diff <= delta) {
            loss += 0.5 * diff * diff;
        } else {
            loss += delta * (diff - 0.5 * delta);
        }
    }
    return loss / @as(f32, @floatFromInt(predictions.len));
}

fn hingeLoss(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    var loss: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        loss += @max(0.0, 1.0 - targets[i] * pred);
    }
    return loss / @as(f32, @floatFromInt(predictions.len));
}

fn focalLoss(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    const alpha: f32 = 0.25;
    const gamma: f32 = 2.0;
    var loss: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        const clipped_pred = @max(@min(pred, 1.0 - 1e-7), 1e-7);
        const pt = if (targets[i] == 1.0) clipped_pred else 1.0 - clipped_pred;
        const alpha_t = if (targets[i] == 1.0) alpha else 1.0 - alpha;
        loss -= alpha_t * std.math.pow(f32, 1.0 - pt, gamma) * @log(pt);
    }
    return loss / @as(f32, @floatFromInt(predictions.len));
}

fn cosineSimilarityLoss(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    var dot: f32 = 0.0;
    var pred_norm: f32 = 0.0;
    var target_norm: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        dot += pred * targets[i];
        pred_norm += pred * pred;
        target_norm += targets[i] * targets[i];
    }
    const denom = @sqrt(pred_norm) * @sqrt(target_norm) + 1e-7;
    return 1.0 - dot / denom;
}

fn diceLoss(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    var intersection: f32 = 0.0;
    var sum: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        intersection += pred * targets[i];
        sum += pred + targets[i];
    }
    const smooth: f32 = 1.0;
    return 1.0 - (2.0 * intersection + smooth) / (sum + smooth);
}

fn contrastiveLoss(predictions: []const f32, targets: []const f32) f32 {
    if (predictions.len != targets.len) return std.math.inf(f32);
    var loss: f32 = 0.0;
    for (predictions, 0..) |pred, i| {
        const label = targets[i];
        const dist = pred;
        if (label == 1.0) {
            loss += dist * dist;
        } else {
            const margin = @max(0.0, 1.0 - dist);
            loss += margin * margin;
        }
    }
    return loss / (2.0 * @as(f32, @floatFromInt(predictions.len)));
}

fn tripletLoss(predictions: []const f32, targets: []const f32) f32 {
    _ = targets;
    if (predictions.len < 3) return 0.0;
    const anchor = predictions[0];
    const positive = predictions[1];
    const negative = predictions[2];
    const margin: f32 = 0.5;
    return @max(0.0, anchor - positive + margin - (anchor - negative));
}

pub fn LossFunction_computePublic(kind: LossFunction, predictions: []const f32, targets: []const f32) f32 {
    return LossFunction_compute(kind, predictions, targets);
}

pub const TrainingMetrics = metrics_mod.Metrics;

/// Convenience helper for callers that only need metric computation.
pub fn computeLoss(kind: LossFunction, predictions: []const f32, targets: []const f32) f32 {
    return LossFunction_compute(kind, predictions, targets);
}

const DummyModel = struct {
    output_size: usize = 1,
    forward_calls: usize = 0,
    apply_calls: usize = 0,

    fn toHandle(self: *DummyModel) ModelHandle {
        return ModelHandle{
            .context = self,
            .ops = ModelOps{
                .set_training = struct {
                    fn set(ctx: *anyopaque, training: bool) void {
                        _ = training;
                        const model: *DummyModel = @ptrCast(@alignCast(ctx));
                        _ = model;
                    }
                }.set,
                .get_output_size = struct {
                    fn get(ctx: *anyopaque) usize {
                        const model: *DummyModel = @ptrCast(@alignCast(ctx));
                        return model.output_size;
                    }
                }.get,
                .forward = struct {
                    fn forward(ctx: *anyopaque, _: []const f32, output: []f32) anyerror!void {
                        const model: *DummyModel = @ptrCast(@alignCast(ctx));
                        model.forward_calls += 1;
                        for (output) |*value| value.* = 0.5;
                    }
                }.forward,
                .apply_gradients = struct {
                    fn apply(ctx: *anyopaque, _: optimizers.OptimizerHandle) anyerror!void {
                        const model: *DummyModel = @ptrCast(@alignCast(ctx));
                        model.apply_calls += 1;
                    }
                }.apply,
            },
        };
    }
};

test "loss function compute dispatch" {
    const preds = [_]f32{ 0.9, 0.1 };
    const targets = [_]f32{ 1.0, 0.0 };
    const loss = computeLoss(.cross_entropy, preds[0..], targets[0..]);
    try std.testing.expect(loss > 0.0);
}

test "model trainer executes training loop with model handle" {
    var dummy = DummyModel{};
    var optimizer = try optimizers.createStatelessHandle(std.testing.allocator, .{});
    errdefer optimizer.deinit();

    const trainer = try ModelTrainer.init(
        std.testing.allocator,
        dummy.toHandle(),
        .{ .epochs = 1, .batch_size = 1, .validation_split = 0.0 },
        .mean_squared_error,
        optimizer,
        .{},
    );
    defer trainer.deinit();

    const inputs = [_][]const f32{&[_]f32{1.0}};
    const targets = [_][]const f32{&[_]f32{0.0}};

    var metrics = try trainer.train(&inputs, &targets);
    defer metrics.deinit(std.testing.allocator);

    try std.testing.expect(metrics.items.len >= 1);
    try std.testing.expect(dummy.forward_calls >= 1);
    try std.testing.expect(dummy.apply_calls >= 1);
}
