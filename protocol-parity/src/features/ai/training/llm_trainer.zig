//! LLM Trainer for LLaMA-style models.
//!
//! All weights, gradients, activations, and optimizer state are f32.
//! Mixed precision mode (when enabled) uses FP16 for the forward-pass
//! working copy only; master weights, gradients, and all optimizer
//! arithmetic remain in f32 for numerical stability.
//!
//! Provides a complete training loop for transformer language models:
//! - Forward pass with activation caching
//! - Backward pass with gradient computation
//! - Optimizer integration (SGD, Adam, AdamW)
//! - Gradient accumulation and clipping
//! - Mixed precision with dynamic loss scaling
//! - NaN/Inf gradient guards
//! - Checkpointing and resumption

const std = @import("std");
const time = @import("../../../foundation/mod.zig").time;
const trainable_model = @import("trainable_model.zig");
const loss_mod = @import("loss.zig");
const mod = @import("mod.zig");
const llm_checkpoint = @import("llm_checkpoint.zig");
const logging = @import("logging.zig");
const training_bridge = @import("../../gpu/training_bridge.zig");
const mixed_precision = @import("mixed_precision.zig");

// Sub-module imports
const types = @import("llm_trainer/types.zig");
const optimizer_mod = @import("llm_trainer/optimizer.zig");
const checkpoint_mod = @import("llm_trainer/checkpoint.zig");
const evaluation_mod = @import("llm_trainer/evaluation.zig");
const logging_ext = @import("llm_trainer/logging_ext.zig");

// Re-export types
pub const LlmTrainingConfig = types.LlmTrainingConfig;
pub const TrainingStats = types.TrainingStats;
pub const TrainingReport = types.TrainingReport;
pub const EvalResult = types.EvalResult;
pub const EarlyStoppingConfig = types.EarlyStoppingConfig;
pub const TrainerError = types.TrainerError;

/// LLM Trainer.
pub const LlamaTrainer = struct {
    allocator: std.mem.Allocator,
    config: LlmTrainingConfig,
    model: *trainable_model.TrainableModel,
    loss_fn: loss_mod.CrossEntropyLoss,
    accumulator: mod.GradientAccumulator,
    optimizer_state: types.OptimizerState,
    stats: TrainingStats,
    /// GPU training bridge (GPU-accelerated ops with CPU fallback)
    gpu_bridge: ?training_bridge.GpuTrainingBridge,
    /// Mixed precision loss scaler (active when config.mixed_precision is true)
    loss_scaler: ?mixed_precision.LossScaler,
    /// Number of steps skipped due to NaN/Inf gradients
    nan_skip_count: u64,
    /// Accumulated loss for logging
    accum_loss: f32,
    /// Loss history for early stopping
    loss_history: std.ArrayListUnmanaged(f32),
    /// Validation data (optional)
    val_data: ?[]const u32,
    /// Early stopping state
    early_stopping: types.EarlyStoppingState,
    /// Best model checkpoint weights
    best_weights: ?[]f32,
    /// Best validation accuracy
    best_val_accuracy: f32,
    /// Total checkpoints saved
    checkpoints_saved: u32,
    /// Metrics logger (optional)
    logger: ?logging.TrainingLogger,
    /// Timer for throughput logging
    log_timer: ?time.Timer,
    last_log_time_ns: u64,
    last_log_tokens: u64,
    accum_correct: u64,
    accum_tokens: u64,

    pub const StepMetrics = types.StepMetrics;

    pub fn init(
        allocator: std.mem.Allocator,
        model: *trainable_model.TrainableModel,
        config: LlmTrainingConfig,
    ) !LlamaTrainer {
        try config.validate();

        var loss_fn = loss_mod.CrossEntropyLoss.init(allocator, model.config.vocab_size);
        loss_fn.setLabelSmoothing(config.label_smoothing);

        const num_params = model.numParams();
        const accumulator = try mod.GradientAccumulator.init(allocator, num_params);

        // Initialize optimizer state (Adam moments)
        const m = try allocator.alloc(f32, num_params);
        @memset(m, 0);
        const v = try allocator.alloc(f32, num_params);
        @memset(v, 0);

        var logger: ?logging.TrainingLogger = null;
        if (config.enable_tensorboard or config.enable_wandb) {
            logger = try logging.TrainingLogger.init(allocator, .{
                .log_dir = config.log_dir.?,
                .enable_tensorboard = config.enable_tensorboard,
                .enable_wandb = config.enable_wandb,
                .enable_metrics_stream = config.enable_metrics_stream,
                .metrics_path = null,
                .wandb_project = config.wandb_project,
                .wandb_run_name = config.wandb_run_name,
                .wandb_entity = config.wandb_entity,
            });
        }

        // Initialize GPU bridge if requested
        var gpu_bridge_inst: ?training_bridge.GpuTrainingBridge = null;
        if (config.use_gpu) {
            var bridge = training_bridge.GpuTrainingBridge.init(allocator);
            if (bridge.gpu_available) {
                gpu_bridge_inst = bridge;
                std.log.info("GPU training bridge initialized (backend: {s})", .{bridge.stats.backend_name});
            } else {
                bridge.deinit();
                std.log.info("GPU not available, using CPU-only training", .{});
            }
        }

        var log_timer = time.Timer.start() catch null;
        const initial_time = if (log_timer) |*t| t.read() else 0;

        // Initialize loss scaler for mixed precision
        var loss_scaler: ?mixed_precision.LossScaler = null;
        if (config.mixed_precision) {
            loss_scaler = mixed_precision.LossScaler.init(allocator, .{
                .enabled = true,
                .initial_scale = 65536.0,
            });
            std.log.info("Mixed precision training enabled (FP16 forward, FP32 gradients)", .{});
        }

        return .{
            .allocator = allocator,
            .config = config,
            .model = model,
            .loss_fn = loss_fn,
            .accumulator = accumulator,
            .optimizer_state = .{
                .m = m,
                .v = v,
            },
            .stats = .{},
            .gpu_bridge = gpu_bridge_inst,
            .loss_scaler = loss_scaler,
            .nan_skip_count = 0,
            .accum_loss = 0,
            .loss_history = std.ArrayListUnmanaged(f32).empty,
            .val_data = null,
            .early_stopping = .{},
            .best_weights = null,
            .best_val_accuracy = 0,
            .checkpoints_saved = 0,
            .logger = logger,
            .log_timer = log_timer,
            .last_log_time_ns = initial_time,
            .last_log_tokens = 0,
            .accum_correct = 0,
            .accum_tokens = 0,
        };
    }

    /// Set validation data.
    pub fn setValidationData(self: *LlamaTrainer, data: []const u32) void {
        self.val_data = data;
    }

    pub fn deinit(self: *LlamaTrainer) void {
        if (self.best_weights) |bw| self.allocator.free(bw);
        if (self.logger) |*logger| logger.deinit();
        if (self.gpu_bridge) |*bridge| bridge.deinit();
        if (self.loss_scaler) |*scaler| scaler.deinit();
        self.loss_history.deinit(self.allocator);
        if (self.optimizer_state.v) |v| self.allocator.free(v);
        if (self.optimizer_state.m) |m| self.allocator.free(m);
        self.accumulator.deinit();
        self.loss_fn.deinit();
        self.* = undefined;
    }

    /// Perform a single training step.
    /// Returns the loss for this batch.
    pub fn trainStep(
        self: *LlamaTrainer,
        input_ids: []const u32,
        labels: []const u32,
    ) !f32 {
        const result = try self.trainStepWithMetrics(input_ids, labels);
        return result.loss;
    }

    /// Perform a single training step with metrics.
    pub fn trainStepWithMetrics(
        self: *LlamaTrainer,
        input_ids: []const u32,
        labels: []const u32,
    ) !StepMetrics {
        const seq_len: u32 = @intCast(input_ids.len);

        // Ensure model is prepared for training
        if (self.model.activations == null) {
            try self.model.prepareForTraining(seq_len);
        }

        const vocab_size = self.model.config.vocab_size;
        const token_count = labels.len;
        const logits = try self.allocator.alloc(f32, token_count * vocab_size);
        defer self.allocator.free(logits);

        // Forward pass through the model (GPU-accelerated when available)
        if (self.gpu_bridge) |*bridge| {
            try self.model.forwardGpu(input_ids, logits, bridge);
        } else {
            try self.model.forward(input_ids, logits);
        }

        // Compute loss using CrossEntropy
        var loss = try self.loss_fn.forward(logits, labels);

        // Scale loss for mixed precision (prevents gradient underflow)
        if (self.loss_scaler) |*scaler| {
            loss = scaler.scaleLoss(loss);
        }

        // Backward pass (compute gradients through loss function)
        const d_logits = try self.allocator.alloc(f32, logits.len);
        defer self.allocator.free(d_logits);
        self.loss_fn.backward(d_logits);

        // Backward pass through the model (GPU-accelerated when available)
        if (self.gpu_bridge) |*bridge| {
            try self.model.backwardGpu(d_logits, input_ids, bridge);
        } else {
            try self.model.backward(d_logits, input_ids);
        }

        // NaN/Inf gradient guard: skip step if gradients are corrupted
        if (self.model.hasNonFiniteGradients()) {
            self.nan_skip_count += 1;
            if (self.config.mixed_precision) {
                if (self.loss_scaler) |*scaler| {
                    _ = scaler.unscaleGradients(self.model.weights.d_token_embedding);
                    scaler.update(false);
                }
            } else {
                std.log.warn("NaN/Inf in gradients at step {d} (skip #{d})", .{
                    self.stats.global_step,
                    self.nan_skip_count,
                });
            }
            self.model.zeroGradients();
            return .{ .loss = loss, .accuracy = 0 };
        }

        // Unscale gradients for mixed precision
        if (self.loss_scaler) |*scaler| {
            const grads_valid = scaler.unscaleGradients(self.model.weights.d_token_embedding);
            scaler.update(grads_valid);
            if (!grads_valid) {
                self.model.zeroGradients();
                return .{ .loss = loss, .accuracy = 0 };
            }
        }

        // Compute accuracy
        var correct: u64 = 0;
        var tokens: u64 = 0;
        var i: usize = 0;
        while (i < labels.len) : (i += 1) {
            const logit_start = i * vocab_size;
            if (logit_start + vocab_size > logits.len) break;
            var max_idx: u32 = 0;
            var max_val: f32 = logits[logit_start];
            for (1..vocab_size) |j| {
                const value = logits[logit_start + j];
                if (value > max_val) {
                    max_val = value;
                    max_idx = @intCast(j);
                }
            }
            if (max_idx == labels[i]) {
                correct += 1;
            }
            tokens += 1;
        }
        const accuracy = if (tokens > 0)
            @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(tokens))
        else
            0;

        // Accumulate gradients
        self.accum_loss += loss;
        self.accum_correct += correct;
        self.accum_tokens += tokens;
        self.stats.micro_step += 1;

        // Check if we should do an optimizer step
        if (self.stats.micro_step >= self.config.grad_accum_steps) {
            try optimizer_mod.optimizerStep(
                self.allocator,
                self.model,
                self.config,
                &self.optimizer_state,
                &self.stats,
                self.accum_loss,
                self.accum_correct,
                self.accum_tokens,
                &self.loss_history,
            );
            try logging_ext.maybeLog(
                self.config,
                &self.stats,
                &self.logger,
                &self.gpu_bridge,
                &self.log_timer,
                &self.last_log_time_ns,
                &self.last_log_tokens,
            );
            self.stats.micro_step = 0;
            self.accum_loss = 0;
            self.accum_correct = 0;
            self.accum_tokens = 0;
        }

        // Update stats
        self.stats.tokens_processed += tokens;

        return .{
            .loss = loss,
            .accuracy = accuracy,
        };
    }

    /// Train for one epoch.
    pub fn trainEpoch(
        self: *LlamaTrainer,
        data: []const u32,
        num_samples: usize,
    ) !f32 {
        const tokens_per_sample = self.config.max_seq_len;
        const batch_tokens = self.config.batch_size * tokens_per_sample;

        var epoch_loss: f32 = 0;
        var num_batches: u32 = 0;

        var offset: usize = 0;
        while (offset + batch_tokens <= data.len) {
            const batch_data = data[offset .. offset + batch_tokens];

            const input_ids = batch_data[0 .. batch_tokens - 1];
            const labels = batch_data[1..batch_tokens];

            const metrics = try self.trainStepWithMetrics(input_ids, labels);
            epoch_loss += metrics.loss;
            num_batches += 1;

            offset += batch_tokens;

            // Check for checkpoint
            if (self.config.checkpoint_interval > 0 and
                self.stats.global_step % self.config.checkpoint_interval == 0)
            {
                try self.saveCheckpoint();
            }
        }

        _ = num_samples;
        self.stats.epoch += 1;
        return if (num_batches > 0) epoch_loss / @as(f32, @floatFromInt(num_batches)) else 0;
    }

    /// Save checkpoint.
    pub fn saveCheckpoint(self: *LlamaTrainer) !void {
        try checkpoint_mod.saveCheckpoint(
            self.allocator,
            self.model,
            self.config,
            &self.optimizer_state,
            &self.stats,
            &self.checkpoints_saved,
        );
    }

    /// Load checkpoint.
    pub fn loadCheckpoint(self: *LlamaTrainer, path: []const u8) !void {
        try checkpoint_mod.loadCheckpoint(
            self.allocator,
            self.model,
            &self.optimizer_state,
            &self.stats,
            path,
        );
    }

    /// Evaluate model on validation data.
    pub fn evaluate(self: *LlamaTrainer, data: []const u32) !EvalResult {
        return evaluation_mod.evaluate(
            self.allocator,
            self.model,
            &self.loss_fn,
            &self.gpu_bridge,
            self.config,
            data,
        );
    }

    /// Run validation if interval reached and data available.
    pub fn maybeValidate(self: *LlamaTrainer, early_stop_config: EarlyStoppingConfig) !bool {
        return evaluation_mod.maybeValidate(
            self.allocator,
            self.model,
            &self.loss_fn,
            &self.gpu_bridge,
            self.config,
            &self.stats,
            self.val_data,
            early_stop_config,
            &self.early_stopping,
            &self.best_val_accuracy,
            &self.best_weights,
            &self.logger,
        );
    }

    /// Check if training was early stopped.
    pub fn wasEarlyStopped(self: *const LlamaTrainer) bool {
        return evaluation_mod.wasEarlyStopped(&self.early_stopping);
    }

    /// Get current training stats.
    pub fn getStats(self: *const LlamaTrainer) TrainingStats {
        return self.stats;
    }

    /// Get GPU training statistics.
    pub fn getGpuStats(self: *const LlamaTrainer) training_bridge.GpuTrainingStats {
        if (self.gpu_bridge) |*bridge| return bridge.getStats();
        return .{};
    }

    /// Get training report.
    pub fn getReport(self: *const LlamaTrainer) TrainingReport {
        return .{
            .final_loss = self.stats.loss,
            .final_accuracy = self.stats.accuracy,
            .best_val_loss = self.early_stopping.best_metric,
            .best_val_accuracy = self.best_val_accuracy,
            .final_perplexity = self.stats.perplexity,
            .total_steps = self.stats.global_step,
            .total_tokens = self.stats.tokens_processed,
            .total_time_ns = self.stats.time_ns,
            .avg_throughput = self.stats.throughput,
            .checkpoints_saved = self.checkpoints_saved,
            .early_stopped = self.early_stopping.stopped,
        };
    }

    /// Finalize logging (write summary metrics).
    pub fn finalizeLogging(self: *LlamaTrainer) !void {
        try logging_ext.finalizeLogging(
            &self.logger,
            &self.stats,
            &self.early_stopping,
            self.best_val_accuracy,
        );
    }

    /// Train for one epoch with validation and early stopping.
    pub fn trainEpochWithValidation(
        self: *LlamaTrainer,
        data: []const u32,
        early_stop_config: EarlyStoppingConfig,
    ) !struct { loss: f32, early_stopped: bool } {
        const tokens_per_sample = self.config.max_seq_len;
        const batch_tokens = self.config.batch_size * tokens_per_sample;

        var epoch_loss: f32 = 0;
        var num_batches: u32 = 0;

        var offset: usize = 0;
        while (offset + batch_tokens <= data.len) {
            const batch_data = data[offset .. offset + batch_tokens];

            const input_ids = batch_data[0 .. batch_tokens - 1];
            const labels = batch_data[1..batch_tokens];

            const metrics = try self.trainStepWithMetrics(input_ids, labels);
            epoch_loss += metrics.loss;
            num_batches += 1;
            offset += batch_tokens;

            // Check for checkpoint
            if (self.config.checkpoint_interval > 0 and
                self.stats.global_step % self.config.checkpoint_interval == 0)
            {
                try self.saveCheckpoint();
            }

            // Maybe validate and check early stopping
            const should_continue = try self.maybeValidate(early_stop_config);
            if (!should_continue) {
                return .{
                    .loss = if (num_batches > 0) epoch_loss / @as(f32, @floatFromInt(num_batches)) else 0,
                    .early_stopped = true,
                };
            }
        }

        self.stats.epoch += 1;

        return .{
            .loss = if (num_batches > 0) epoch_loss / @as(f32, @floatFromInt(num_batches)) else 0,
            .early_stopped = false,
        };
    }
};

/// Create and run a training session.
pub fn trainLlm(
    allocator: std.mem.Allocator,
    model: *trainable_model.TrainableModel,
    config: LlmTrainingConfig,
    train_data: []const u32,
) !TrainingReport {
    return trainLlmWithValidation(allocator, model, config, train_data, null, .{});
}

/// Create and run a training session with validation.
pub fn trainLlmWithValidation(
    allocator: std.mem.Allocator,
    model: *trainable_model.TrainableModel,
    config: LlmTrainingConfig,
    train_data: []const u32,
    val_data: ?[]const u32,
    early_stop_config: EarlyStoppingConfig,
) !TrainingReport {
    var trainer = try LlamaTrainer.init(allocator, model, config);
    defer trainer.deinit();

    if (val_data) |vd| {
        trainer.setValidationData(vd);
    }

    var timer = time.Timer.start() catch return error.TimerFailed;

    for (0..config.epochs) |_| {
        const result = try trainer.trainEpochWithValidation(train_data, early_stop_config);
        std.log.info("Epoch {d}: loss={d:.4} ppl={d:.2} acc={d:.2}%", .{
            trainer.stats.epoch,
            result.loss,
            @exp(result.loss),
            trainer.stats.accuracy * 100,
        });

        if (result.early_stopped) {
            std.log.info("Training stopped early at epoch {d}", .{trainer.stats.epoch});
            break;
        }
    }

    trainer.stats.time_ns = timer.read();
    trainer.stats.throughput = @as(f32, @floatFromInt(trainer.stats.tokens_processed)) /
        (@as(f32, @floatFromInt(trainer.stats.time_ns)) / 1e9);

    if (config.export_gguf_path) |path| {
        try model.exportToGguf(allocator, path, .{
            .name = config.export_name,
        });
    }

    try trainer.finalizeLogging();

    return trainer.getReport();
}

test "llm training config validation" {
    const valid_config = LlmTrainingConfig{
        .epochs = 1,
        .batch_size = 2,
        .max_seq_len = 64,
    };
    try valid_config.validate();

    const invalid_config = LlmTrainingConfig{
        .epochs = 0, // Invalid
        .batch_size = 2,
        .max_seq_len = 64,
    };
    try std.testing.expectError(error.InvalidConfiguration, invalid_config.validate());
}

test "llama trainer init" {
    const allocator = std.testing.allocator;

    const model_config = trainable_model.TrainableModelConfig{
        .hidden_dim = 32,
        .num_layers = 2,
        .num_heads = 4,
        .num_kv_heads = 4,
        .intermediate_dim = 64,
        .vocab_size = 100,
        .max_seq_len = 64,
    };

    var model = try trainable_model.TrainableModel.init(allocator, model_config);
    defer model.deinit();

    const train_config = LlmTrainingConfig{
        .epochs = 1,
        .batch_size = 2,
        .max_seq_len = 32,
    };

    var trainer = try LlamaTrainer.init(allocator, &model, train_config);
    defer trainer.deinit();

    try std.testing.expectEqual(@as(u32, 0), trainer.stats.epoch);
}

test {
    // Ensure sub-modules are analyzed
    _ = types;
    _ = optimizer_mod;
    _ = checkpoint_mod;
    _ = evaluation_mod;
    _ = logging_ext;
    std.testing.refAllDecls(@This());
}
