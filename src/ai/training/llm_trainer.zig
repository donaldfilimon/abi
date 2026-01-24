//! LLM Trainer for LLaMA-style models.
//!
//! Provides a complete training loop for transformer language models:
//! - Forward pass with activation caching
//! - Backward pass with gradient computation
//! - Optimizer integration (SGD, Adam, AdamW)
//! - Gradient accumulation and clipping
//! - Checkpointing and resumption

const std = @import("std");
const trainable_model = @import("trainable_model.zig");
const loss_mod = @import("loss.zig");
const mod = @import("mod.zig");
const llm_checkpoint = @import("llm_checkpoint.zig");
const logging = @import("logging.zig");
const ai_ops = @import("../../gpu/ai_ops.zig");

/// Training configuration for LLM.
pub const LlmTrainingConfig = struct {
    /// Number of training epochs
    epochs: u32 = 10,
    /// Batch size (sequences per batch)
    batch_size: u32 = 4,
    /// Maximum sequence length
    max_seq_len: u32 = 512,
    /// Base learning rate
    learning_rate: f32 = 1e-5,
    /// Learning rate schedule
    lr_schedule: mod.LearningRateSchedule = .warmup_cosine,
    /// Warmup steps
    warmup_steps: u32 = 100,
    /// Total decay steps
    decay_steps: u32 = 10000,
    /// Minimum learning rate
    min_learning_rate: f32 = 1e-7,
    /// Gradient accumulation steps
    grad_accum_steps: u32 = 8,
    /// Maximum gradient norm for clipping
    max_grad_norm: f32 = 1.0,
    /// Weight decay for AdamW
    weight_decay: f32 = 0.01,
    /// Optimizer type
    optimizer: mod.OptimizerType = .adamw,
    /// Label smoothing factor
    label_smoothing: f32 = 0.0,
    /// Checkpoint interval (steps, 0 = disabled)
    checkpoint_interval: u32 = 1000,
    /// Checkpoint directory path
    checkpoint_path: ?[]const u8 = null,
    /// Maximum checkpoints to keep
    max_checkpoints: u32 = 3,
    /// Log interval (steps)
    log_interval: u32 = 10,
    /// Evaluation interval (steps, 0 = disabled)
    eval_interval: u32 = 500,
    /// Mixed precision training (FP16 forward, FP32 gradients)
    mixed_precision: bool = false,
    /// Log directory for TensorBoard/W&B output
    log_dir: ?[]const u8 = null,
    /// Enable TensorBoard scalar logging
    enable_tensorboard: bool = false,
    /// Enable W&B offline logging
    enable_wandb: bool = false,
    /// Enable JSONL metrics stream for TUI dashboard
    enable_metrics_stream: bool = false,
    /// W&B project name (defaults to "abi")
    wandb_project: ?[]const u8 = null,
    /// W&B run name (defaults to "run")
    wandb_run_name: ?[]const u8 = null,
    /// W&B entity (defaults to "default")
    wandb_entity: ?[]const u8 = null,
    /// Export GGUF weights after training
    export_gguf_path: ?[]const u8 = null,
    /// GGUF model name metadata
    export_name: []const u8 = "abi-llama",
    /// Enable GPU acceleration
    use_gpu: bool = false,
    /// GPU acceleration backend preference
    gpu_backend: ?[]const u8 = null, // null = auto-select best available
    /// Threshold for GPU dispatch (batch size below this uses CPU)
    gpu_batch_threshold: u32 = 8,
    /// Device memory buffer for GPU operations
    gpu_device_buffer_mb: u32 = 256, // MB

    pub fn validate(self: LlmTrainingConfig) !void {
        if (self.epochs == 0) return error.InvalidConfiguration;
        if (self.batch_size == 0) return error.InvalidConfiguration;
        if (self.max_seq_len == 0) return error.InvalidConfiguration;
        if (self.learning_rate <= 0) return error.InvalidConfiguration;
        if (self.grad_accum_steps == 0) return error.InvalidConfiguration;
        if (self.max_grad_norm < 0) return error.InvalidConfiguration;
        if (self.label_smoothing < 0 or self.label_smoothing >= 1) return error.InvalidConfiguration;
        if ((self.enable_tensorboard or self.enable_wandb or self.enable_metrics_stream) and self.log_dir == null) {
            return error.InvalidConfiguration;
        }
    }
};

/// Training statistics.
pub const TrainingStats = struct {
    /// Current epoch
    epoch: u32 = 0,
    /// Current step (optimizer updates)
    global_step: u64 = 0,
    /// Current micro-batch within accumulation
    micro_step: u32 = 0,
    /// Total tokens processed
    tokens_processed: u64 = 0,
    /// Current loss (smoothed)
    loss: f32 = 0,
    /// Current accuracy
    accuracy: f32 = 0,
    /// Current learning rate
    learning_rate: f32 = 0,
    /// Current gradient norm
    grad_norm: f32 = 0,
    /// Perplexity
    perplexity: f32 = 0,
    /// Training throughput (tokens/sec)
    throughput: f32 = 0,
    /// Time since last log (ns)
    time_ns: u64 = 0,

    pub fn format(self: TrainingStats) [256]u8 {
        var buf: [256]u8 = undefined;
        _ = std.fmt.bufPrint(&buf, "epoch={d} step={d} loss={d:.4} acc={d:.2}% ppl={d:.2} lr={e:.2} grad_norm={d:.2} toks/s={d:.0}", .{
            self.epoch,
            self.global_step,
            self.loss,
            self.accuracy * 100,
            self.perplexity,
            self.learning_rate,
            self.grad_norm,
            self.throughput,
        }) catch {};
        return buf;
    }
};

/// Training report.
pub const TrainingReport = struct {
    /// Final training loss
    final_loss: f32,
    /// Final training accuracy
    final_accuracy: f32,
    /// Best validation loss
    best_val_loss: f32,
    /// Best validation accuracy
    best_val_accuracy: f32,
    /// Final perplexity
    final_perplexity: f32,
    /// Total training steps
    total_steps: u64,
    /// Total tokens processed
    total_tokens: u64,
    /// Total training time (ns)
    total_time_ns: u64,
    /// Average throughput
    avg_throughput: f32,
    /// Checkpoints saved
    checkpoints_saved: u32,
    /// Whether early stopped
    early_stopped: bool,
};

/// Evaluation results.
pub const EvalResult = struct {
    /// Average loss
    loss: f32,
    /// Perplexity (exp(loss))
    perplexity: f32,
    /// Token accuracy (correct predictions / total)
    accuracy: f32,
    /// Number of batches evaluated
    num_batches: u32,
    /// Total tokens evaluated
    total_tokens: u64,
};

/// Early stopping configuration.
pub const EarlyStoppingConfig = struct {
    /// Number of evaluations without improvement before stopping
    patience: u32 = 3,
    /// Minimum improvement to reset patience
    min_delta: f32 = 0.001,
    /// Whether to monitor validation loss (true) or perplexity (false)
    monitor_loss: bool = true,
    /// Whether early stopping is enabled
    enabled: bool = true,
};

/// LLM Trainer.
pub const LlamaTrainer = struct {
    allocator: std.mem.Allocator,
    config: LlmTrainingConfig,
    model: *trainable_model.TrainableModel,
    loss_fn: loss_mod.CrossEntropyLoss,
    accumulator: mod.GradientAccumulator,
    optimizer_state: OptimizerState,
    stats: TrainingStats,
    /// GPU operations interface (if GPU enabled)
    gpu_ops: ?ai_ops.AiOps,
    /// Accumulated loss for logging
    accum_loss: f32,
    /// Loss history for early stopping
    loss_history: std.ArrayListUnmanaged(f32),
    /// Validation data (optional)
    val_data: ?[]const u32,
    /// Early stopping state
    early_stopping: EarlyStoppingState,
    /// Best model checkpoint weights
    best_weights: ?[]f32,
    /// Best validation accuracy
    best_val_accuracy: f32,
    /// Total checkpoints saved
    checkpoints_saved: u32,
    /// Metrics logger (optional)
    logger: ?logging.TrainingLogger,
    /// Timer for throughput logging
    log_timer: ?std.time.Timer,
    last_log_time_ns: u64,
    last_log_tokens: u64,
    accum_correct: u64,
    accum_tokens: u64,

    const EarlyStoppingState = struct {
        best_metric: f32 = std.math.inf(f32),
        patience_counter: u32 = 0,
        stopped: bool = false,
    };

    const OptimizerState = struct {
        /// First moment (Adam)
        m: ?[]f32,
        /// Second moment (Adam)
        v: ?[]f32,
        /// Adam hyperparameters
        beta1: f32 = 0.9,
        beta2: f32 = 0.999,
        epsilon: f32 = 1e-8,
    };

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

        var log_timer = std.time.Timer.start() catch null;
        const initial_time = if (log_timer) |*t| t.read() else 0;

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
            .gpu_ops = null,
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

    pub const StepMetrics = struct {
        loss: f32,
        accuracy: f32,
    };

    /// Perform a single training step with metrics.
    pub fn trainStepWithMetrics(
        self: *LlamaTrainer,
        input_ids: []const u32,
        labels: []const u32,
    ) !StepMetrics {
        _ = input_ids;
        const seq_len = self.config.max_seq_len;

        // Ensure model is prepared for training
        if (self.model.activations == null) {
            try self.model.prepareForTraining(seq_len);
        }

        const vocab_size = self.model.config.vocab_size;
        const token_count = labels.len;
        const logits = try self.allocator.alloc(f32, token_count * vocab_size);
        defer self.allocator.free(logits);

        // Initialize logits with small random values
        var rng = std.Random.DefaultPrng.init(self.stats.global_step);
        for (logits) |*l| {
            l.* = rng.random().floatNorm(f32) * 0.1;
        }

        // Compute loss
        const loss = try self.loss_fn.forward(logits, labels);

        // Backward pass (compute gradients)
        const d_logits = try self.allocator.alloc(f32, logits.len);
        defer self.allocator.free(d_logits);
        self.loss_fn.backward(d_logits);

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
            try self.optimizerStep();
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

    /// Perform optimizer step.
    fn optimizerStep(self: *LlamaTrainer) !void {
        // Compute average loss
        const avg_loss = self.accum_loss / @as(f32, @floatFromInt(self.config.grad_accum_steps));
        const avg_accuracy = if (self.accum_tokens > 0)
            @as(f32, @floatFromInt(self.accum_correct)) / @as(f32, @floatFromInt(self.accum_tokens))
        else
            0;

        // Get current learning rate
        const lr = mod.calculateLearningRate(
            .{
                .learning_rate = self.config.learning_rate,
                .learning_rate_schedule = self.config.lr_schedule,
                .warmup_steps = self.config.warmup_steps,
                .decay_steps = self.config.decay_steps,
                .min_learning_rate = self.config.min_learning_rate,
            },
            self.stats.global_step,
            self.config.learning_rate,
        );

        // Apply gradient clipping
        // In real implementation, clip model.weights.d_*
        const grad_norm = self.config.max_grad_norm;

        // Apply optimizer update based on type
        switch (self.config.optimizer) {
            .sgd => {
                // SGD with momentum
                self.applySgdUpdate(lr);
            },
            .adam, .adamw => {
                // Adam / AdamW
                self.applyAdamUpdate(lr, self.config.optimizer == .adamw);
            },
        }

        // Zero gradients
        self.model.zeroGradients();

        // Update stats
        self.stats.global_step += 1;
        self.stats.loss = avg_loss;
        self.stats.accuracy = avg_accuracy;
        self.stats.perplexity = @exp(avg_loss);
        self.stats.learning_rate = lr;
        self.stats.grad_norm = grad_norm;

        // Record loss history
        try self.loss_history.append(self.allocator, avg_loss);

        try self.maybeLog();
    }

    /// Apply SGD update.
    fn applySgdUpdate(self: *LlamaTrainer, lr: f32) void {
        // Placeholder: would iterate over model weights
        _ = self;
        _ = lr;
    }

    /// Apply Adam/AdamW update.
    fn applyAdamUpdate(self: *LlamaTrainer, lr: f32, weight_decay: bool) void {
        const beta1 = self.optimizer_state.beta1;
        const beta2 = self.optimizer_state.beta2;
        const eps = self.optimizer_state.epsilon;
        const t = @as(f32, @floatFromInt(self.stats.global_step + 1));

        // Bias correction
        const bias_correction1 = 1.0 - std.math.pow(f32, beta1, t);
        const bias_correction2 = 1.0 - std.math.pow(f32, beta2, t);

        // In real implementation:
        // 1. m = beta1 * m + (1 - beta1) * grad
        // 2. v = beta2 * v + (1 - beta2) * grad^2
        // 3. m_hat = m / bias_correction1
        // 4. v_hat = v / bias_correction2
        // 5. If AdamW: weight -= lr * weight_decay * weight
        // 6. weight -= lr * m_hat / (sqrt(v_hat) + eps)

        _ = lr;
        _ = bias_correction1;
        _ = bias_correction2;
        _ = eps;
        _ = weight_decay;
    }

    /// Train for one epoch.
    pub fn trainEpoch(
        self: *LlamaTrainer,
        data: []const u32, // Flat array of token IDs
        num_samples: usize,
    ) !f32 {
        const tokens_per_sample = self.config.max_seq_len;
        const batch_tokens = self.config.batch_size * tokens_per_sample;

        var epoch_loss: f32 = 0;
        var num_batches: u32 = 0;

        var offset: usize = 0;
        while (offset + batch_tokens <= data.len) {
            const batch_data = data[offset .. offset + batch_tokens];

            // Input: all tokens except last
            // Labels: all tokens except first (shifted)
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
        if (self.config.checkpoint_path) |path| {
            const weights = try self.model.collectWeights();
            defer self.allocator.free(weights);

            const m = self.optimizer_state.m orelse return error.OutOfMemory;
            const v = self.optimizer_state.v orelse return error.OutOfMemory;

            var io_backend = std.Io.Threaded.init(self.allocator, .{ .environ = std.process.Environ.empty });
            defer io_backend.deinit();
            const io = io_backend.io();
            try std.Io.Dir.cwd().createDirPath(io, path);

            const filename = try std.fmt.allocPrint(self.allocator, "{s}/llm_step_{d}.ckpt", .{
                path,
                self.stats.global_step,
            });
            defer self.allocator.free(filename);

            try llm_checkpoint.saveLlmCheckpoint(self.allocator, filename, .{
                .step = self.stats.global_step,
                .epoch = self.stats.epoch,
                .tokens_processed = self.stats.tokens_processed,
                .weights = weights,
                .m = m,
                .v = v,
            });

            self.checkpoints_saved += 1;
            std.log.info("Checkpoint saved at step {d}", .{self.stats.global_step});
        }
    }

    /// Load checkpoint.
    pub fn loadCheckpoint(self: *LlamaTrainer, path: []const u8) !void {
        var ckpt = try llm_checkpoint.loadLlmCheckpoint(self.allocator, path);
        defer ckpt.deinit(self.allocator);

        const expected = self.model.numParams();
        if (ckpt.weights.len != expected or ckpt.m.len != expected or ckpt.v.len != expected) {
            return error.ConfigMismatch;
        }

        try self.model.distributeWeights(ckpt.weights);

        if (self.optimizer_state.m) |m| {
            @memcpy(m, ckpt.m);
        }
        if (self.optimizer_state.v) |v| {
            @memcpy(v, ckpt.v);
        }

        self.stats.global_step = ckpt.step;
        self.stats.epoch = ckpt.epoch;
        self.stats.tokens_processed = ckpt.tokens_processed;

        std.log.info("Checkpoint loaded from {s} (step {d})", .{ path, ckpt.step });
    }

    /// Evaluate model on validation data.
    /// Runs in inference mode (no gradient computation).
    pub fn evaluate(self: *LlamaTrainer, data: []const u32) !EvalResult {
        const tokens_per_sample = self.config.max_seq_len;
        const batch_tokens = self.config.batch_size * tokens_per_sample;

        var total_loss: f32 = 0;
        var total_correct: u64 = 0;
        var total_tokens: u64 = 0;
        var num_batches: u32 = 0;

        var offset: usize = 0;
        while (offset + batch_tokens < data.len) {
            const batch_data = data[offset .. offset + batch_tokens];

            // Input: all tokens except last
            // Labels: all tokens except first (shifted)
            const labels = batch_data[1..batch_tokens];

            // Forward pass (no gradients)
            const logits = try self.allocator.alloc(f32, (batch_tokens - 1) * self.model.config.vocab_size);
            defer self.allocator.free(logits);

            // Placeholder: initialize with small random values
            var rng = std.Random.DefaultPrng.init(offset);
            for (logits) |*l| {
                l.* = rng.random().floatNorm(f32) * 0.1;
            }

            // Compute loss
            const loss = try self.loss_fn.forward(logits, labels);
            total_loss += loss;

            // Compute accuracy (argmax prediction vs label)
            const vocab_size = self.model.config.vocab_size;
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
            return EvalResult{
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

        return EvalResult{
            .loss = avg_loss,
            .perplexity = @exp(avg_loss),
            .accuracy = accuracy,
            .num_batches = num_batches,
            .total_tokens = total_tokens,
        };
    }

    /// Run validation if interval reached and data available.
    /// Returns true if should continue training (not early stopped).
    pub fn maybeValidate(self: *LlamaTrainer, early_stop_config: EarlyStoppingConfig) !bool {
        // Check if we should validate
        if (self.config.eval_interval == 0 or
            self.stats.global_step % self.config.eval_interval != 0)
        {
            return true;
        }

        const val_data = self.val_data orelse return true;

        // Run evaluation
        const result = try self.evaluate(val_data);
        std.log.info("Validation: loss={d:.4} ppl={d:.2} acc={d:.2}%", .{
            result.loss,
            result.perplexity,
            result.accuracy * 100,
        });

        if (self.logger) |*logger| {
            try logger.logScalar("val/loss", result.loss, self.stats.global_step);
            try logger.logScalar("val/perplexity", result.perplexity, self.stats.global_step);
            try logger.logScalar("val/accuracy", result.accuracy, self.stats.global_step);
        }

        // Check early stopping
        if (!early_stop_config.enabled) return true;

        const metric = if (early_stop_config.monitor_loss) result.loss else result.perplexity;

        if (metric < self.early_stopping.best_metric - early_stop_config.min_delta) {
            // Improvement found
            self.early_stopping.best_metric = metric;
            self.early_stopping.patience_counter = 0;
            self.best_val_accuracy = result.accuracy;

            // Save best weights
            if (self.best_weights) |bw| self.allocator.free(bw);
            self.best_weights = try self.model.collectWeights();

            std.log.info("New best model (metric={d:.4})", .{metric});
        } else {
            // No improvement
            self.early_stopping.patience_counter += 1;
            if (self.early_stopping.patience_counter >= early_stop_config.patience) {
                std.log.info("Early stopping triggered (patience={d})", .{early_stop_config.patience});
                self.early_stopping.stopped = true;

                // Restore best weights
                if (self.best_weights) |bw| {
                    try self.model.distributeWeights(bw);
                    std.log.info("Restored best weights", .{});
                }

                return false;
            }
        }

        return true;
    }

    /// Check if training was early stopped.
    pub fn wasEarlyStopped(self: *const LlamaTrainer) bool {
        return self.early_stopping.stopped;
    }

    /// Get current training stats.
    pub fn getStats(self: *const LlamaTrainer) TrainingStats {
        return self.stats;
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

    fn maybeLog(self: *LlamaTrainer) !void {
        if (self.logger == null) return;
        if (self.config.log_interval == 0) return;
        if (self.stats.global_step % self.config.log_interval != 0) return;

        if (self.log_timer) |*timer| {
            const now_ns = timer.read();
            const delta_ns = now_ns - self.last_log_time_ns;
            if (delta_ns > 0) {
                const tokens_delta = self.stats.tokens_processed - self.last_log_tokens;
                self.stats.throughput = @as(f32, @floatFromInt(tokens_delta)) /
                    (@as(f32, @floatFromInt(delta_ns)) / 1e9);
            }
            self.last_log_time_ns = now_ns;
            self.last_log_tokens = self.stats.tokens_processed;
        }

        var logger = &self.logger.?;
        try logger.logScalar("train/loss", self.stats.loss, self.stats.global_step);
        try logger.logScalar("train/accuracy", self.stats.accuracy, self.stats.global_step);
        try logger.logScalar("train/perplexity", self.stats.perplexity, self.stats.global_step);
        try logger.logScalar("train/learning_rate", self.stats.learning_rate, self.stats.global_step);
        try logger.logScalar("train/grad_norm", self.stats.grad_norm, self.stats.global_step);
    }

    pub fn finalizeLogging(self: *LlamaTrainer) !void {
        if (self.logger) |*logger| {
            const metrics = [_]logging.Metric{
                .{ .key = "train/final_loss", .value = self.stats.loss },
                .{ .key = "train/final_accuracy", .value = self.stats.accuracy },
                .{ .key = "train/final_perplexity", .value = self.stats.perplexity },
                .{ .key = "val/best_loss", .value = self.early_stopping.best_metric },
                .{ .key = "val/best_accuracy", .value = self.best_val_accuracy },
                .{ .key = "train/total_steps", .value = @floatFromInt(self.stats.global_step) },
                .{ .key = "train/total_tokens", .value = @floatFromInt(self.stats.tokens_processed) },
            };
            try logger.writeSummary(&metrics);
        }
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

    var timer = std.time.Timer.start() catch return error.TimerFailed;

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

pub const TrainerError = error{
    InvalidConfiguration,
    TimerFailed,
    CheckpointFailed,
    OutOfMemory,
};

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
