//! LLM Trainer types — shared between all llm_trainer submodules.

const std = @import("std");
const mod = @import("../mod.zig");
const logging = @import("../logging.zig");

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
    /// Enable accelerator path by default (GPU/NPU backends auto-selected when available).
    use_gpu: bool = true,
    /// GPU acceleration backend preference
    gpu_backend: ?[]const u8 = null,
    /// Threshold for GPU dispatch (batch size below this uses CPU)
    gpu_batch_threshold: u32 = 8,
    /// Device memory buffer for GPU operations
    gpu_device_buffer_mb: u32 = 256,

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
        }) catch |err| {
            std.log.debug("TrainingProgress format buffer overflow: {t}", .{err});
        };
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

pub const TrainerError = error{
    InvalidConfiguration,
    TimerFailed,
    CheckpointFailed,
    OutOfMemory,
};

pub const EarlyStoppingState = struct {
    best_metric: f32 = std.math.inf(f32),
    patience_counter: u32 = 0,
    stopped: bool = false,
};

pub const OptimizerState = struct {
    /// First moment (Adam)
    m: ?[]f32,
    /// Second moment (Adam)
    v: ?[]f32,
    /// Adam hyperparameters
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
};

pub const StepMetrics = struct {
    loss: f32,
    accuracy: f32,
};

test {
    std.testing.refAllDecls(@This());
}
