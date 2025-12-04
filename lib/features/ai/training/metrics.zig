const std = @import("std");

/// Comprehensive training metrics shared across subsystems.
pub const Metrics = struct {
    // Basic metrics
    loss: f32,
    accuracy: f32 = 0.0,
    val_loss: ?f32 = null,
    val_accuracy: ?f32 = null,

    // Additional metrics
    precision: ?f32 = null,
    recall: ?f32 = null,
    f1_score: ?f32 = null,
    auc_roc: ?f32 = null,

    // Training progress
    epoch: usize,
    step: usize = 0,
    training_time_ms: u64,
    inference_time_ms: ?u64 = null,

    // Performance metrics
    throughput_samples_per_sec: f32 = 0.0,
    memory_usage_mb: f32 = 0.0,
    gpu_utilization: ?f32 = null,

    // Learning dynamics
    learning_rate: f32,
    gradient_norm: ?f32 = null,
    weight_norm: ?f32 = null,

    // Custom metrics
    custom_metrics: ?std.StringHashMap(f32) = null,
};

/// Backwards compatibility alias for legacy call sites.
pub const TrainingMetrics = Metrics;
