const std = @import("std");
const adaptive_mod = @import("../execution/adaptive.zig");

pub const ExecutionMethod = enum {
    accelerate,
    gpu,
    simd,
    scalar,
    failed,
};

pub const CoordinatorConfig = struct {
    prefer_gpu: bool = true,
    fallback_chain: []const ExecutionMethod = &.{ .gpu, .simd, .scalar },
    gpu_threshold_size: usize = 1024, // Min elements for GPU
    simd_threshold_size: usize = 4, // Min elements for SIMD
    backend_timeout_ms: u64 = 1000,
    /// Enable logging when fallback occurs (useful for debugging)
    log_fallbacks: bool = false,
    /// Enable adaptive threshold tuning based on runtime performance
    enable_adaptive_thresholds: bool = true,
    /// Sample window for adaptive threshold calculations
    adaptive_sample_window: usize = 100,
    /// Minimum improvement factor to change method (1.1 = 10% faster)
    adaptive_min_improvement: f64 = 1.1,
};

pub const OperationType = enum {
    vector_add,
    vector_multiply,
    matrix_multiply,
    dot_product,
    activation,
};

/// Performance sample for adaptive threshold learning (re-exported)
pub const PerformanceSample = adaptive_mod.PerformanceSample;

/// Adaptive threshold manager (re-exported from execution/adaptive.zig)
pub const AdaptiveThresholds = adaptive_mod.AdaptiveThresholds;
