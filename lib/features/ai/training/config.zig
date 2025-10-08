const optimizers = @import("../optimizers/mod.zig");

/// Data augmentation techniques that can be shared across training pipelines.
pub const DataAugmentation = struct {
    horizontal_flip: bool = false,
    vertical_flip: bool = false,
    rotation_range: f32 = 0.0,
    width_shift_range: f32 = 0.0,
    height_shift_range: f32 = 0.0,
    brightness_range: ?[2]f32 = null,
    zoom_range: f32 = 0.0,
    channel_shift_range: f32 = 0.0,
    fill_mode: enum { constant, nearest, reflect, wrap } = .constant,
    gaussian_noise_std: f32 = 0.0,
    cutout_probability: f32 = 0.0,
    cutout_size: ?[2]usize = null,
    mixup_alpha: f32 = 0.0,
    cutmix_alpha: f32 = 0.0,
};

/// Model training configuration with advanced options reused across subsystems.
pub const Config = struct {
    // Basic training parameters
    batch_size: usize = 32,
    epochs: usize = 100,
    validation_split: f32 = 0.2,
    optimizer: optimizers.OptimizerConfig = .{},

    // Early stopping and checkpointing
    early_stopping_patience: usize = 10,
    early_stopping_min_delta: f32 = 0.001,
    save_best_only: bool = true,
    checkpoint_frequency: usize = 10,

    // Regularization
    gradient_clipping: ?f32 = null,
    gradient_clipping_norm: enum { l1, l2, inf } = .l2,

    // Data augmentation
    data_augmentation: ?DataAugmentation = null,

    // Distributed training
    use_mixed_precision: bool = false,
    accumulate_gradients: usize = 1,
    sync_batch_norm: bool = false,

    // Monitoring and logging
    log_frequency: usize = 100,
    validate_frequency: usize = 1,
    tensorboard_logging: bool = false,
    profiling_enabled: bool = false,
};

/// Backwards compatibility alias for legacy call sites.
pub const TrainingConfig = Config;
