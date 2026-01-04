//! Training pipeline utilities, gradient aggregation, and checkpointing.
const std = @import("std");
const checkpoint = @import("checkpoint.zig");
const gradient = @import("gradient.zig");

pub const Checkpoint = checkpoint.Checkpoint;
pub const CheckpointError = checkpoint.CheckpointError;
pub const CheckpointStore = checkpoint.CheckpointStore;
pub const CheckpointView = checkpoint.CheckpointView;
pub const LoadCheckpointError = checkpoint.LoadError;
pub const SaveCheckpointError = checkpoint.SaveError;
pub const SaveLatestCheckpointError = checkpoint.SaveLatestError;
pub const GradientAccumulator = gradient.GradientAccumulator;
pub const GradientError = gradient.GradientError;

pub const TrainingError = error{
    InvalidConfiguration,
};

pub const TrainError =
    TrainingError ||
    GradientError ||
    SaveLatestCheckpointError ||
    std.mem.Allocator.Error;

pub const TrainingConfig = struct {
    epochs: u32 = 1,
    batch_size: u32 = 16,
    sample_count: u32 = 256,
    model_size: u32 = 128,
    learning_rate: f32 = 0.001,
    gradient_accumulation_steps: u32 = 1,
    checkpoint_interval: u32 = 0,
    max_checkpoints: u32 = 3,
    checkpoint_path: ?[]const u8 = null,

    /// Validate configuration values.
    pub fn validate(self: TrainingConfig) TrainingError!void {
        if (self.epochs == 0) return TrainingError.InvalidConfiguration;
        if (self.batch_size == 0) return TrainingError.InvalidConfiguration;
        if (self.sample_count == 0) return TrainingError.InvalidConfiguration;
        if (self.model_size == 0) return TrainingError.InvalidConfiguration;
        if (self.learning_rate <= 0) return TrainingError.InvalidConfiguration;
        if (self.gradient_accumulation_steps == 0) {
            return TrainingError.InvalidConfiguration;
        }
    }
};

pub const TrainingReport = struct {
    epochs: u32,
    batches: u32,
    final_loss: f32,
    final_accuracy: f32,
    gradient_updates: u64 = 0,
    checkpoints_saved: u32 = 0,
};

pub const ModelState = struct {
    allocator: std.mem.Allocator,
    weights: []f32,
    step: u64,

    /// Initialize model state with a fixed number of weights.
    /// @param allocator Memory allocator for allocations
    /// @param size Number of weights to allocate
    /// @return Initialized ModelState
    pub fn init(allocator: std.mem.Allocator, size: usize) !ModelState {
        const weights = try allocator.alloc(f32, size);
        initializeWeights(weights);
        return .{
            .allocator = allocator,
            .weights = weights,
            .step = 0,
        };
    }

    /// Release resources owned by the model state.
    pub fn deinit(self: *ModelState) void {
        self.allocator.free(self.weights);
        self.* = undefined;
    }
};

pub const TrainingResult = struct {
    report: TrainingReport,
    model: ModelState,
    checkpoints: CheckpointStore,

    /// Release resources owned by the training result.
    pub fn deinit(self: *TrainingResult) void {
        self.model.deinit();
        self.checkpoints.deinit();
        self.* = undefined;
    }
};

/// Run a training session and discard detailed state.
/// @param allocator Memory allocator for allocations
/// @param config Training configuration
pub fn train(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!void {
    var result = try trainWithResult(allocator, config);
    defer result.deinit();
}

/// Run a training session and return summary metrics.
/// @param allocator Memory allocator for allocations
/// @param config Training configuration
/// @return TrainingReport for the run
pub fn trainAndReport(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!TrainingReport {
    var result = try trainWithResult(allocator, config);
    defer result.deinit();
    return result.report;
}

/// Run a training session and return model state plus checkpoints.
/// @param allocator Memory allocator for allocations
/// @param config Training configuration
/// @return TrainingResult including checkpoints and model weights
pub fn trainWithResult(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!TrainingResult {
    try config.validate();

    var model = try ModelState.init(allocator, config.model_size);
    errdefer model.deinit();

    var accumulator = try GradientAccumulator.init(allocator, model.weights.len);
    defer accumulator.deinit();

    var checkpoints = CheckpointStore.init(allocator, config.max_checkpoints);
    errdefer checkpoints.deinit();

    const batches_per_epoch =
        (config.sample_count + config.batch_size - 1) / config.batch_size;
    const gradient_buffer = try allocator.alloc(f32, model.weights.len);
    defer allocator.free(gradient_buffer);

    var loss: f32 = 1.0;
    var accuracy: f32 = 0.5;
    var gradient_updates: u64 = 0;
    var checkpoints_saved: u32 = 0;

    var epoch: u32 = 0;
    while (epoch < config.epochs) : (epoch += 1) {
        var batch: u32 = 0;
        while (batch < batches_per_epoch) : (batch += 1) {
            simulateGradient(gradient_buffer, config.learning_rate, epoch, batch);
            try accumulator.add(gradient_buffer);

            const is_last_batch = batch + 1 == batches_per_epoch;
            if (accumulator.count >= config.gradient_accumulation_steps or is_last_batch) {
                try accumulator.apply(model.weights, config.learning_rate);
                accumulator.reset();
                model.step += 1;
                gradient_updates += 1;

                if (config.checkpoint_interval > 0 and
                    model.step % config.checkpoint_interval == 0)
                {
                    try checkpoints.add(model.step, model.weights);
                    checkpoints_saved += 1;
                    if (config.checkpoint_path) |path| {
                        try checkpoints.saveLatestToFile(path);
                    }
                }
            }

            const step = config.learning_rate * 0.1;
            loss = if (loss > step) loss - step else 0.0;
            accuracy = @min(1.0, accuracy + config.learning_rate * 0.05);
        }
    }

    if (config.checkpoint_path != null and config.checkpoint_interval == 0 and model.step > 0) {
        try checkpoints.add(model.step, model.weights);
        checkpoints_saved += 1;
        try checkpoints.saveLatestToFile(config.checkpoint_path.?);
    }

    return .{
        .report = .{
            .epochs = config.epochs,
            .batches = batches_per_epoch,
            .final_loss = loss,
            .final_accuracy = accuracy,
            .gradient_updates = gradient_updates,
            .checkpoints_saved = checkpoints_saved,
        },
        .model = model,
        .checkpoints = checkpoints,
    };
}

fn initializeWeights(weights: []f32) void {
    for (weights, 0..) |*value, i| {
        const scale: f32 = @floatFromInt((i % 10) + 1);
        value.* = scale * 0.01;
    }
}

fn simulateGradient(
    buffer: []f32,
    learning_rate: f32,
    epoch: u32,
    batch: u32,
) void {
    const epoch_factor: f32 = @as(f32, @floatFromInt((epoch % 5) + 1)) * 0.001;
    const batch_factor: f32 = @as(f32, @floatFromInt((batch % 7) + 1)) * 0.0005;
    for (buffer, 0..) |*value, i| {
        const index_factor: f32 = @as(f32, @floatFromInt((i % 13) + 1)) * 0.0003;
        value.* = learning_rate + epoch_factor + batch_factor + index_factor;
    }
}

test "training result includes checkpoints" {
    var result = try trainWithResult(std.testing.allocator, .{
        .epochs = 1,
        .batch_size = 2,
        .sample_count = 4,
        .model_size = 8,
        .gradient_accumulation_steps = 1,
        .checkpoint_interval = 1,
        .max_checkpoints = 2,
    });
    defer result.deinit();

    try std.testing.expect(result.checkpoints.count() <= 2);
    try std.testing.expect(result.report.checkpoints_saved >= 1);
}

test "training honors gradient accumulation steps" {
    var result = try trainWithResult(std.testing.allocator, .{
        .epochs = 1,
        .batch_size = 2,
        .sample_count = 4,
        .model_size = 4,
        .gradient_accumulation_steps = 2,
    });
    defer result.deinit();

    try std.testing.expectEqual(@as(u64, 1), result.model.step);
    try std.testing.expectEqual(@as(u64, 1), result.report.gradient_updates);
}
