//! Training Sub-module
//!
//! Model training pipelines with checkpointing and distributed support.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../config.zig");

// Re-export from existing training module
const features_training = @import("../../features/ai/training/mod.zig");

pub const TrainingConfig = features_training.TrainingConfig;
pub const TrainingReport = features_training.TrainingReport;
pub const TrainingResult = features_training.TrainingResult;
pub const TrainError = features_training.TrainError;
pub const OptimizerType = features_training.OptimizerType;
pub const LearningRateSchedule = features_training.LearningRateSchedule;
pub const CheckpointStore = features_training.CheckpointStore;
pub const Checkpoint = features_training.Checkpoint;
pub const GradientAccumulator = features_training.GradientAccumulator;

// LLM training
pub const LlmTrainingConfig = features_training.LlmTrainingConfig;
pub const LlamaTrainer = features_training.LlamaTrainer;
pub const TrainableModel = features_training.TrainableModel;

pub const Error = error{
    TrainingDisabled,
    InvalidConfig,
    CheckpointFailed,
    TrainingFailed,
    OutOfMemory,
};

/// Training context for framework integration.
pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.TrainingConfig,
    checkpoint_store: ?*CheckpointStore = null,

    pub fn init(allocator: std.mem.Allocator, cfg: config_module.TrainingConfig) !*Context {
        if (!isEnabled()) return error.TrainingDisabled;

        const ctx = try allocator.create(Context);
        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };
        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.checkpoint_store) |cs| {
            cs.deinit();
            self.allocator.destroy(cs);
        }
        self.allocator.destroy(self);
    }

    /// Run training with the given configuration.
    pub fn train(self: *Context, train_config: TrainingConfig) !TrainingResult {
        return features_training.trainWithResult(self.allocator, train_config);
    }

    /// Get or create checkpoint store.
    pub fn getCheckpointStore(self: *Context) !*CheckpointStore {
        if (self.checkpoint_store) |cs| return cs;

        const store = try self.allocator.create(CheckpointStore);
        store.* = try CheckpointStore.init(self.allocator, self.config.checkpoint_dir orelse "checkpoints");
        self.checkpoint_store = store;
        return store;
    }

    /// Save a checkpoint.
    pub fn saveCheckpoint(self: *Context, name: []const u8, data: anytype) !void {
        const store = try self.getCheckpointStore();
        try store.save(name, data);
    }

    /// Load a checkpoint.
    pub fn loadCheckpoint(self: *Context, name: []const u8, comptime T: type) !T {
        const store = try self.getCheckpointStore();
        return store.load(name, T);
    }
};

pub fn isEnabled() bool {
    return build_options.enable_ai;
}

// Convenience functions
pub fn trainAndReport(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingReport {
    return features_training.trainAndReport(allocator, config);
}

pub fn trainWithResult(allocator: std.mem.Allocator, config: TrainingConfig) TrainError!TrainingResult {
    return features_training.trainWithResult(allocator, config);
}

pub fn trainLlm(allocator: std.mem.Allocator, config: LlmTrainingConfig) !void {
    return features_training.trainLlm(allocator, config);
}
