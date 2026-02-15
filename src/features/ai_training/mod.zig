//! AI Training Module — Training Pipelines, Federated Learning
//!
//! This module provides model training infrastructure: training pipelines,
//! fine-tuning, checkpointing, data loading, federated learning, and the
//! AI-database bridge (WDBX token datasets).
//!
//! Gated by `-Denable-training`.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../core/config/mod.zig");

// ============================================================================
// Sub-module re-exports (from features/ai/)
// ============================================================================

pub const training = if (build_options.enable_training)
    @import("../ai/training/mod.zig")
else
    @import("../ai/training/stub.zig");

pub const federated = @import("../ai/federated/mod.zig");

pub const database = if (build_options.enable_training)
    @import("../ai/database/mod.zig")
else
    @import("../ai/database/stub.zig");

// ============================================================================
// Convenience type re-exports
// ============================================================================

pub const TrainingConfig = training.TrainingConfig;
pub const TrainingReport = training.TrainingReport;
pub const TrainingResult = training.TrainingResult;
pub const TrainError = training.TrainError;
pub const OptimizerType = training.OptimizerType;
pub const LearningRateSchedule = training.LearningRateSchedule;
pub const CheckpointStore = training.CheckpointStore;
pub const Checkpoint = training.Checkpoint;
pub const LlmTrainingConfig = training.LlmTrainingConfig;
pub const trainable_model = training.trainable_model;
pub const TrainableModel = training.TrainableModel;
pub const TrainableModelConfig = training.trainable_model.TrainableModelConfig;
pub const LlamaTrainer = training.LlamaTrainer;
pub const loadCheckpoint = training.loadCheckpoint;
pub const saveCheckpoint = training.saveCheckpoint;
pub const TrainableViTModel = training.TrainableViTModel;
pub const TrainableViTConfig = training.TrainableViTConfig;
pub const TrainableViTWeights = training.TrainableViTWeights;
pub const VisionTrainingError = training.VisionTrainingError;
pub const TrainableCLIPModel = training.TrainableCLIPModel;
pub const CLIPTrainingConfig = training.CLIPTrainingConfig;
pub const MultimodalTrainingError = training.MultimodalTrainingError;
pub const TokenizedDataset = training.TokenizedDataset;
pub const DataLoader = training.DataLoader;
pub const BatchIterator = training.BatchIterator;
pub const Batch = training.Batch;
pub const SequencePacker = training.SequencePacker;
pub const parseInstructionDataset = training.parseInstructionDataset;

// Database/WDBX bridge
pub const WdbxTokenDataset = database.WdbxTokenDataset;
pub const tokenBinToWdbx = database.tokenBinToWdbx;
pub const wdbxToTokenBin = database.wdbxToTokenBin;
pub const readTokenBinFile = database.readTokenBinFile;
pub const writeTokenBinFile = database.writeTokenBinFile;
pub const exportGguf = database.exportGguf;

// ============================================================================
// Error
// ============================================================================

pub const Error = error{
    TrainingDisabled,
    InvalidConfig,
};

// ============================================================================
// Context
// ============================================================================

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AiConfig,
    training_ctx: ?*training.Context = null,

    pub fn init(
        allocator: std.mem.Allocator,
        cfg: config_module.AiConfig,
    ) !*Context {
        if (!isEnabled()) return error.TrainingDisabled;

        const ctx = try allocator.create(Context);
        errdefer allocator.destroy(ctx);

        ctx.* = .{
            .allocator = allocator,
            .config = cfg,
        };

        if (cfg.training) |train_cfg| {
            ctx.training_ctx = try training.Context.init(
                allocator,
                train_cfg,
            );
        }

        return ctx;
    }

    pub fn deinit(self: *Context) void {
        if (self.training_ctx) |t| t.deinit();
        self.allocator.destroy(self);
    }

    pub fn getTraining(self: *Context) Error!*training.Context {
        return self.training_ctx orelse error.TrainingDisabled;
    }
};

// ============================================================================
// Module-level functions
// ============================================================================

pub fn isEnabled() bool {
    return build_options.enable_training;
}

pub fn train(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!TrainingReport {
    return training.trainAndReport(allocator, config);
}

pub fn trainWithResult(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!TrainingResult {
    return training.trainWithResult(allocator, config);
}

// ============================================================================
// Tests
// ============================================================================

test "ai_training module loads" {
    try std.testing.expect(@TypeOf(TrainingConfig) != void);
    try std.testing.expect(@TypeOf(TrainableModel) != void);
}

test "ai_training isEnabled reflects build flag" {
    try std.testing.expectEqual(build_options.enable_training, isEnabled());
}

test "ai_training type re-exports" {
    try std.testing.expect(@sizeOf(TrainingConfig) > 0);
    try std.testing.expect(@TypeOf(TrainingReport) != void);
    try std.testing.expect(@TypeOf(OptimizerType) != void);
    try std.testing.expect(@TypeOf(LearningRateSchedule) != void);
}
