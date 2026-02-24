//! AI Training Stub Module — disabled when AI training is off.

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");

pub const Error = error{ TrainingDisabled, InvalidConfig };

// Sub-module stubs
pub const training = @import("../training/stub.zig");
pub const federated = @import("../federated/stub.zig");
pub const database = @import("../database/stub.zig");

// Re-exports
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
pub const WdbxTokenDataset = database.WdbxTokenDataset;
pub const readTokenBinFile = database.readTokenBinFile;
pub const writeTokenBinFile = database.writeTokenBinFile;
pub const tokenBinToWdbx = database.tokenBinToWdbx;
pub const wdbxToTokenBin = database.wdbxToTokenBin;
pub const exportGguf = database.exportGguf;

pub const Context = struct {
    allocator: std.mem.Allocator,
    config: config_module.AiConfig,
    training_ctx: ?*training.Context = null,

    pub fn init(_: std.mem.Allocator, _: config_module.AiConfig) !*Context {
        return error.TrainingDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn getTraining(_: *Context) Error!*training.Context {
        return error.TrainingDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}
pub fn train(_: std.mem.Allocator, _: TrainingConfig) TrainError!TrainingReport {
    return error.TrainingDisabled;
}
pub fn trainWithResult(_: std.mem.Allocator, _: TrainingConfig) TrainError!TrainingResult {
    return error.TrainingDisabled;
}

test {
    std.testing.refAllDecls(@This());
}
