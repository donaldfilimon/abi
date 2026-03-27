//! Training Stub Module — disabled at compile time.

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");
pub const types = @import("types.zig");

// ── Re-exported types ──────────────────────────────────────────────────────

pub const Error = types.Error;
pub const CheckpointError = types.CheckpointError;
pub const TrainError = types.TrainError;
pub const GradientError = types.GradientError;
pub const VisionTrainingError = types.VisionTrainingError;
pub const MultimodalTrainingError = types.MultimodalTrainingError;
pub const TrainingError = types.TrainingError;
pub const ModelLoadError = types.ModelLoadError;
pub const SaveCheckpointError = types.SaveCheckpointError;
pub const LoadCheckpointError = types.LoadCheckpointError;
pub const SaveLatestCheckpointError = types.SaveLatestCheckpointError;
pub const LoadLlmCheckpointError = types.LoadLlmCheckpointError;
pub const SaveLlmCheckpointError = types.SaveLlmCheckpointError;
pub const OptimizerType = types.OptimizerType;
pub const PrecisionMode = types.PrecisionMode;
pub const LearningRateSchedule = types.LearningRateSchedule;
pub const ExperienceType = types.ExperienceType;
pub const DataKind = types.DataKind;
pub const FeedbackType = types.FeedbackType;
pub const TrainingConfig = types.TrainingConfig;
pub const LlmTrainingConfig = types.LlmTrainingConfig;
pub const TrainableModelConfig = types.TrainableModelConfig;
pub const TrainableViTConfig = types.TrainableViTConfig;
pub const CLIPTrainingConfig = types.CLIPTrainingConfig;
pub const SelfLearningConfig = types.SelfLearningConfig;
pub const LoraConfig = types.LoraConfig;
pub const MixedPrecisionConfig = types.MixedPrecisionConfig;
pub const TrainingLogConfig = types.TrainingLogConfig;
pub const TrainingLogMetric = types.TrainingLogMetric;
pub const CheckpointView = types.CheckpointView;
pub const TrainingReport = types.TrainingReport;
pub const Checkpoint = types.Checkpoint;
pub const TokenBlock = types.TokenBlock;
pub const Batch = types.Batch;
pub const GradientAccumulator = types.GradientAccumulator;
pub const InstructionSample = types.InstructionSample;
pub const CheckpointStore = types.CheckpointStore;
pub const TrainingResult = types.TrainingResult;
pub const LlamaTrainer = types.LlamaTrainer;
pub const TrainableModel = types.TrainableModel;
pub const TrainableViTWeights = types.TrainableViTWeights;
pub const TrainableViTModel = types.TrainableViTModel;
pub const TrainableCLIPModel = types.TrainableCLIPModel;
pub const BatchIterator = types.BatchIterator;
pub const TokenizedDataset = types.TokenizedDataset;
pub const DataLoader = types.DataLoader;
pub const SequencePacker = types.SequencePacker;
pub const SelfLearningSystem = types.SelfLearningSystem;
pub const WdbxTokenDataset = types.WdbxTokenDataset;
pub const LoraAdapter = types.LoraAdapter;
pub const LoraLayerAdapters = types.LoraLayerAdapters;
pub const LoraModel = types.LoraModel;
pub const MixedPrecisionContext = types.MixedPrecisionContext;
pub const LossScaler = types.LossScaler;
pub const MasterWeights = types.MasterWeights;
pub const TrainingLogger = types.TrainingLogger;
pub const TrainableWeights = types.TrainableWeights;
pub const TrainableLayerWeights = types.TrainableLayerWeights;
pub const ActivationCache = types.ActivationCache;
pub const TrainableViTLayerWeights = types.TrainableViTLayerWeights;
pub const ViTActivationCache = types.ViTActivationCache;
pub const TextTransformerLayerWeights = types.TextTransformerLayerWeights;
pub const TrainableTextEncoderWeights = types.TrainableTextEncoderWeights;
pub const LlmCheckpoint = types.LlmCheckpoint;
pub const LlmCheckpointView = types.LlmCheckpointView;
pub const CrossEntropyLoss = types.CrossEntropyLoss;
pub const MSELoss = types.MSELoss;
pub const FocalLoss = types.FocalLoss;
pub const LearningExperience = types.LearningExperience;
pub const ExperienceBuffer = types.ExperienceBuffer;
pub const RewardModel = types.RewardModel;
pub const SelfLearningVisionTrainer = types.SelfLearningVisionTrainer;
pub const DocumentTrainer = types.DocumentTrainer;
pub const ModelState = types.ModelState;
pub const SgdOptimizer = types.SgdOptimizer;
pub const AdamOptimizer = types.AdamOptimizer;
pub const AdamWOptimizer = types.AdamWOptimizer;
pub const Optimizer = types.Optimizer;
pub const Context = types.Context;
pub const distributed = types.distributed;
pub const DistributedConfig = types.DistributedConfig;
pub const DistributedTrainer = types.DistributedTrainer;

// ── Submodule re-exports ───────────────────────────────────────────────────

// Sub-namespace facades (additive)
pub const core_training = struct {};
pub const models = struct {};
pub const data = struct {};
pub const checkpointing = struct {};
pub const specialized = struct {};

pub const checkpoint = struct {
    pub const Checkpoint = types.Checkpoint;
    pub const CheckpointStore = types.CheckpointStore;
    pub const CheckpointView = types.CheckpointView;
};
pub const llm_checkpoint = struct {
    pub const LlmCheckpoint = types.LlmCheckpoint;
    pub const LlmCheckpointView = types.LlmCheckpointView;
};
pub const gradient = struct {
    pub const GradientAccumulator = types.GradientAccumulator;
};
pub const training_utils = struct {};

pub const loss = struct {
    pub const CrossEntropyLoss = types.CrossEntropyLoss;
    pub const MSELoss = types.MSELoss;
    pub const FocalLoss = types.FocalLoss;
    pub const perplexity = @import("stub.zig").perplexity;
    pub const klDivergence = @import("stub.zig").klDivergence;
};

pub const trainable_model = struct {
    pub const TrainableModel = types.TrainableModel;
    pub const TrainableModelConfig = types.TrainableModelConfig;
    pub const TrainableWeights = types.TrainableWeights;
    pub const TrainableLayerWeights = types.TrainableLayerWeights;
    pub const ActivationCache = types.ActivationCache;
    pub const LoadError = types.ModelLoadError;
};

pub const llm_trainer = struct {
    pub const TrainingReport = types.TrainingReport;
    pub const TrainingStats = types.TrainingStats;
    pub const LlmTrainingConfig = types.LlmTrainingConfig;
    pub const LlamaTrainer = types.LlamaTrainer;
    pub const trainLlm = @import("stub.zig").trainLlm;
};

pub const data_loader = struct {
    pub const DataLoader = types.DataLoader;
    pub const TokenizedDataset = types.TokenizedDataset;
    pub const Batch = types.Batch;
    pub const BatchIterator = types.BatchIterator;
    pub const SequencePacker = types.SequencePacker;
    pub const InstructionSample = types.InstructionSample;
    pub const parseInstructionDataset = @import("stub.zig").parseInstructionDataset;
};

pub const token_dataset = struct {
    pub const WdbxTokenDataset = types.WdbxTokenDataset;
    pub const TokenBlock = types.TokenBlock;
    pub const encodeTokenBlock = @import("stub.zig").encodeTokenBlock;
    pub const decodeTokenBlock = @import("stub.zig").decodeTokenBlock;
    pub const readTokenBinFile = @import("stub.zig").readTokenBinFile;
    pub const writeTokenBinFile = @import("stub.zig").writeTokenBinFile;
};

pub const lora = struct {
    pub const LoraAdapter = types.LoraAdapter;
    pub const LoraConfig = types.LoraConfig;
    pub const LoraLayerAdapters = types.LoraLayerAdapters;
    pub const LoraModel = types.LoraModel;
};

pub const mixed_precision = struct {
    pub const MixedPrecisionConfig = types.MixedPrecisionConfig;
    pub const MixedPrecisionContext = types.MixedPrecisionContext;
    pub const LossScaler = types.LossScaler;
    pub const MasterWeights = types.MasterWeights;
    pub const fp32ToFp16 = @import("stub.zig").fp32ToFp16;
    pub const fp16ToFp32 = @import("stub.zig").fp16ToFp32;
};

pub const logging = struct {
    pub const TrainingLogger = types.TrainingLogger;
    pub const LoggerConfig = types.TrainingLogConfig;
    pub const Metric = types.TrainingLogMetric;
};

pub const self_learning = struct {
    pub const SelfLearningSystem = types.SelfLearningSystem;
    pub const SelfLearningConfig = types.SelfLearningConfig;
    pub const LearningExperience = types.LearningExperience;
    pub const ExperienceBuffer = types.ExperienceBuffer;
    pub const RewardModel = types.RewardModel;
    pub const VisionTrainer = types.SelfLearningVisionTrainer;
    pub const DocumentTrainer = types.DocumentTrainer;
    pub const ExperienceType = types.ExperienceType;
    pub const FeedbackType = types.FeedbackType;
    pub const DataKind = types.DataKind;
};

pub const vision_trainer = struct {
    pub const TrainableViTModel = types.TrainableViTModel;
    pub const TrainableViTConfig = types.TrainableViTConfig;
    pub const TrainableViTWeights = types.TrainableViTWeights;
    pub const TrainableViTLayerWeights = types.TrainableViTLayerWeights;
    pub const ViTActivationCache = types.ViTActivationCache;
    pub const VisionTrainingError = types.VisionTrainingError;
};

pub const multimodal_trainer = struct {
    pub const TrainableCLIPModel = types.TrainableCLIPModel;
    pub const CLIPTrainingConfig = types.CLIPTrainingConfig;
    pub const TrainableTextEncoderWeights = types.TrainableTextEncoderWeights;
    pub const TextTransformerLayerWeights = types.TextTransformerLayerWeights;
    pub const MultimodalTrainingError = types.MultimodalTrainingError;
};

pub const optimizer_mod = struct {
    pub const Optimizer = types.Optimizer;
    pub const SgdOptimizer = types.SgdOptimizer;
    pub const AdamOptimizer = types.AdamOptimizer;
    pub const AdamWOptimizer = types.AdamWOptimizer;
};

pub const trainer = struct {
    pub const TrainingResult = types.TrainingResult;
    pub const train = @import("stub.zig").train;
    pub const trainAndReport = @import("stub.zig").trainAndReport;
    pub const trainWithResult = @import("stub.zig").trainWithResult;
    pub const calculateLearningRate = @import("stub.zig").calculateLearningRate;
    pub const clipGradients = @import("stub.zig").clipGradients;
    pub const saveModelToWdbx = @import("stub.zig").saveModelToWdbx;
};

// ── Free functions ─────────────────────────────────────────────────────────

pub fn selfLearningConfigFromCore(_: config_module.TrainingConfig) SelfLearningConfig {
    return .{};
}
pub fn train(_: std.mem.Allocator, _: TrainingConfig) Error!void {
    return error.FeatureDisabled;
}
pub fn isEnabled() bool {
    return false;
}
pub fn trainAndReport(_: std.mem.Allocator, _: TrainingConfig) Error!TrainingReport {
    return error.FeatureDisabled;
}
pub fn trainWithResult(_: std.mem.Allocator, _: TrainingConfig) Error!TrainingResult {
    return error.FeatureDisabled;
}
pub fn trainLlm(_: std.mem.Allocator, _: *TrainableModel, _: LlmTrainingConfig, _: []const u32) Error!TrainingReport {
    return error.FeatureDisabled;
}
pub fn encodeTokenBlock(_: std.mem.Allocator, _: []const u32, _: ?[]const u8) Error![]u8 {
    return error.FeatureDisabled;
}
pub fn decodeTokenBlock(_: std.mem.Allocator, _: []const u8) Error!TokenBlock {
    return error.FeatureDisabled;
}
pub fn readTokenBinFile(_: std.mem.Allocator, _: []const u8) Error![]u32 {
    return error.FeatureDisabled;
}
pub fn writeTokenBinFile(_: std.mem.Allocator, _: []const u8, _: []const u32) Error!void {
    return error.FeatureDisabled;
}
pub fn loadCheckpoint(_: std.mem.Allocator, _: []const u8) LoadCheckpointError!Checkpoint {
    return error.FeatureDisabled;
}
pub fn saveCheckpoint(_: std.mem.Allocator, _: []const u8, _: CheckpointView) SaveCheckpointError!void {
    return error.FeatureDisabled;
}
pub fn parseInstructionDataset(_: std.mem.Allocator, _: []const u8) Error!std.ArrayListUnmanaged(InstructionSample) {
    return error.FeatureDisabled;
}
pub fn loadLlmCheckpoint(_: std.mem.Allocator, _: []const u8) LoadLlmCheckpointError!LlmCheckpoint {
    return error.FeatureDisabled;
}
pub fn saveLlmCheckpoint(_: std.mem.Allocator, _: []const u8, _: anytype) SaveLlmCheckpointError!void {
    return error.FeatureDisabled;
}
pub fn fp32ToFp16(_: []const f32, _: []u16) void {}
pub fn fp16ToFp32(_: []const u16, _: []f32) void {}
pub fn perplexity(_: f32) f32 {
    return 0;
}
pub fn klDivergence(_: []const f32, _: []const f32) f32 {
    return 0;
}
pub fn clipGradients(_: []f32, _: f32) f32 {
    return 0;
}
pub fn saveModelToWdbx(_: std.mem.Allocator, _: anytype, _: []const u8) Error!void {
    return error.FeatureDisabled;
}
pub fn calculateLearningRate(_: TrainingConfig, _: u64, _: f32) f32 {
    return 0;
}

test {
    std.testing.refAllDecls(@This());
}
