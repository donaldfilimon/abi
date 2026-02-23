//! Training Stub Module

const std = @import("std");
const config_module = @import("../../../core/config/mod.zig");
const vision = @import("../vision/stub.zig");

// ── Errors ─────────────────────────────────────────────────────────────────

pub const Error = error{ FeatureDisabled, InvalidConfig, CheckpointFailed, TrainingFailed, OutOfMemory };
pub const CheckpointError = error{ InvalidFormat, UnsupportedVersion, PayloadTooLarge };
pub const SaveError = Error || CheckpointError;
pub const LoadError = Error || CheckpointError;
pub const TrainError = Error;
pub const GradientError = error{ NormOverflow, InvalidGradient };
pub const VisionTrainingError = error{ InvalidImageSize, InvalidBatchSize, ConfigMismatch, NoActivationCache, OutOfMemory, FeatureDisabled };
pub const MultimodalTrainingError = error{ InvalidBatchSize, DimensionMismatch, NoActivationCache, OutOfMemory, InvalidTemperature, FeatureDisabled };

// ── Enums ──────────────────────────────────────────────────────────────────

pub const OptimizerType = enum { sgd, adam, adamw };
pub const LearningRateSchedule = enum { constant, linear, cosine, warmup_cosine, step, polynomial, cosine_warm_restarts };
pub const ExperienceType = enum { text_conversation, vision, video, audio, document, code, reasoning, multi_modal, any };
pub const DataKind = enum { text, image, video, audio, document, other };
pub const FeedbackType = enum { positive, negative, implicit_accept, implicit_reject, self_eval, none };

// ── Config structs ─────────────────────────────────────────────────────────

pub const TrainingConfig = struct {
    epochs: u32 = 10,
    batch_size: u32 = 32,
    sample_count: usize = 1024,
    model_size: u32 = 512,
    learning_rate: f32 = 0.001,
    optimizer: OptimizerType = .adamw,
    learning_rate_schedule: LearningRateSchedule = .constant,
    warmup_steps: u32 = 100,
    decay_steps: u32 = 1000,
    min_learning_rate: f32 = 0.0001,
    gradient_accumulation_steps: u32 = 1,
    gradient_clip_norm: f32 = 1.0,
    weight_decay: f32 = 0.01,
    checkpoint_interval: u32 = 0,
    max_checkpoints: u32 = 5,
    checkpoint_path: ?[]const u8 = null,
    early_stopping_patience: u32 = 5,
    early_stopping_threshold: f32 = 1e-4,
    mixed_precision: bool = false,
    pub fn validate(_: TrainingConfig) Error!void {
        return error.FeatureDisabled;
    }
};

pub const LlmTrainingConfig = struct {
    epochs: u32 = 1,
    batch_size: u32 = 4,
    learning_rate: f32 = 1e-5,
    use_gpu: bool = false,
    grad_accum_steps: u32 = 1,
    max_seq_len: u32 = 512,
    warmup_steps: u32 = 100,
    decay_steps: u32 = 1000,
    min_learning_rate: f32 = 1e-6,
    weight_decay: f32 = 0.01,
    max_grad_norm: f32 = 1.0,
    label_smoothing: f32 = 0.0,
    checkpoint_interval: u32 = 0,
    checkpoint_path: ?[]const u8 = null,
    max_checkpoints: u32 = 5,
    optimizer: OptimizerType = .adamw,
    lr_schedule: LearningRateSchedule = .constant,
    mixed_precision: bool = false,
    log_interval: u32 = 10,
    log_dir: ?[]const u8 = null,
    enable_tensorboard: bool = false,
    enable_wandb: bool = false,
    enable_metrics_stream: bool = false,
    export_gguf_path: ?[]const u8 = null,
    export_name: ?[]const u8 = null,
};

pub const TrainableModelConfig = struct {
    hidden_dim: u32 = 0,
    num_layers: u32 = 0,
    num_heads: u32 = 0,
    num_kv_heads: u32 = 0,
    intermediate_dim: u32 = 0,
    vocab_size: u32 = 0,
    max_seq_len: u32 = 2048,
    rope_theta: f32 = 10000.0,
    norm_eps: f32 = 1e-5,
    tie_embeddings: bool = true,
    pub fn headDim(self: TrainableModelConfig) u32 {
        return if (self.num_heads == 0) 0 else self.hidden_dim / self.num_heads;
    }
    pub fn numParams(_: TrainableModelConfig) usize {
        return 0;
    }
};

pub const TrainableViTConfig = struct {
    vit_config: vision.ViTConfig = .{},
    max_batch_size: u32 = 1,
    num_classes: u32 = 1000,
    projection_dim: u32 = 0,
    dropout: f32 = 0.1,
    label_smoothing: f32 = 0.1,
    gradient_checkpointing: bool = false,
    pub fn numParams(_: TrainableViTConfig) usize {
        return 0;
    }
};

pub const CLIPTrainingConfig = struct {
    vision_config: TrainableViTConfig = .{},
    text_hidden_size: u32 = 512,
    text_vocab_size: u32 = 49408,
    text_max_len: u32 = 77,
    text_num_layers: u32 = 12,
    text_num_heads: u32 = 8,
    projection_dim: u32 = 512,
    temperature: f32 = 0.07,
    learnable_temperature: bool = true,
    label_smoothing: f32 = 0.0,
    pub fn numParams(_: CLIPTrainingConfig) usize {
        return 0;
    }
};

pub const SelfLearningConfig = struct {
    enable_rlhf: bool = true,
    enable_vision: bool = true,
    enable_documents: bool = true,
    enable_video: bool = true,
    enable_audio: bool = true,
    enable_all_modalities: bool = true,
    replay_buffer_size: usize = 10000,
    batch_size: u32 = 16,
    learning_rate: f32 = 1e-6,
    gamma: f32 = 0.99,
    ppo_clip: f32 = 0.2,
    value_coef: f32 = 0.5,
    entropy_coef: f32 = 0.01,
    kl_target: f32 = 0.01,
    max_grad_norm: f32 = 0.5,
    ppo_epochs: u32 = 4,
    min_buffer_size: usize = 100,
    update_frequency: usize = 64,
    reward_shaping: bool = true,
    self_eval_threshold: f32 = 0.7,
    checkpoint_interval: u32 = 100,
    continuous_learning: bool = true,
};

// ── Data structs ───────────────────────────────────────────────────────────

pub const CheckpointView = struct { step: u64, timestamp: u64, weights: []const f32 };

pub const TrainingReport = struct {
    epochs: u32 = 0,
    batches: u32 = 0,
    final_loss: f32 = 0,
    final_accuracy: f32 = 0,
    best_loss: f32 = 0,
    learning_rate: f32 = 0,
    gradient_updates: u64 = 0,
    checkpoints_saved: u32 = 0,
    early_stopped: bool = false,
    total_time_ms: u64 = 0,
    best_val_loss: f32 = 0,
    best_val_accuracy: f32 = 0,
    final_perplexity: f32 = 0,
    total_steps: u64 = 0,
    total_tokens: u64 = 0,
    total_time_ns: u64 = 0,
    avg_throughput: f32 = 0,
};

pub const Checkpoint = struct {
    step: u64 = 0,
    timestamp: u64 = 0,
    weights: []f32 = &.{},
    pub fn deinit(self: *Checkpoint, allocator: std.mem.Allocator) void {
        if (self.weights.len > 0) allocator.free(self.weights);
        self.* = undefined;
    }
};

pub const StepMetrics = struct { loss: f32 = 0.0, accuracy: f32 = 0.0, learning_rate: f32 = 0.0, grad_norm: f32 = 0.0, step_time_ms: u64 = 0 };

pub const TrainingStats = struct {
    epoch: u32 = 0,
    global_step: u64 = 0,
    micro_step: u32 = 0,
    tokens_processed: u64 = 0,
    loss: f32 = 0,
    accuracy: f32 = 0,
    learning_rate: f32 = 0,
    gradient_norm: f32 = 0,
    best_val_loss: f32 = 0,
    best_val_accuracy: f32 = 0,
    training_time_ns: u64 = 0,
    checkpoints_saved: u32 = 0,
};

pub const LearningStats = struct {
    total_experiences: u64 = 0,
    total_updates: u64 = 0,
    avg_reward: f32 = 0,
    avg_loss: f32 = 0,
    positive_feedback_count: u64 = 0,
    negative_feedback_count: u64 = 0,
    vision_samples: u64 = 0,
    video_samples: u64 = 0,
    audio_samples: u64 = 0,
    document_samples: u64 = 0,
    any_samples: u64 = 0,
    improvement_rate: f32 = 0,
};

pub const TokenBlock = struct {
    allocator: std.mem.Allocator,
    tokens: []u32 = &.{},
    text: ?[]u8 = null,
    pub fn deinit(_: *TokenBlock) void {}
};

pub const Batch = struct { input_ids: []const u32 = &.{}, labels: []const u32 = &.{}, attention_mask: ?[]const u8 = null, batch_size: u32 = 0, seq_len: u32 = 0 };
pub const GradientAccumulator = struct {};
pub const InstructionSample = struct { instruction: []const u8 = "", input: ?[]const u8 = null, output: []const u8 = "" };

// ── Stub type impls ────────────────────────────────────────────────────────

pub const CheckpointStore = struct {
    pub fn init(_: std.mem.Allocator, _: u32) CheckpointStore {
        return .{};
    }
    pub fn deinit(_: *CheckpointStore) void {}
    pub fn count(_: *const CheckpointStore) usize {
        return 0;
    }
    pub fn add(_: *CheckpointStore, _: u64, _: []const f32) !void {}
    pub fn latest(_: *const CheckpointStore) ?Checkpoint {
        return null;
    }
};

pub const TrainingResult = struct {
    report: TrainingReport = .{},
    checkpoints: CheckpointStore = .{},
    loss_history: []f32 = &.{},
    accuracy_history: []f32 = &.{},
    pub fn deinit(_: *TrainingResult) void {}
};

pub const GpuTrainingStats = struct {
    total_gpu_ops: u64 = 0,
    gpu_time_ns: u64 = 0,
    cpu_fallback_ops: u64 = 0,
    utilization: f32 = 0,
    backend_name: []const u8 = "none",
    gpu_available: bool = false,
    pub fn avgKernelTimeMs(_: GpuTrainingStats) f32 {
        return 0;
    }
    pub fn gpuRatio(_: GpuTrainingStats) f32 {
        return 0;
    }
};

pub const LlamaTrainer = struct {
    pub fn init(_: std.mem.Allocator, _: *TrainableModel, _: LlmTrainingConfig) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
    pub fn trainStepWithMetrics(_: *@This(), _: []const u32, _: []const u32) Error!StepMetrics {
        return error.FeatureDisabled;
    }
    pub fn saveCheckpoint(_: *@This(), _: []const u8) Error!void {
        return error.FeatureDisabled;
    }
    pub fn getStats(_: *const @This()) TrainingStats {
        return .{};
    }
    pub fn getGpuStats(_: *const @This()) GpuTrainingStats {
        return .{};
    }
    pub fn getReport(_: *const @This()) TrainingReport {
        return .{};
    }
};

pub const TrainableModel = struct {
    config: TrainableModelConfig = .{},
    pub fn init(_: std.mem.Allocator, _: anytype) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn fromGguf(_: std.mem.Allocator, _: []const u8) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
    pub fn numParams(_: @This()) u64 {
        return 0;
    }
    pub fn exportToGguf(_: *const @This(), _: std.mem.Allocator, _: []const u8, _: anytype) Error!void {
        return error.FeatureDisabled;
    }
};

pub const TrainableViTWeights = struct {
    allocator: std.mem.Allocator,
    config: TrainableViTConfig,
    pub fn init(_: std.mem.Allocator, _: TrainableViTConfig) VisionTrainingError!TrainableViTWeights {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *TrainableViTWeights) void {}
    pub fn zeroGradients(_: *TrainableViTWeights) void {}
};

pub const TrainableViTModel = struct {
    allocator: std.mem.Allocator,
    config: TrainableViTConfig,
    pub fn init(_: std.mem.Allocator, _: TrainableViTConfig) VisionTrainingError!TrainableViTModel {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *TrainableViTModel) void {}
    pub fn forward(_: *TrainableViTModel, _: []const f32, _: u32, _: []f32) VisionTrainingError!void {
        return error.FeatureDisabled;
    }
    pub fn backward(_: *TrainableViTModel, _: []const f32, _: u32) VisionTrainingError!void {
        return error.FeatureDisabled;
    }
    pub fn getGradients(_: *const TrainableViTModel) ?*anyopaque {
        return null;
    }
    pub fn applyGradients(_: *TrainableViTModel, _: f32) VisionTrainingError!void {
        return error.FeatureDisabled;
    }
    pub fn zeroGradients(_: *TrainableViTModel) void {}
    pub fn computeGradientNorm(_: *const TrainableViTModel) f32 {
        return 0.0;
    }
    pub fn clipGradients(_: *TrainableViTModel, _: f32) f32 {
        return 0.0;
    }
    pub fn applySgdUpdate(_: *TrainableViTModel, _: f32) void {}
};

pub const TrainableCLIPModel = struct {
    allocator: std.mem.Allocator,
    config: CLIPTrainingConfig,
    pub fn init(_: std.mem.Allocator, _: CLIPTrainingConfig) MultimodalTrainingError!TrainableCLIPModel {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *TrainableCLIPModel) void {}
    pub fn encodeImages(_: *TrainableCLIPModel, _: []const f32, _: u32, _: []f32) MultimodalTrainingError!void {
        return error.FeatureDisabled;
    }
    pub fn encodeText(_: *TrainableCLIPModel, _: []const u32, _: u32, _: []f32) MultimodalTrainingError!void {
        return error.FeatureDisabled;
    }
    pub fn computeContrastiveLoss(_: *TrainableCLIPModel, _: []const f32, _: []const f32, _: u32, _: []f32, _: []f32) f32 {
        return 0.0;
    }
    pub fn zeroGradients(_: *TrainableCLIPModel) void {}
    pub fn computeGradientNorm(_: *const TrainableCLIPModel) f32 {
        return 0.0;
    }
    pub fn applySgdUpdate(_: *TrainableCLIPModel, _: f32) void {}
    pub fn getTemperature(_: *const TrainableCLIPModel) f32 {
        return 0.07;
    }
};

pub const BatchIterator = struct {
    pub fn init(_: std.mem.Allocator, _: *const TokenizedDataset, _: u32, _: u32, _: bool) Error!BatchIterator {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *BatchIterator) void {}
    pub fn next(_: *BatchIterator) ?Batch {
        return null;
    }
    pub fn reset(_: *BatchIterator) void {}
    pub fn numBatches(_: *const BatchIterator) usize {
        return 0;
    }
};

pub const TokenizedDataset = struct {
    allocator: std.mem.Allocator = undefined,
    data: []const u32 = &.{},
    owns_data: bool = false,
    pub fn load(_: std.mem.Allocator, _: []const u8) Error!TokenizedDataset {
        return error.FeatureDisabled;
    }
    pub fn fromSlice(allocator: std.mem.Allocator, data: []const u32) TokenizedDataset {
        return .{ .allocator = allocator, .data = data, .owns_data = false };
    }
    pub fn deinit(_: *TokenizedDataset) void {}
    pub fn len(self: *const TokenizedDataset) usize {
        return self.data.len;
    }
    pub fn numBatches(_: *const TokenizedDataset, _: u32, _: u32) usize {
        return 0;
    }
    pub fn batches(_: *const TokenizedDataset, allocator: std.mem.Allocator, batch_size: u32, seq_len: u32, shuffle: bool) Error!BatchIterator {
        return BatchIterator.init(allocator, undefined, batch_size, seq_len, shuffle);
    }
};

pub const DataLoader = struct {
    allocator: std.mem.Allocator,
    dataset: TokenizedDataset,
    batch_size: u32,
    seq_len: u32,
    shuffle: bool,
    drop_last: bool,
    pub const Config = struct { batch_size: u32 = 4, seq_len: u32 = 512, shuffle: bool = true, drop_last: bool = true };
    pub fn init(allocator: std.mem.Allocator, dataset: TokenizedDataset, cfg: Config) DataLoader {
        return .{ .allocator = allocator, .dataset = dataset, .batch_size = cfg.batch_size, .seq_len = cfg.seq_len, .shuffle = cfg.shuffle, .drop_last = cfg.drop_last };
    }
    pub fn deinit(_: *DataLoader) void {}
    pub fn iterator(_: *const DataLoader) Error!BatchIterator {
        return error.FeatureDisabled;
    }
    pub fn numBatches(_: *const DataLoader) usize {
        return 0;
    }
    pub fn numTokens(_: *const DataLoader) usize {
        return 0;
    }
};

pub const SequencePacker = struct {
    allocator: std.mem.Allocator,
    max_seq_len: u32,
    pad_token_id: u32,
    pub fn init(allocator: std.mem.Allocator, max_seq_len: u32, pad_token_id: u32) SequencePacker {
        return .{ .allocator = allocator, .max_seq_len = max_seq_len, .pad_token_id = pad_token_id };
    }
    pub fn deinit(_: *SequencePacker) void {}
    pub fn addSequence(_: *SequencePacker, _: []const u32) Error!void {
        return error.FeatureDisabled;
    }
    pub fn pack(_: *SequencePacker, _: u32) Error!PackedBatch {
        return error.FeatureDisabled;
    }
    pub const PackedBatch = struct {
        allocator: std.mem.Allocator,
        tokens: []u32 = @constCast(&[_]u32{}),
        attention_mask: []u8 = @constCast(&[_]u8{}),
        batch_size: u32 = 0,
        seq_len: u32 = 0,
        num_batches: u32 = 0,
        pub fn deinit(_: *PackedBatch) void {}
        pub fn getBatch(_: *const PackedBatch, _: u32) Batch {
            return .{};
        }
    };
};

pub const SelfLearningSystem = struct {
    pub fn init(_: std.mem.Allocator, _: SelfLearningConfig) Error!SelfLearningSystem {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *SelfLearningSystem) void {}
    pub fn recordExperience(_: *SelfLearningSystem, _: []const u32, _: []const u32, _: FeedbackType, _: f32, _: ExperienceType) Error!void {
        return error.FeatureDisabled;
    }
    pub fn recordVisionExperience(_: *SelfLearningSystem, _: []const u32, _: []const u32, _: []const u8, _: FeedbackType, _: f32) Error!void {
        return error.FeatureDisabled;
    }
    pub fn recordVideoExperience(_: *SelfLearningSystem, _: []const u32, _: []const u32, _: []const u8, _: FeedbackType, _: f32) Error!void {
        return error.FeatureDisabled;
    }
    pub fn recordAudioExperience(_: *SelfLearningSystem, _: []const u32, _: []const u32, _: []const u8, _: FeedbackType, _: f32) Error!void {
        return error.FeatureDisabled;
    }
    pub fn recordGenericExperience(_: *SelfLearningSystem, _: []const u32, _: []const u32, _: []const u8, _: ?[]const u8, _: FeedbackType, _: f32) Error!void {
        return error.FeatureDisabled;
    }
    pub fn update(_: *SelfLearningSystem) Error!void {
        return error.FeatureDisabled;
    }
    pub fn getStats(_: *const SelfLearningSystem) LearningStats {
        return .{};
    }
};

pub const WdbxTokenDataset = struct {
    pub fn init(_: std.mem.Allocator, _: []const u8) Error!WdbxTokenDataset {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *WdbxTokenDataset) void {}
    pub fn save(_: *WdbxTokenDataset) Error!void {
        return error.FeatureDisabled;
    }
    pub fn appendTokens(_: *WdbxTokenDataset, _: []const u32, _: ?[]const u8) Error!void {
        return error.FeatureDisabled;
    }
    pub fn importTokenBin(_: *WdbxTokenDataset, _: []const u32, _: u32) Error!void {
        return error.FeatureDisabled;
    }
    pub fn collectTokens(_: *WdbxTokenDataset, _: std.mem.Allocator, _: usize) Error![]u32 {
        return error.FeatureDisabled;
    }
    pub fn exportTokenBinFile(_: *WdbxTokenDataset, _: std.mem.Allocator, _: []const u8, _: usize) Error!void {
        return error.FeatureDisabled;
    }
    pub fn ingestText(_: *WdbxTokenDataset, _: std.mem.Allocator, _: anytype, _: []const u8, _: u32) Error!void {
        return error.FeatureDisabled;
    }
};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.TrainingConfig) Error!*Context {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *Context) void {}
    pub fn train(_: *Context, _: TrainingConfig) Error!TrainingResult {
        return error.FeatureDisabled;
    }
    pub fn getCheckpointStore(_: *Context) Error!*CheckpointStore {
        return error.FeatureDisabled;
    }
    pub fn saveCheckpoint(_: *Context, _: []const u8, _: anytype) Error!void {
        return error.FeatureDisabled;
    }
    pub fn loadCheckpointData(_: *Context, _: []const u8, comptime T: type) Error!T {
        return error.FeatureDisabled;
    }
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
pub fn loadCheckpoint(_: std.mem.Allocator, _: []const u8) LoadError!Checkpoint {
    return error.FeatureDisabled;
}
pub fn saveCheckpoint(_: std.mem.Allocator, _: []const u8, _: CheckpointView) SaveError!void {
    return error.FeatureDisabled;
}
pub fn parseInstructionDataset(_: std.mem.Allocator, _: []const u8) Error!std.ArrayListUnmanaged(InstructionSample) {
    return error.FeatureDisabled;
}

// ── Distributed training stub ──────────────────────────────────────────────

pub const distributed = struct {
    pub const ReduceOp = enum { sum, average };

    pub const DistributedConfig = struct {
        world_size: u32 = 1,
        rank: u32 = 0,
        is_coordinator: bool = true,
        bucket_size_bytes: usize = 25 * 1024 * 1024,
        enable_compression: bool = false,
        reduce_op: ReduceOp = .average,
        pub fn validate(_: @This()) error{InvalidConfig}!void {}
    };

    const Self = @This();

    pub const DistributedTrainer = struct {
        pub const Stats = struct {
            total_allreduce_calls: u64 = 0,
            total_bytes_synced: u64 = 0,
            total_sync_time_ns: u64 = 0,
            epochs_completed: u32 = 0,
        };
        pub fn init(_: std.mem.Allocator, _: Self.DistributedConfig) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
        pub fn synchronizeGradients(_: *@This(), _: []f32) void {}
        pub fn shardData(_: *const @This(), comptime T: type, data: []const T) []const T {
            return data;
        }
        pub fn shouldLog(_: *const @This()) bool {
            return false;
        }
        pub fn recordEpoch(_: *@This()) void {}
        pub fn getStats(_: *const @This()) Stats {
            return .{};
        }
    };
};

pub const DistributedConfig = distributed.DistributedConfig;
pub const DistributedTrainer = distributed.DistributedTrainer;

// ── Missing type stubs for mod.zig parity ──────────────────────────────────

pub const LoraConfig = struct { rank: u32 = 8, alpha: f32 = 16.0, dropout: f32 = 0.1, target_modules: []const []const u8 = &.{} };
pub const LoraAdapter = struct {
    pub fn init(_: std.mem.Allocator, _: LoraConfig) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};
pub const LoraLayerAdapters = struct {};
pub const LoraModel = struct {
    pub fn init(_: std.mem.Allocator, _: anytype, _: LoraConfig) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};

pub const MixedPrecisionConfig = struct { enabled: bool = false, loss_scale: f32 = 1.0 };
pub const MixedPrecisionContext = struct {
    pub fn init(_: std.mem.Allocator, _: MixedPrecisionConfig) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};
pub const LossScaler = struct {};
pub const MasterWeights = struct {};
pub fn fp32ToFp16(_: []const f32, _: []u16) void {}
pub fn fp16ToFp32(_: []const u16, _: []f32) void {}

pub const TrainingLogger = struct {
    pub fn init(_: std.mem.Allocator, _: TrainingLogConfig) Error!@This() {
        return error.FeatureDisabled;
    }
    pub fn deinit(_: *@This()) void {}
};
pub const TrainingLogConfig = struct { log_interval: u32 = 10, log_dir: ?[]const u8 = null };
pub const TrainingLogMetric = struct { name: []const u8 = "", value: f32 = 0 };

pub const TrainableWeights = struct {};
pub const TrainableLayerWeights = struct {};
pub const ActivationCache = struct {};
pub const ModelLoadError = error{InvalidFormat};
pub const TrainableViTLayerWeights = struct {};
pub const ViTActivationCache = struct {};
pub const TextTransformerLayerWeights = struct {};
pub const TrainableTextEncoderWeights = struct {};

pub const SaveCheckpointError = SaveError;
pub const LoadCheckpointError = LoadError;
pub const SaveLatestCheckpointError = SaveError;
pub const LlmCheckpoint = struct {};
pub const LlmCheckpointView = struct {};
pub const LoadLlmCheckpointError = LoadError;
pub const SaveLlmCheckpointError = SaveError;
pub fn loadLlmCheckpoint(_: std.mem.Allocator, _: []const u8) LoadError!LlmCheckpoint {
    return error.FeatureDisabled;
}
pub fn saveLlmCheckpoint(_: std.mem.Allocator, _: []const u8, _: anytype) SaveError!void {
    return error.FeatureDisabled;
}
pub const GradientAccumulatorFull = struct {};

pub const CrossEntropyLoss = struct {};
pub const MSELoss = struct {};
pub const FocalLoss = struct {};
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

// ── Submodule re-exports ───────────────────────────────────────────────────

const stub_root = @This();
pub const llm_trainer = struct {
    pub const TrainingReport = stub_root.TrainingReport;
    pub const TrainingStats = stub_root.TrainingStats;
    pub const LlmTrainingConfig = stub_root.LlmTrainingConfig;
    pub const LlamaTrainer = stub_root.LlamaTrainer;
    pub const trainLlm = stub_root.trainLlm;
};
pub const trainable_model = struct {
    pub const TrainableModel = stub_root.TrainableModel;
    pub const TrainableModelConfig = stub_root.TrainableModelConfig;
};

test {
    std.testing.refAllDecls(@This());
}
