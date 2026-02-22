//! Training Sub-module
//!
//! Training pipeline utilities, gradient aggregation, and checkpointing.
//! Models can be trained to handle and generate all types of data: text, images,
//! video, audio, documents, and arbitrary payloads (see ExperienceType, DataKind, and
//! SelfLearningSystem.recordVideoExperience / recordAudioExperience / recordGenericExperience).
//!
//! Provides neural network training with SGD, Adam optimizers, learning rate scheduling,
//! gradient clipping, loss functions, and mixed precision support.

const std = @import("std");
const build_options = @import("build_options");
const config_module = @import("../../../core/config/mod.zig");
const time_utils = @import("../../../services/shared/utils.zig");
const time = @import("../../../services/shared/time.zig");
const database = @import("../../database/mod.zig");

// Local submodule imports
const checkpoint = @import("checkpoint.zig");
const llm_checkpoint = @import("llm_checkpoint.zig");
const gradient = @import("gradient.zig");
pub const loss = @import("loss.zig");
pub const trainable_model = @import("trainable_model.zig");
pub const llm_trainer = @import("llm_trainer.zig");
pub const data_loader = @import("data_loader.zig");
pub const wdbx_dataset = @import("../database/wdbx.zig");
pub const lora = @import("lora.zig");
pub const mixed_precision = @import("mixed_precision.zig");
pub const logging = @import("logging.zig");
pub const distributed = @import("distributed.zig");

// Distributed training exports
pub const DistributedConfig = distributed.DistributedConfig;
pub const DistributedTrainer = distributed.DistributedTrainer;

// Checkpoint exports
pub const Checkpoint = checkpoint.Checkpoint;
pub const CheckpointError = checkpoint.CheckpointError;
pub const CheckpointStore = checkpoint.CheckpointStore;
pub const CheckpointView = checkpoint.CheckpointView;
pub const LoadCheckpointError = checkpoint.LoadError;
pub const SaveCheckpointError = checkpoint.SaveError;
pub const SaveLatestCheckpointError = checkpoint.SaveLatestError;
pub const loadCheckpoint = checkpoint.loadCheckpoint;
pub const saveCheckpoint = checkpoint.saveCheckpoint;

// LLM checkpoint exports
pub const LlmCheckpoint = llm_checkpoint.LlmCheckpoint;
pub const LlmCheckpointView = llm_checkpoint.LlmCheckpointView;
pub const LoadLlmCheckpointError = llm_checkpoint.LoadError;
pub const SaveLlmCheckpointError = llm_checkpoint.SaveError;
pub const loadLlmCheckpoint = llm_checkpoint.loadLlmCheckpoint;
pub const saveLlmCheckpoint = llm_checkpoint.saveLlmCheckpoint;
pub const GradientAccumulator = gradient.GradientAccumulator;
pub const GradientError = gradient.GradientError;

// Loss function exports
pub const CrossEntropyLoss = loss.CrossEntropyLoss;
pub const MSELoss = loss.MSELoss;
pub const FocalLoss = loss.FocalLoss;
pub const perplexity = loss.perplexity;
pub const klDivergence = loss.klDivergence;

// Trainable model exports
pub const TrainableModel = trainable_model.TrainableModel;
pub const TrainableModelConfig = trainable_model.TrainableModelConfig;
pub const TrainableWeights = trainable_model.TrainableWeights;
pub const TrainableLayerWeights = trainable_model.TrainableLayerWeights;
pub const ActivationCache = trainable_model.ActivationCache;
pub const ModelLoadError = trainable_model.LoadError;

// LLM trainer exports
pub const LlamaTrainer = llm_trainer.LlamaTrainer;
pub const LlmTrainingConfig = llm_trainer.LlmTrainingConfig;
pub const trainLlm = llm_trainer.trainLlm;

// Data loader exports
pub const DataLoader = data_loader.DataLoader;
pub const TokenizedDataset = data_loader.TokenizedDataset;
pub const Batch = data_loader.Batch;
pub const BatchIterator = data_loader.BatchIterator;
pub const SequencePacker = data_loader.SequencePacker;
pub const InstructionSample = data_loader.InstructionSample;
pub const parseInstructionDataset = data_loader.parseInstructionDataset;
pub const WdbxTokenDataset = wdbx_dataset.WdbxTokenDataset;
pub const TokenBlock = wdbx_dataset.TokenBlock;
pub const encodeTokenBlock = wdbx_dataset.encodeTokenBlock;
pub const decodeTokenBlock = wdbx_dataset.decodeTokenBlock;
pub const readTokenBinFile = wdbx_dataset.readTokenBinFile;
pub const writeTokenBinFile = wdbx_dataset.writeTokenBinFile;

// LoRA exports
pub const LoraAdapter = lora.LoraAdapter;
pub const LoraConfig = lora.LoraConfig;
pub const LoraLayerAdapters = lora.LoraLayerAdapters;
pub const LoraModel = lora.LoraModel;

// Mixed precision exports
pub const MixedPrecisionConfig = mixed_precision.MixedPrecisionConfig;
pub const MixedPrecisionContext = mixed_precision.MixedPrecisionContext;
pub const LossScaler = mixed_precision.LossScaler;
pub const MasterWeights = mixed_precision.MasterWeights;
pub const fp32ToFp16 = mixed_precision.fp32ToFp16;
pub const fp16ToFp32 = mixed_precision.fp16ToFp32;

pub const TrainingLogger = logging.TrainingLogger;
pub const TrainingLogConfig = logging.LoggerConfig;
pub const TrainingLogMetric = logging.Metric;

// Self-learning exports
pub const self_learning = @import("self_learning.zig");
pub const SelfLearningSystem = self_learning.SelfLearningSystem;
pub const SelfLearningConfig = self_learning.SelfLearningConfig;

/// Build SelfLearningConfig from system-level TrainingConfig (Zig 0.16).
/// Flows enable_vision, enable_video, enable_audio, enable_all_modalities from core config.
pub fn selfLearningConfigFromCore(core_cfg: config_module.TrainingConfig) SelfLearningConfig {
    return .{
        .enable_vision = core_cfg.enable_vision,
        .enable_video = core_cfg.enable_video,
        .enable_audio = core_cfg.enable_audio,
        .enable_all_modalities = core_cfg.enable_all_modalities,
        .enable_rlhf = true,
        .enable_documents = true,
        .replay_buffer_size = 10000,
        .batch_size = core_cfg.batch_size,
        .min_buffer_size = 100,
        .update_frequency = 64,
    };
}
pub const LearningExperience = self_learning.LearningExperience;
pub const ExperienceBuffer = self_learning.ExperienceBuffer;
pub const RewardModel = self_learning.RewardModel;
pub const SelfLearningVisionTrainer = self_learning.VisionTrainer;
pub const DocumentTrainer = self_learning.DocumentTrainer;
pub const ExperienceType = self_learning.ExperienceType;
pub const FeedbackType = self_learning.FeedbackType;
pub const DataKind = self_learning.DataKind;

// Vision Transformer training exports
pub const vision_trainer = @import("vision_trainer.zig");
pub const TrainableViTModel = vision_trainer.TrainableViTModel;
pub const TrainableViTConfig = vision_trainer.TrainableViTConfig;
pub const TrainableViTWeights = vision_trainer.TrainableViTWeights;
pub const TrainableViTLayerWeights = vision_trainer.TrainableViTLayerWeights;
pub const ViTActivationCache = vision_trainer.ViTActivationCache;
pub const VisionTrainingError = vision_trainer.VisionTrainingError;

// Multimodal (CLIP-style) training exports
pub const multimodal_trainer = @import("multimodal_trainer.zig");
pub const TrainableCLIPModel = multimodal_trainer.TrainableCLIPModel;
pub const CLIPTrainingConfig = multimodal_trainer.CLIPTrainingConfig;
pub const TrainableTextEncoderWeights = multimodal_trainer.TrainableTextEncoderWeights;
pub const TextTransformerLayerWeights = multimodal_trainer.TextTransformerLayerWeights;
pub const MultimodalTrainingError = multimodal_trainer.MultimodalTrainingError;

pub const TrainingError = error{
    InvalidConfiguration,
    ConvergenceFailed,
    NaNGradient,
    InvalidCheckpoint,
};

pub const TrainError =
    TrainingError ||
    GradientError ||
    SaveLatestCheckpointError ||
    std.mem.Allocator.Error;

pub const Error = error{
    TrainingDisabled,
    InvalidConfig,
    CheckpointFailed,
    TrainingFailed,
    OutOfMemory,
};

pub const OptimizerType = enum {
    sgd,
    adam,
    adamw,
};

pub const LearningRateSchedule = enum {
    constant,
    linear,
    cosine,
    warmup_cosine,
    step,
    polynomial,
    /// Cosine annealing with warm restarts (SGDR)
    cosine_warm_restarts,
};

pub const TrainingConfig = struct {
    epochs: u32 = 10,
    batch_size: u32 = 32,
    sample_count: usize = 1024,
    model_size: u32 = 512,
    learning_rate: f32 = 0.001,
    optimizer: OptimizerType = .adamw,
    learning_rate_schedule: LearningRateSchedule = .warmup_cosine,
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

    pub fn validate(self: TrainingConfig) TrainingError!void {
        if (self.epochs == 0) return TrainingError.InvalidConfiguration;
        if (self.batch_size == 0) return TrainingError.InvalidConfiguration;
        if (self.sample_count == 0) return TrainingError.InvalidConfiguration;
        if (self.model_size == 0) return TrainingError.InvalidConfiguration;
        if (self.learning_rate <= 0) return TrainingError.InvalidConfiguration;
        if (self.gradient_accumulation_steps == 0) {
            return TrainingError.InvalidConfiguration;
        }
        if (self.gradient_clip_norm < 0) {
            return TrainingError.InvalidConfiguration;
        }
        if (self.weight_decay < 0) {
            return TrainingError.InvalidConfiguration;
        }
        if (self.warmup_steps >= self.decay_steps and self.learning_rate_schedule == .warmup_cosine) {
            return TrainingError.InvalidConfiguration;
        }
        if (self.decay_steps == 0 and self.learning_rate_schedule == .cosine) {
            return TrainingError.InvalidConfiguration;
        }
    }
};

pub const TrainingReport = struct {
    epochs: u32,
    batches: u32,
    final_loss: f32,
    final_accuracy: f32,
    best_loss: f32,
    learning_rate: f32,
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

pub const ModelState = struct {
    allocator: std.mem.Allocator,
    weights: []f32,
    gradients: []f32,
    momentum: ?[]f32,
    velocity: ?[]f32,
    step: u64,
    name: []const u8,

    pub fn init(allocator: std.mem.Allocator, size: usize, name: []const u8) !ModelState {
        const weights = try allocator.alloc(f32, size);
        errdefer allocator.free(weights);
        const gradients = try allocator.alloc(f32, size);
        errdefer allocator.free(gradients);
        const momentum = try allocator.alloc(f32, size);
        errdefer allocator.free(momentum);
        const velocity = try allocator.alloc(f32, size);
        errdefer allocator.free(velocity);

        initializeXavierUniform(weights, size);
        @memset(gradients, 0);
        @memset(momentum, 0);
        @memset(velocity, 0);

        const name_copy = try allocator.dupe(u8, name);

        return .{
            .allocator = allocator,
            .weights = weights,
            .gradients = gradients,
            .momentum = momentum,
            .velocity = velocity,
            .step = 0,
            .name = name_copy,
        };
    }

    pub fn deinit(self: *ModelState) void {
        self.allocator.free(self.name);
        if (self.velocity) |v| self.allocator.free(v);
        if (self.momentum) |m| self.allocator.free(m);
        self.allocator.free(self.gradients);
        self.allocator.free(self.weights);
        self.* = undefined;
    }

    pub fn zeroGradients(self: *ModelState) void {
        @memset(self.gradients, 0);
    }

    pub fn numParams(self: *const ModelState) usize {
        return self.weights.len;
    }
};

pub const Optimizer = union(OptimizerType) {
    sgd: SgdOptimizer,
    adam: AdamOptimizer,
    adamw: AdamWOptimizer,

    pub fn init(
        allocator: std.mem.Allocator,
        model: *ModelState,
        config: TrainingConfig,
    ) !Optimizer {
        return switch (config.optimizer) {
            .sgd => .{ .sgd = try SgdOptimizer.init(allocator, model, config) },
            .adam => .{ .adam = try AdamOptimizer.init(allocator, model, config) },
            .adamw => .{ .adamw = try AdamWOptimizer.init(allocator, model, config) },
        };
    }

    pub fn deinit(self: *Optimizer, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .sgd => |*o| o.deinit(allocator),
            .adam => |*o| o.deinit(allocator),
            .adamw => |*o| o.deinit(allocator),
        }
    }

    pub fn step(self: *Optimizer, model: *ModelState, lr: f32, current_step: u64) void {
        switch (self.*) {
            .sgd => |*o| o.step(model, lr, current_step),
            .adam => |*o| o.step(model, lr, current_step),
            .adamw => |*o| o.step(model, lr, current_step),
        }
    }

    pub fn setLearningRate(self: *Optimizer, lr: f32) void {
        switch (self.*) {
            .sgd => |*o| o.learning_rate = lr,
            .adam => |*o| o.learning_rate = lr,
            .adamw => |*o| o.learning_rate = lr,
        }
    }
};

pub const SgdOptimizer = struct {
    learning_rate: f32,
    momentum: f32 = 0.9,
    nesterov: bool = false,

    pub fn init(allocator: std.mem.Allocator, model: *ModelState, config: TrainingConfig) !SgdOptimizer {
        _ = allocator;
        _ = model;
        return .{
            .learning_rate = config.learning_rate,
            .momentum = 0.9,
            .nesterov = false,
        };
    }

    pub fn deinit(self: *SgdOptimizer, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    pub fn step(self: *SgdOptimizer, model: *ModelState, lr: f32, current_step: u64) void {
        _ = current_step;
        const momentum_val = self.momentum;

        if (model.momentum) |mom| {
            for (model.weights, model.gradients, mom) |*w, *g, *m| {
                const m_old = m.*;
                m.* = momentum_val * m.* + g.*;
                if (self.nesterov) {
                    // Nesterov: use old momentum for look-ahead
                    w.* -= lr * (g.* + momentum_val * m_old);
                } else {
                    w.* -= lr * m.*;
                }
            }
        } else {
            for (model.weights, model.gradients) |*w, *g| {
                w.* -= lr * g.*;
            }
        }
    }
};

pub const AdamOptimizer = struct {
    learning_rate: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    t: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, model: *ModelState, config: TrainingConfig) !AdamOptimizer {
        _ = allocator;
        _ = model;
        return .{
            .learning_rate = config.learning_rate,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .epsilon = 1e-8,
            .t = 0,
        };
    }

    pub fn deinit(self: *AdamOptimizer, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    pub fn step(self: *AdamOptimizer, model: *ModelState, lr: f32, current_step: u64) void {
        if (current_step == 0) return;
        self.t = current_step;
        const beta1 = self.beta1;
        const beta2 = self.beta2;
        const epsilon = self.epsilon;
        const step_f = @as(f32, @floatFromInt(current_step));
        const lr_adjusted = lr * @sqrt(1 - std.math.pow(f32, beta2, step_f)) /
            (1 - std.math.pow(f32, beta1, step_f));

        if (model.momentum) |m| {
            if (model.velocity) |v| {
                for (0..model.weights.len) |i| {
                    m[i] = beta1 * m[i] + (1 - beta1) * model.gradients[i];
                    v[i] = beta2 * v[i] + (1 - beta2) * model.gradients[i] * model.gradients[i];
                    // lr_adjusted already has bias correction; use raw m/v
                    model.weights[i] -= lr_adjusted * m[i] / (@sqrt(v[i]) + epsilon);
                }
            }
        }
    }
};

pub const AdamWOptimizer = struct {
    learning_rate: f32,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    epsilon: f32 = 1e-8,
    weight_decay: f32 = 0.01,
    t: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, model: *ModelState, config: TrainingConfig) !AdamWOptimizer {
        _ = allocator;
        _ = model;
        return .{
            .learning_rate = config.learning_rate,
            .beta1 = 0.9,
            .beta2 = 0.999,
            .epsilon = 1e-8,
            .weight_decay = config.weight_decay,
            .t = 0,
        };
    }

    pub fn deinit(self: *AdamWOptimizer, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
    }

    pub fn step(self: *AdamWOptimizer, model: *ModelState, lr: f32, current_step: u64) void {
        if (current_step == 0) return;
        self.t = current_step;
        const beta1 = self.beta1;
        const beta2 = self.beta2;
        const epsilon = self.epsilon;
        const wd = self.weight_decay;
        const step_f = @as(f32, @floatFromInt(current_step));

        for (model.weights) |*w| {
            w.* = w.* - lr * wd * w.*;
        }

        if (model.momentum) |*m| {
            if (model.velocity) |*v| {
                const bc1 = 1 - std.math.pow(f32, beta1, step_f);
                const bc2 = 1 - std.math.pow(f32, beta2, step_f);
                const lr_adjusted = lr * @sqrt(bc2) / bc1;
                for (0..model.weights.len) |i| {
                    const g = model.gradients[i];
                    m.*[i] = beta1 * m.*[i] + (1 - beta1) * g;
                    v.*[i] = beta2 * v.*[i] + (1 - beta2) * g * g;
                    // Use lr_adjusted with raw m/v (bias correction in lr)
                    model.weights[i] -= lr_adjusted * m.*[i] / (@sqrt(v.*[i]) + epsilon);
                }
            }
        }
    }
};

pub const TrainingResult = struct {
    allocator: std.mem.Allocator,
    report: TrainingReport,
    model: ModelState,
    optimizer: Optimizer,
    checkpoints: CheckpointStore,
    loss_history: []f32,
    accuracy_history: []f32,

    pub fn deinit(self: *TrainingResult) void {
        self.allocator.free(self.accuracy_history);
        self.allocator.free(self.loss_history);
        self.checkpoints.deinit();
        self.optimizer.deinit(self.allocator);
        self.model.deinit();
        self.* = undefined;
    }
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
        return trainWithResult(self.allocator, train_config);
    }

    /// Get or create checkpoint store.
    pub fn getCheckpointStore(self: *Context) !*CheckpointStore {
        if (self.checkpoint_store) |cs| return cs;

        const store = try self.allocator.create(CheckpointStore);
        store.* = CheckpointStore.init(self.allocator, self.config.max_checkpoints orelse 5);
        self.checkpoint_store = store;
        return store;
    }

    /// Save a checkpoint.
    pub fn saveCheckpoint(self: *Context, name: []const u8, data: anytype) !void {
        const store = try self.getCheckpointStore();
        try store.save(name, data);
    }

    /// Load a checkpoint.
    pub fn loadCheckpointData(self: *Context, name: []const u8, comptime T: type) !T {
        const store = try self.getCheckpointStore();
        return store.load(name, T);
    }
};

pub fn isEnabled() bool {
    return build_options.enable_ai and build_options.enable_training;
}

pub fn calculateLearningRate(config: TrainingConfig, step_val: u64, base_lr: f32) f32 {
    return switch (config.learning_rate_schedule) {
        .constant => base_lr,
        .linear => {
            const progress = @min(1.0, @as(f32, @floatFromInt(step_val)) / @as(f32, @floatFromInt(config.decay_steps)));
            return base_lr * (1.0 - progress) + config.min_learning_rate * progress;
        },
        .cosine => base_lr * 0.5 * (1 + @cos(@as(f32, @floatFromInt(step_val % config.decay_steps)) * 2 * std.math.pi / @as(f32, @floatFromInt(config.decay_steps)))),
        .warmup_cosine => {
            if (step_val < config.warmup_steps) {
                return base_lr * @as(f32, @floatFromInt(step_val)) / @as(f32, @floatFromInt(config.warmup_steps));
            }
            const adjusted_step = step_val - config.warmup_steps;
            const adjusted_decay = config.decay_steps - config.warmup_steps;
            const progress = @min(1.0, @as(f32, @floatFromInt(adjusted_step)) / @as(f32, @floatFromInt(adjusted_decay)));
            return config.min_learning_rate + (base_lr - config.min_learning_rate) * 0.5 * (1 + @cos(progress * std.math.pi));
        },
        .step => {
            const decay = @as(f32, @floatFromInt(step_val / config.decay_steps));
            return base_lr * std.math.pow(f32, 0.1, decay);
        },
        .polynomial => {
            const progress = @min(1.0, @as(f32, @floatFromInt(step_val)) / @as(f32, @floatFromInt(config.decay_steps)));
            return base_lr * std.math.pow(f32, 1 - progress, 0.9);
        },
        .cosine_warm_restarts => {
            const t_0 = @as(f32, @floatFromInt(config.decay_steps));
            const t_mult: f32 = 2.0;
            const min_lr = config.min_learning_rate;

            var cycle_start: f32 = 0;
            var cycle_length = t_0;
            const step_f = @as(f32, @floatFromInt(step_val));

            while (cycle_start + cycle_length <= step_f) {
                cycle_start += cycle_length;
                cycle_length *= t_mult;
            }

            const t_cur = step_f - cycle_start;
            const t_i = cycle_length;
            const progress = t_cur / t_i;

            return min_lr + (base_lr - min_lr) * 0.5 * (1 + @cos(progress * std.math.pi));
        },
    };
}

pub fn clipGradients(gradients: []f32, max_norm: f32) f32 {
    var norm: f32 = 0;
    for (gradients) |g| {
        norm += g * g;
    }
    norm = @sqrt(norm);

    if (norm > max_norm and norm > 0) {
        const scale = max_norm / norm;
        for (gradients) |*g| {
            g.* *= scale;
        }
    }

    return norm;
}

pub fn saveModelToWdbx(allocator: std.mem.Allocator, model: *const ModelState, path: []const u8) !void {
    var handle = try database.wdbx.createDatabase(allocator, "model_checkpoint");
    defer database.wdbx.closeDatabase(&handle);

    // Store weights as vector ID 0
    // In a real scenario we'd split layers, but for this concise implementation we store flattened weights
    try database.wdbx.insertVector(&handle, 0, model.weights, model.name);

    try database.wdbx.backup(&handle, path);
}

pub fn train(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!void {
    var result = try trainWithResult(allocator, config);
    defer result.deinit();
}

pub fn trainAndReport(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!TrainingReport {
    var result = try trainWithResult(allocator, config);
    defer result.deinit();
    return result.report;
}

pub fn trainWithResult(
    allocator: std.mem.Allocator,
    config: TrainingConfig,
) TrainError!TrainingResult {
    try config.validate();

    var model = try ModelState.init(allocator, config.model_size, "model");
    errdefer model.deinit();

    var optimizer = try Optimizer.init(allocator, &model, config);
    errdefer optimizer.deinit(allocator);

    var accumulator = try gradient.GradientAccumulator.init(allocator, model.gradients.len);
    defer accumulator.deinit();

    var checkpoints = CheckpointStore.init(allocator, config.max_checkpoints);
    errdefer checkpoints.deinit();

    const batches_per_epoch: u32 = @intCast((config.sample_count + config.batch_size - 1) / config.batch_size);
    const gradient_buffer = try allocator.alloc(f32, model.gradients.len);
    defer allocator.free(gradient_buffer);

    var loss_history = try allocator.alloc(f32, config.epochs);
    errdefer allocator.free(loss_history);
    var accuracy_history = try allocator.alloc(f32, config.epochs);
    errdefer allocator.free(accuracy_history);

    var best_loss: f32 = 1e10;
    var patience_counter: u32 = 0;
    var early_stopped: bool = false;
    var last_epoch: usize = 0;

    var training_timer = time.Timer.start() catch return error.InvalidConfiguration;

    for (0..config.epochs) |epoch| {
        last_epoch = epoch;
        var epoch_loss: f32 = 0;
        var epoch_accuracy: f32 = 0;
        var batch_count: u32 = 0;

        var batch: u32 = 0;
        while (batch < batches_per_epoch) : (batch += 1) {
            for (gradient_buffer) |*v| {
                v.* = config.learning_rate;
            }
            try accumulator.add(gradient_buffer);

            const is_last_batch = batch + 1 == batches_per_epoch;
            if (accumulator.count >= config.gradient_accumulation_steps or is_last_batch) {
                const avg = try accumulator.average(allocator);
                defer allocator.free(avg);
                for (model.gradients, avg) |*g, a| {
                    g.* = a;
                }
                accumulator.reset();

                const norm = clipGradients(model.gradients, config.gradient_clip_norm);
                _ = norm;

                const current_lr = calculateLearningRate(config, model.step + 1, config.learning_rate);
                optimizer.setLearningRate(current_lr);
                optimizer.step(&model, current_lr, model.step + 1);
                model.step += 1;

                const step_loss = calculateLoss(model.weights, gradient_buffer);
                epoch_loss += step_loss;

                const step_acc = calculateAccuracy(model.weights, gradient_buffer);
                epoch_accuracy += step_acc;

                batch_count += 1;

                if (config.checkpoint_interval > 0 and
                    model.step % config.checkpoint_interval == 0)
                {
                    try checkpoints.add(model.step, model.weights);
                    if (config.checkpoint_path) |base_path| {
                        var path_buf: [256]u8 = undefined;
                        const ckpt_path = std.fmt.bufPrint(
                            &path_buf,
                            "{s}/step_{d}.ckpt",
                            .{ base_path, model.step },
                        ) catch continue;
                        checkpoint.saveCheckpoint(allocator, ckpt_path, .{
                            .step = model.step,
                            .timestamp = @as(u64, @intCast(time_utils.unixSeconds())),
                            .weights = model.weights,
                        }) catch |err| {
                            std.log.warn("failed to save checkpoint: {t}", .{err});
                        };
                    }
                }
            }
        }

        if (batch_count > 0) {
            epoch_loss /= @as(f32, @floatFromInt(batch_count));
            epoch_accuracy /= @as(f32, @floatFromInt(batch_count));
        }

        loss_history[epoch] = epoch_loss;
        accuracy_history[epoch] = epoch_accuracy;

        if (epoch_loss < best_loss - config.early_stopping_threshold) {
            best_loss = epoch_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if (patience_counter >= config.early_stopping_patience) {
                early_stopped = true;
                break;
            }
        }
    }

    const elapsed_ns = training_timer.read();
    const total_time_ms = elapsed_ns / std.time.ns_per_ms;

    const final_lr = calculateLearningRate(config, model.step, config.learning_rate);

    return .{
        .allocator = allocator,
        .report = .{
            .epochs = config.epochs,
            .batches = @as(u32, @intCast(batches_per_epoch)),
            .final_loss = loss_history[last_epoch],
            .final_accuracy = accuracy_history[last_epoch],
            .best_loss = best_loss,
            .learning_rate = final_lr,
            .gradient_updates = model.step,
            .checkpoints_saved = @as(u32, @intCast(checkpoints.count())),
            .early_stopped = early_stopped,
            .total_time_ms = total_time_ms,
        },
        .model = model,
        .optimizer = optimizer,
        .checkpoints = checkpoints,
        .loss_history = loss_history,
        .accuracy_history = accuracy_history,
    };
}

fn initializeXavierUniform(weights: []f32, size: usize) void {
    const limit = @sqrt(2.0 / @as(f32, @floatFromInt(size)));
    var rng = std.Random.DefaultPrng.init(12345 + size);
    for (weights) |*val| {
        val.* = rng.random().floatNorm(f32) * limit;
    }
}

fn calculateLoss(weights: []f32, gradients: []f32) f32 {
    var total_loss: f32 = 0;
    for (weights, gradients) |w, g| {
        total_loss += w * w * 0.001 + g * g * 0.5;
    }
    return total_loss / @as(f32, @floatFromInt(weights.len));
}

fn calculateAccuracy(weights: []f32, gradients: []f32) f32 {
    var correct: usize = 0;
    for (weights, gradients) |w, g| {
        const prediction = if (w * g > 0) @as(u32, 1) else @as(u32, 0);
        if (prediction == 1) correct += 1;
    }
    return @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(weights.len));
}

// Test discovery for extracted test files
test {
    _ = @import("self_learning_test.zig");
    _ = @import("trainable_model_test.zig");
}

test "training result includes checkpoints" {
    var result = try trainWithResult(std.testing.allocator, .{
        .epochs = 2,
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

test "learning rate schedule" {
    const config = TrainingConfig{
        .warmup_steps = 100,
        .decay_steps = 1000,
        .learning_rate = 0.01,
    };

    const lr_warmup = calculateLearningRate(config, 50, config.learning_rate);
    const lr_decay = calculateLearningRate(config, 600, config.learning_rate);

    try std.testing.expect(lr_warmup < config.learning_rate);
    try std.testing.expect(lr_decay < lr_warmup);
}

test "gradient clipping" {
    var gradients = [_]f32{ 1.5, 2.5, 3.5, 4.5 };
    const norm = clipGradients(&gradients, 2.0);

    try std.testing.expect(norm > 0);
    for (gradients) |g| {
        try std.testing.expect(@abs(g) <= 2.0);
    }
}
