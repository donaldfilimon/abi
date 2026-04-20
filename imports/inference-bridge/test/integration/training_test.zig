//! Integration Tests: Training AI Sub-Feature
//!
//! Tests the training module exports, lifecycle queries, configuration types,
//! optimizer types, and sub-module accessibility through the public
//! `abi.ai.training` surface.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const training = abi.ai.training;

// ============================================================================
// Feature gate
// ============================================================================

test "training: isEnabled reflects feature flag" {
    if (build_options.feat_training) {
        try std.testing.expect(training.isEnabled());
    } else {
        try std.testing.expect(!training.isEnabled());
    }
}

// ============================================================================
// Error types
// ============================================================================

test "training: Error type is accessible" {
    const E = training.Error;
    const err: E = error.FeatureDisabled;
    try std.testing.expect(err == error.FeatureDisabled);
}

test "training: Error includes expected variants" {
    const e1: training.Error = error.InvalidConfig;
    const e2: training.Error = error.CheckpointFailed;
    const e3: training.Error = error.TrainingFailed;
    try std.testing.expect(e1 == error.InvalidConfig);
    try std.testing.expect(e2 == error.CheckpointFailed);
    try std.testing.expect(e3 == error.TrainingFailed);
}

// ============================================================================
// Config types
// ============================================================================

test "training: TrainingConfig default values" {
    const cfg = training.TrainingConfig{};
    try std.testing.expectEqual(@as(u32, 10), cfg.epochs);
    try std.testing.expectEqual(@as(u32, 32), cfg.batch_size);
    try std.testing.expectEqual(training.OptimizerType.adamw, cfg.optimizer);
    try std.testing.expectEqual(training.LearningRateSchedule.constant, cfg.learning_rate_schedule);
    try std.testing.expect(!cfg.mixed_precision);
}

test "training: TrainingConfig with custom values" {
    const cfg = training.TrainingConfig{
        .epochs = 50,
        .batch_size = 64,
        .learning_rate = 0.01,
        .optimizer = .adam,
        .learning_rate_schedule = .cosine,
        .mixed_precision = true,
    };
    try std.testing.expectEqual(@as(u32, 50), cfg.epochs);
    try std.testing.expectEqual(@as(u32, 64), cfg.batch_size);
    try std.testing.expectEqual(training.OptimizerType.adam, cfg.optimizer);
    try std.testing.expectEqual(training.LearningRateSchedule.cosine, cfg.learning_rate_schedule);
    try std.testing.expect(cfg.mixed_precision);
}

test "training: LlmTrainingConfig default values" {
    const cfg = training.LlmTrainingConfig{};
    try std.testing.expectEqual(@as(u32, 1), cfg.epochs);
    try std.testing.expectEqual(@as(u32, 4), cfg.batch_size);
    try std.testing.expectEqual(@as(u32, 512), cfg.max_seq_len);
    try std.testing.expect(!cfg.use_gpu);
    try std.testing.expect(!cfg.mixed_precision);
}

// ============================================================================
// Enum types
// ============================================================================

test "training: OptimizerType enum variants" {
    const sgd = training.OptimizerType.sgd;
    const adam = training.OptimizerType.adam;
    const adamw = training.OptimizerType.adamw;
    try std.testing.expect(sgd != adam);
    try std.testing.expect(adam != adamw);
}

test "training: LearningRateSchedule enum variants" {
    const constant = training.LearningRateSchedule.constant;
    const linear = training.LearningRateSchedule.linear;
    const cosine = training.LearningRateSchedule.cosine;
    try std.testing.expect(constant != linear);
    try std.testing.expect(linear != cosine);
}

test "training: ExperienceType enum variants" {
    const text = training.ExperienceType.text_conversation;
    const vis = training.ExperienceType.vision;
    const code = training.ExperienceType.code;
    try std.testing.expect(text != vis);
    try std.testing.expect(vis != code);
}

test "training: FeedbackType enum variants" {
    const pos = training.FeedbackType.positive;
    const neg = training.FeedbackType.negative;
    try std.testing.expect(pos != neg);
}

// ============================================================================
// Data types
// ============================================================================

test "training: TrainingReport default values" {
    const report = training.TrainingReport{};
    try std.testing.expectEqual(@as(u32, 0), report.epochs);
    try std.testing.expectEqual(@as(f32, 0), report.final_loss);
    try std.testing.expect(!report.early_stopped);
}

test "training: TrainingStats default values" {
    const stats = training.TrainingStats{};
    try std.testing.expectEqual(@as(u32, 0), stats.epoch);
    try std.testing.expectEqual(@as(u64, 0), stats.global_step);
    try std.testing.expectEqual(@as(f32, 0), stats.loss);
}

test "training: LearningStats default values" {
    const stats = training.LearningStats{};
    try std.testing.expectEqual(@as(u64, 0), stats.total_experiences);
    try std.testing.expectEqual(@as(f32, 0), stats.avg_reward);
}

test "training: LoraConfig defaults and scaling" {
    const lora = training.LoraConfig{};
    try std.testing.expectEqual(@as(u32, 8), lora.rank);
    // alpha / rank = 16.0 / 8 = 2.0
    try std.testing.expectEqual(@as(f32, 2.0), lora.getScaling());
}

// ============================================================================
// Sub-modules
// ============================================================================

test "training: checkpoint sub-module is accessible" {
    const cp = training.checkpoint;
    _ = cp.Checkpoint;
    _ = cp.CheckpointStore;
}

test "training: gradient sub-module is accessible" {
    const g = training.gradient;
    _ = g.GradientAccumulator;
}

test "training: loss sub-module is accessible" {
    const l = training.loss;
    _ = l.CrossEntropyLoss;
    _ = l.MSELoss;
    _ = l.FocalLoss;
}

test "training: lora sub-module is accessible" {
    const lr = training.lora;
    _ = lr.LoraAdapter;
    _ = lr.LoraConfig;
    _ = lr.LoraModel;
}

test "training: data_loader sub-module is accessible" {
    const dl = training.data_loader;
    _ = dl.DataLoader;
    _ = dl.Batch;
}

// ============================================================================
// Stub API
// ============================================================================

test "training: train returns result or FeatureDisabled" {
    const result = training.train(std.testing.allocator, .{});
    if (result) |_| {
        // Feature enabled
    } else |err| {
        try std.testing.expectEqual(error.FeatureDisabled, err);
    }
}

test "training: perplexity returns value" {
    const p = training.loss.perplexity(2.0);
    try std.testing.expect(p >= 0);
}

test "training: clipGradients returns value" {
    var grads = [_]f32{ 1.0, 2.0, 3.0 };
    const norm = training.clipGradients(&grads, 1.0);
    try std.testing.expect(norm >= 0);
}

test "training: calculateLearningRate returns value" {
    const lr = training.calculateLearningRate(.{}, 100, 0.001);
    try std.testing.expect(lr >= 0);
}

test {
    std.testing.refAllDecls(@This());
}
