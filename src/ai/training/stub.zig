//! Training Stub Module

const std = @import("std");
const config_module = @import("../../config.zig");

pub const Error = error{
    TrainingDisabled,
    InvalidConfig,
    CheckpointFailed,
    TrainingFailed,
    OutOfMemory,
};

pub const TrainingConfig = struct {
    epochs: u32 = 10,
    batch_size: u32 = 32,
    learning_rate: f32 = 0.001,
};
pub const TrainingReport = struct {};
pub const TrainingResult = struct {
    pub fn deinit(_: *TrainingResult) void {}
};
pub const TrainError = Error;
pub const OptimizerType = enum { sgd, adam, adamw };
pub const LearningRateSchedule = enum { constant, linear, cosine };
pub const CheckpointStore = struct {};
pub const Checkpoint = struct {};
pub const GradientAccumulator = struct {};
pub const LlmTrainingConfig = struct {};
pub const LlamaTrainer = struct {};
pub const TrainableModel = struct {};

pub const Context = struct {
    pub fn init(_: std.mem.Allocator, _: config_module.TrainingConfig) Error!*Context {
        return error.TrainingDisabled;
    }

    pub fn deinit(_: *Context) void {}

    pub fn train(_: *Context, _: TrainingConfig) Error!TrainingResult {
        return error.TrainingDisabled;
    }

    pub fn getCheckpointStore(_: *Context) Error!*CheckpointStore {
        return error.TrainingDisabled;
    }

    pub fn saveCheckpoint(_: *Context, _: []const u8, _: anytype) Error!void {
        return error.TrainingDisabled;
    }

    pub fn loadCheckpoint(_: *Context, _: []const u8, comptime _: type) Error!void {
        return error.TrainingDisabled;
    }
};

pub fn isEnabled() bool {
    return false;
}

pub fn trainAndReport(_: std.mem.Allocator, _: TrainingConfig) Error!TrainingReport {
    return error.TrainingDisabled;
}

pub fn trainWithResult(_: std.mem.Allocator, _: TrainingConfig) Error!TrainingResult {
    return error.TrainingDisabled;
}

pub fn trainLlm(_: std.mem.Allocator, _: LlmTrainingConfig) Error!void {
    return error.TrainingDisabled;
}
