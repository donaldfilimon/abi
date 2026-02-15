//! Trainable LLM model wrapper for training.
//!
//! This file is intentionally a compatibility façade that delegates to the
//! split implementation modules under `src/features/ai/training/trainable_model/`.

const core = @import("trainable_model/core.zig");

pub const TrainableLayerWeights = core.TrainableLayerWeights;
pub const TrainableWeights = core.TrainableWeights;
pub const ActivationCache = core.ActivationCache;
pub const GradientCheckpointer = core.GradientCheckpointer;
pub const ModelCheckpoint = core.ModelCheckpoint;
pub const LoadError = core.LoadError;
pub const CheckpointingStrategy = core.CheckpointingStrategy;
pub const TrainableModelConfig = core.TrainableModelConfig;
pub const TrainableModel = core.TrainableModel;
pub const TrainStepResult = core.TrainableModel.TrainStepResult;
