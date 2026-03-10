//! Trainable Model Module
//!
//! Refactored model structure.
const std = @import("std");

pub const config = @import("config");
pub const layer = @import("layer");
pub const weights = @import("weights");
pub const cache = @import("cache");
pub const model = @import("model");
pub const utils = @import("utils");

// Re-exports
pub const CheckpointingStrategy = config.CheckpointingStrategy;
pub const TrainableModelConfig = config.TrainableModelConfig;
pub const TrainableLayerWeights = layer.TrainableLayerWeights;
pub const TrainableWeights = weights.TrainableWeights;
pub const ActivationCache = cache.ActivationCache;
pub const TrainableModel = model.TrainableModel;

test {
    std.testing.refAllDecls(@This());
}
