//! Trainable Model Module
//!
//! Refactored model structure.

pub const config = @import("config.zig");
pub const layer = @import("layer.zig");
pub const weights = @import("weights.zig");
pub const cache = @import("cache.zig");
pub const model = @import("model.zig");
pub const utils = @import("utils.zig");

// Re-exports
pub const CheckpointingStrategy = config.CheckpointingStrategy;
pub const TrainableModelConfig = config.TrainableModelConfig;
pub const TrainableLayerWeights = layer.TrainableLayerWeights;
pub const TrainableWeights = weights.TrainableWeights;
pub const ActivationCache = cache.ActivationCache;
pub const TrainableModel = model.TrainableModel;
