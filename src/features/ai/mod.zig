//! AI Feature Module
//!
//! Comprehensive AI/ML functionality

const std = @import("std");

// Core AI components
pub const neural = @import("neural.zig");
pub const layer = @import("layer.zig");
pub const activations = @import("activations/mod.zig");
pub const activation = activations; // Legacy alias
pub const localml = @import("localml.zig");
pub const training = @import("training/mod.zig");

// Advanced AI features
pub const transformer = @import("transformer.zig");
pub const reinforcement_learning = @import("reinforcement_learning.zig");
pub const enhanced_agent = @import("enhanced_agent.zig");
pub const agent = @import("agent.zig");
pub const persona_manifest = @import("persona_manifest.zig");

// Infrastructure
pub const serialization = @import("serialization/mod.zig");
pub const model_serialization = serialization;
pub const model_registry = @import("model_registry.zig");
pub const distributed = @import("distributed/mod.zig");
pub const distributed_training = distributed;
pub const dynamic = @import("dynamic.zig");

pub const optimizers = @import("optimizers/mod.zig");
pub const interfaces = @import("interfaces.zig");

// Data structures
pub const data_structures = @import("data_structures/mod.zig");

// Legacy compatibility - unified AI core
pub const ai_core = @import("ai_core.zig");

// Commonly used types re-exported for ergonomic access
pub const NeuralNetwork = ai_core.NeuralNetwork;
pub const EmbeddingGenerator = ai_core.EmbeddingGenerator;
pub const Layer = ai_core.Layer;
pub const LayerType = ai_core.LayerType;
pub const Activation = ai_core.Activation;
pub const Optimizer = ai_core.Optimizer;
pub const LRScheduler = ai_core.LRScheduler;
pub const Regularization = ai_core.Regularization;
pub const MemoryStrategy = ai_core.MemoryStrategy;
pub const ComputeBackend = ai_core.ComputeBackend;
pub const createMLP = ai_core.createMLP;
pub const createCNN = ai_core.createCNN;
pub const createTrainer = ai_core.createTrainer;
pub const createModelHandle = ai_core.createModelHandle;

test {
    std.testing.refAllDecls(@This());
}
