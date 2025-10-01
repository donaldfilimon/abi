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
pub const tools = struct {
    pub const summarize = @import("tools/summarize.zig");
    pub const embed = @import("tools/embed.zig");
};

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

test {
    std.testing.refAllDecls(@This());
}
