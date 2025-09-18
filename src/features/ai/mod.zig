//! AI Feature Module
//!
//! Comprehensive AI/ML functionality

const std = @import("std");

// Core AI components
pub const neural = @import("neural.zig");
pub const layer = @import("layer.zig");
pub const activation = @import("activation.zig");
pub const localml = @import("localml.zig");

// Advanced AI features
pub const transformer = @import("transformer.zig");
pub const reinforcement_learning = @import("reinforcement_learning.zig");
pub const enhanced_agent = @import("enhanced_agent.zig");
pub const agent = @import("agent.zig");

// Infrastructure
pub const model_serialization = @import("model_serialization.zig");
pub const model_registry = @import("model_registry.zig");
pub const distributed_training = @import("distributed_training.zig");
pub const dynamic = @import("dynamic.zig");

// Data structures
pub const data_structures = @import("data_structures/mod.zig");

// Legacy compatibility - unified AI core
pub const ai_core = @import("ai_core.zig");

test {
    std.testing.refAllDecls(@This());
}
