//! Model module for LLM architectures.
//!
//! Implements the LLaMA model architecture and supporting components.

const std = @import("std");

pub const config = @import("config");
pub const weights = @import("weights");
pub const layer = @import("layer");
pub const llama = @import("llama");

// Re-exports
pub const LlamaConfig = config.LlamaConfig;
pub const ModelConfig = config.ModelConfig;

pub const LlamaWeights = weights.LlamaWeights;

pub const TransformerLayer = layer.TransformerLayer;

pub const LlamaModel = llama.LlamaModel;

test "model module imports" {
    _ = config;
    _ = weights;
    _ = layer;
    _ = llama;
}

test {
    std.testing.refAllDecls(@This());
}
