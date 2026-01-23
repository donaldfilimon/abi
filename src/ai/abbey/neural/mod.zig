//! Abbey Neural Module
//!
//! Neural network components for Abbey's learning architecture:
//! - Tensor operations with SIMD support
//! - Neural network layers (Linear, Embedding, LayerNorm)
//! - Attention mechanisms (Multi-head, Self, Cross, Adaptive)
//! - Online learning with experience replay
//! - GPU-accelerated operations with automatic CPU fallback

const std = @import("std");

pub const tensor = @import("tensor.zig");
pub const layer = @import("layer.zig");
pub const attention = @import("attention.zig");
pub const learning = @import("learning.zig");
pub const gpu_ops = @import("gpu_ops.zig");

// Tensor exports
pub const Tensor = tensor.Tensor;
pub const F32Tensor = tensor.F32Tensor;
pub const F64Tensor = tensor.F64Tensor;

// Layer exports
pub const Layer = layer.Layer;
pub const LinearLayer = layer.LinearLayer;
pub const EmbeddingLayer = layer.EmbeddingLayer;
pub const LayerNorm = layer.LayerNorm;
pub const Dropout = layer.Dropout;
pub const ReLU = layer.ReLU;
pub const GELU = layer.GELU;
pub const Sequential = layer.Sequential;

// Attention exports
pub const MultiHeadAttention = attention.MultiHeadAttention;
pub const SelfAttention = attention.SelfAttention;
pub const CrossAttention = attention.CrossAttention;
pub const AdaptiveAttention = attention.AdaptiveAttention;
pub const AttentionOutput = attention.AttentionOutput;
pub const scaledDotProductAttention = attention.scaledDotProductAttention;
pub const createCausalMask = attention.createCausalMask;
pub const createPaddingMask = attention.createPaddingMask;

// Learning exports
pub const Experience = learning.Experience;
pub const ReplayBuffer = learning.ReplayBuffer;
pub const SGDOptimizer = learning.SGDOptimizer;
pub const AdamOptimizer = learning.AdamOptimizer;
pub const LossFn = learning.LossFn;
pub const OnlineLearner = learning.OnlineLearner;
pub const GradientAccumulator = learning.GradientAccumulator;

// GPU operations exports
pub const GpuOpsContext = gpu_ops.GpuOpsContext;
pub const GpuStats = gpu_ops.GpuStats;
pub const createGpuContext = gpu_ops.createContext;

// ============================================================================
// Convenience Builders
// ============================================================================

/// Create a simple MLP
pub fn createMLP(
    allocator: std.mem.Allocator,
    layer_sizes: []const usize,
    activation: enum { relu, gelu, none },
) !Sequential {
    var seq = Sequential.init(allocator);
    errdefer seq.deinit();

    for (0..layer_sizes.len - 1) |i| {
        var linear = try LinearLayer.init(allocator, layer_sizes[i], layer_sizes[i + 1]);
        try seq.add(linear.layer());

        if (i < layer_sizes.len - 2) {
            switch (activation) {
                .relu => {
                    const relu = ReLU.init(allocator);
                    _ = relu; // Would add activation layer
                },
                .gelu => {
                    const gelu = GELU.init(allocator);
                    _ = gelu;
                },
                .none => {},
            }
        }
    }

    return seq;
}

/// Create an attention block with layer norm
pub fn createAttentionBlock(
    allocator: std.mem.Allocator,
    d_model: usize,
    num_heads: usize,
) !struct { attention: MultiHeadAttention, norm1: LayerNorm, norm2: LayerNorm } {
    var attn = try MultiHeadAttention.init(allocator, d_model, num_heads);
    errdefer attn.deinit();

    var norm1 = try LayerNorm.init(allocator, d_model);
    errdefer norm1.deinit();

    const norm2 = try LayerNorm.init(allocator, d_model);

    return .{
        .attention = attn,
        .norm1 = norm1,
        .norm2 = norm2,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "neural module imports" {
    const allocator = std.testing.allocator;

    // Test tensor creation
    var t = try F32Tensor.zeros(allocator, &.{ 2, 3 });
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 6), t.size());

    // Test linear layer
    var linear = try LinearLayer.init(allocator, 4, 2);
    defer linear.deinit();
    try std.testing.expectEqual(@as(usize, 10), linear.paramCount()); // 4*2 + 2

    // Test attention
    var attn = try SelfAttention.init(allocator, 8, 2);
    defer attn.deinit();
}
