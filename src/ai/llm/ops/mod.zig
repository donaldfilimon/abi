//! Operations module for LLM computation kernels.
//!
//! Provides optimized implementations of core operations:
//! - Matrix multiplication (regular and quantized)
//! - Self-attention with KV caching
//! - RoPE (Rotary Position Embeddings)
//! - RMSNorm (Root Mean Square Layer Normalization)
//! - Activation functions (SiLU, GELU, softmax)
//! - GPU-accelerated operations with CPU fallback
//! - Backward pass operations for training

const std = @import("std");

pub const matmul = @import("matmul.zig");
pub const matmul_quant = @import("matmul_quant.zig");
pub const attention = @import("attention.zig");
pub const rope = @import("rope.zig");
pub const rmsnorm = @import("rmsnorm.zig");
pub const activations = @import("activations.zig");
pub const ffn = @import("ffn.zig");
pub const gpu = @import("gpu.zig");
pub const backward = @import("backward/mod.zig");

// Re-exports for convenience
pub const matrixMultiply = matmul.matrixMultiply;
pub const matrixMultiplyTransposed = matmul.matrixMultiplyTransposed;

pub const quantizedMatmulQ4 = matmul_quant.quantizedMatmulQ4;
pub const quantizedMatmulQ8 = matmul_quant.quantizedMatmulQ8;

pub const selfAttention = attention.selfAttention;
pub const scaledDotProductAttention = attention.scaledDotProductAttention;

pub const applyRope = rope.applyRope;
pub const RopeCache = rope.RopeCache;

pub const rmsNorm = rmsnorm.rmsNorm;
pub const rmsNormInPlace = rmsnorm.rmsNormInPlace;

pub const silu = activations.silu;
pub const gelu = activations.gelu;
pub const softmax = activations.softmax;
pub const softmaxInPlace = activations.softmaxInPlace;

pub const feedForward = ffn.feedForward;
pub const swiglu = ffn.swiglu;

// GPU re-exports
pub const GpuOpsContext = gpu.GpuOpsContext;
pub const GpuStats = gpu.GpuStats;
pub const createGpuContext = gpu.createContext;

// Backward pass re-exports
pub const matmulBackward = backward.matmulBackward;
pub const matrixVectorBackward = backward.matrixVectorBackward;
pub const rmsNormBackward = backward.rmsNormBackward;
pub const softmaxBackward = backward.softmaxBackward;
pub const ropeBackward = backward.ropeBackward;
pub const attentionBackward = backward.attentionBackward;
pub const swigluBackward = backward.swigluBackward;
pub const AttentionCache = backward.attention_backward.AttentionCache;
pub const SwigluCache = backward.ffn_backward.SwigluCache;

test "ops module imports" {
    _ = matmul;
    _ = matmul_quant;
    _ = attention;
    _ = rope;
    _ = rmsnorm;
    _ = activations;
    _ = ffn;
    _ = gpu;
    _ = backward;
}
