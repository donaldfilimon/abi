//! Operations module for LLM computation kernels.
//!
//! Provides optimized implementations of core operations:
//! - Matrix multiplication (regular and quantized)
//! - Self-attention with KV caching
//! - RoPE (Rotary Position Embeddings)
//! - RMSNorm (Root Mean Square Layer Normalization)
//! - Activation functions (SiLU, GELU, softmax)

const std = @import("std");

pub const matmul = @import("matmul.zig");
pub const matmul_quant = @import("matmul_quant.zig");
pub const attention = @import("attention.zig");
pub const rope = @import("rope.zig");
pub const rmsnorm = @import("rmsnorm.zig");
pub const activations = @import("activations.zig");
pub const ffn = @import("ffn.zig");

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

test "ops module imports" {
    _ = matmul;
    _ = matmul_quant;
    _ = attention;
    _ = rope;
    _ = rmsnorm;
    _ = activations;
    _ = ffn;
}
