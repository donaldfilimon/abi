// Consolidated SIMD module - re-exported from simd/mod.zig
const simd_mod = @import("simd/mod.zig");

// Re-export all commonly used declarations for backward compatibility
pub const Vector = simd_mod.Vector;
pub const VectorOps = simd_mod.VectorOps;
pub const MatrixOps = simd_mod.MatrixOps;
pub const PerformanceMonitor = simd_mod.PerformanceMonitor;
pub const getPerformanceMonitor = simd_mod.getPerformanceMonitor;

// Re-export common functions
pub const distance = simd_mod.distance;
pub const cosineSimilarity = simd_mod.cosineSimilarity;
pub const add = simd_mod.add;
pub const subtract = simd_mod.subtract;
pub const scale = simd_mod.scale;
pub const normalize = simd_mod.normalize;
pub const dotProduct = simd_mod.dotProduct;
pub const matrixVectorMultiply = simd_mod.matrixVectorMultiply;
pub const matrixMultiply = simd_mod.matrixMultiply;
pub const transpose = simd_mod.transpose;

// Re-export common types
pub const f32x4 = simd_mod.f32x4;
pub const f32x8 = simd_mod.f32x8;
pub const f32x16 = simd_mod.f32x16;
