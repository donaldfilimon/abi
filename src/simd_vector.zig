// Consolidated SIMD module - re-exported from simd/mod.zig
const std = @import("std");
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
pub fn dotProductSIMD(a: []const f32, b: []const f32, opts: anytype) f32 {
    _ = opts;
    return simd_mod.dotProduct(a, b);
}
pub const matrixVectorMultiply = simd_mod.matrixVectorMultiply;
pub const matrixMultiply = simd_mod.matrixMultiply;
pub const transpose = simd_mod.transpose;

// Re-export common types
pub const f32x4 = simd_mod.f32x4;
pub const f32x8 = simd_mod.f32x8;
pub const f32x16 = simd_mod.f32x16;

// Compatibility shims for legacy benchmark interfaces
pub const SIMDOpts = struct {};

pub const SIMDAlignment = struct {
    pub fn ensureAligned(allocator: std.mem.Allocator, data: []const f32) ![]const f32 {
        // For now, return original slice; caller frees only if different
        _ = allocator;
        return data;
    }
};

pub fn vectorAddSIMD(a: []const f32, b: []const f32, result: []f32) void {
    simd_mod.add(result, a, b);
}

pub fn normalizeSIMD(vector: []f32, opts: anytype) []f32 {
    _ = opts;
    // In-place normalize
    const len = vector.len;
    var norm: f32 = 0.0;
    var i: usize = 0;
    while (i < len) : (i += 1) norm += vector[i] * vector[i];
    norm = @sqrt(norm);
    if (norm == 0.0) return vector;
    i = 0;
    while (i < len) : (i += 1) vector[i] /= norm;
    return vector;
}
