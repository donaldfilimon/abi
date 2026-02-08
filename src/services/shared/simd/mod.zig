//! SIMD vector operations
//!
//! Re-exports from focused submodules. Every public symbol from the
//! original monolithic simd.zig is available here.

pub const vector_ops = @import("vector_ops.zig");
pub const activations = @import("activations.zig");
pub const distances = @import("distances.zig");
pub const integer_ops = @import("integer_ops.zig");
pub const extras = @import("extras.zig");

// ============================================================================
// Re-exports: vector_ops.zig
// ============================================================================

pub const vectorAdd = vector_ops.vectorAdd;
pub const vectorDot = vector_ops.vectorDot;
pub const vectorL2Norm = vector_ops.vectorL2Norm;
pub const cosineSimilarity = vector_ops.cosineSimilarity;
pub const batchCosineSimilarityFast = vector_ops.batchCosineSimilarityFast;
pub const batchCosineSimilarity = vector_ops.batchCosineSimilarity;
pub const batchCosineSimilarityPrecomputed = vector_ops.batchCosineSimilarityPrecomputed;
pub const batchDotProduct = vector_ops.batchDotProduct;
pub const vectorReduce = vector_ops.vectorReduce;
pub const hasSimdSupport = vector_ops.hasSimdSupport;

// ============================================================================
// Re-exports: activations.zig
// ============================================================================

pub const siluInPlace = activations.siluInPlace;
pub const geluInPlace = activations.geluInPlace;
pub const reluInPlace = activations.reluInPlace;
pub const leakyReluInPlace = activations.leakyReluInPlace;
pub const maxValue = activations.maxValue;
pub const expSubtractMax = activations.expSubtractMax;
pub const sum = activations.sum;
pub const divideByScalar = activations.divideByScalar;
pub const softmaxInPlace = activations.softmaxInPlace;
pub const logSoftmaxInPlace = activations.logSoftmaxInPlace;
pub const squaredSum = activations.squaredSum;
pub const rmsNormInPlace = activations.rmsNormInPlace;
pub const layerNormInPlace = activations.layerNormInPlace;

// ============================================================================
// Re-exports: distances.zig
// ============================================================================

pub const l2DistanceSquared = distances.l2DistanceSquared;
pub const l2Distance = distances.l2Distance;
pub const innerProduct = distances.innerProduct;
pub const SimdCapabilities = distances.SimdCapabilities;
pub const getSimdCapabilities = distances.getSimdCapabilities;
pub const matrixMultiply = distances.matrixMultiply;

// ============================================================================
// Re-exports: integer_ops.zig
// ============================================================================

pub const vectorAddI32 = integer_ops.vectorAddI32;
pub const sumI32 = integer_ops.sumI32;
pub const maxI32 = integer_ops.maxI32;
pub const minI32 = integer_ops.minI32;
pub const fma = integer_ops.fma;
pub const fmaScalar = integer_ops.fmaScalar;
pub const scaleInPlace = integer_ops.scaleInPlace;
pub const addScalarInPlace = integer_ops.addScalarInPlace;

// ============================================================================
// Re-exports: extras.zig
// ============================================================================

pub const hadamard = extras.hadamard;
pub const absInPlace = extras.absInPlace;
pub const clampInPlace = extras.clampInPlace;
pub const countGreaterThan = extras.countGreaterThan;
pub const copyF32 = extras.copyF32;
pub const fillF32 = extras.fillF32;
pub const euclideanDistance = extras.euclideanDistance;
pub const softmax = extras.softmax;
pub const saxpy = extras.saxpy;
pub const reduceSum = extras.reduceSum;
pub const reduceMin = extras.reduceMin;
pub const reduceMax = extras.reduceMax;
pub const scale = extras.scale;

// ============================================================================
// Test discovery
// ============================================================================

comptime {
    if (@import("builtin").is_test) {
        _ = vector_ops;
        _ = activations;
        _ = distances;
        _ = integer_ops;
        _ = extras;
        _ = @import("simd_test.zig");
    }
}
