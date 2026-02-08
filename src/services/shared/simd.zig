//! Compatibility shim — redirects to simd/mod.zig
//!
//! Many files import this path directly. The actual implementation has been
//! split into src/services/shared/simd/ submodules.
//!
//! New code should import via the parent mod.zig chain instead of using
//! a direct file path.

const mod = @import("simd/mod.zig");

// Submodule namespaces
pub const vector_ops = mod.vector_ops;
pub const activations = mod.activations;
pub const distances = mod.distances;
pub const integer_ops = mod.integer_ops;
pub const extras = mod.extras;

// Re-exports: vector_ops
pub const vectorAdd = mod.vectorAdd;
pub const vectorDot = mod.vectorDot;
pub const vectorL2Norm = mod.vectorL2Norm;
pub const cosineSimilarity = mod.cosineSimilarity;
pub const batchCosineSimilarityFast = mod.batchCosineSimilarityFast;
pub const batchCosineSimilarity = mod.batchCosineSimilarity;
pub const batchCosineSimilarityPrecomputed = mod.batchCosineSimilarityPrecomputed;
pub const batchDotProduct = mod.batchDotProduct;
pub const vectorReduce = mod.vectorReduce;
pub const hasSimdSupport = mod.hasSimdSupport;

// Re-exports: activations
pub const siluInPlace = mod.siluInPlace;
pub const geluInPlace = mod.geluInPlace;
pub const reluInPlace = mod.reluInPlace;
pub const leakyReluInPlace = mod.leakyReluInPlace;
pub const maxValue = mod.maxValue;
pub const expSubtractMax = mod.expSubtractMax;
pub const sum = mod.sum;
pub const divideByScalar = mod.divideByScalar;
pub const softmaxInPlace = mod.softmaxInPlace;
pub const logSoftmaxInPlace = mod.logSoftmaxInPlace;
pub const squaredSum = mod.squaredSum;
pub const rmsNormInPlace = mod.rmsNormInPlace;
pub const layerNormInPlace = mod.layerNormInPlace;

// Re-exports: distances
pub const l2DistanceSquared = mod.l2DistanceSquared;
pub const l2Distance = mod.l2Distance;
pub const innerProduct = mod.innerProduct;
pub const SimdCapabilities = mod.SimdCapabilities;
pub const getSimdCapabilities = mod.getSimdCapabilities;
pub const matrixMultiply = mod.matrixMultiply;

// Re-exports: integer_ops
pub const vectorAddI32 = mod.vectorAddI32;
pub const sumI32 = mod.sumI32;
pub const maxI32 = mod.maxI32;
pub const minI32 = mod.minI32;
pub const fma = mod.fma;
pub const fmaScalar = mod.fmaScalar;
pub const scaleInPlace = mod.scaleInPlace;
pub const addScalarInPlace = mod.addScalarInPlace;

// Re-exports: extras
pub const hadamard = mod.hadamard;
pub const absInPlace = mod.absInPlace;
pub const clampInPlace = mod.clampInPlace;
pub const countGreaterThan = mod.countGreaterThan;
pub const copyF32 = mod.copyF32;
pub const fillF32 = mod.fillF32;
pub const euclideanDistance = mod.euclideanDistance;
pub const softmax = mod.softmax;
pub const saxpy = mod.saxpy;
pub const reduceSum = mod.reduceSum;
pub const reduceMin = mod.reduceMin;
pub const reduceMax = mod.reduceMax;
pub const scale = mod.scale;
