//! Vector Math Property Tests
//!
//! Property-based tests for vector operations including:
//! - Algebraic properties (commutativity, associativity, distributivity)
//! - Normalization properties
//! - Distance metric properties
//! - Numerical stability tests
//!
//! Uses the SIMD operations from src/services/shared/simd/mod.zig

const std = @import("std");
const property = @import("mod.zig");
const generators = @import("generators.zig");
const abi = @import("abi");
const simd = abi.services.simd;

const forAll = property.forAll;
const assert = property.assert;
const Generator = property.Generator;

// ============================================================================
// Test Configuration
// ============================================================================

const TestConfig = property.PropertyConfig{
    .iterations = 100,
    .seed = 42,
    .verbose = false,
};

const HighPrecisionConfig = property.PropertyConfig{
    .iterations = 200,
    .seed = 42,
    .verbose = false,
};

// Tolerance for floating point comparisons
const EPSILON: f32 = 1e-5;
const EPSILON_LOOSE: f32 = 1e-4;

// ============================================================================
// Dot Product Properties
// ============================================================================

test "dot product is commutative: dot(a,b) == dot(b,a)" {
    const gen = generators.vectorPairF32(8);

    const result = forAll(property.VectorPair(8), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(8)) bool {
            const dot_ab = simd.vectorDot(&pair.a, &pair.b);
            const dot_ba = simd.vectorDot(&pair.b, &pair.a);
            return assert.approxEqual(dot_ab, dot_ba, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "dot product with zero vector is zero" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            const zero = [_]f32{0} ** 8;
            const dot = simd.vectorDot(&vec, &zero);
            return assert.approxEqual(dot, 0.0, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "dot product with self equals squared norm" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            const dot_self = simd.vectorDot(&vec, &vec);
            const norm = simd.vectorL2Norm(&vec);
            const norm_sq = norm * norm;
            return assert.approxEqual(dot_self, norm_sq, EPSILON_LOOSE);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "dot product is non-negative for same vector" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            const dot = simd.vectorDot(&vec, &vec);
            return dot >= 0.0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Vector Addition Properties
// ============================================================================

test "vector addition is commutative: a + b == b + a" {
    const gen = generators.vectorPairF32(8);

    const result = forAll(property.VectorPair(8), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(8)) bool {
            var result_ab: [8]f32 = undefined;
            var result_ba: [8]f32 = undefined;

            simd.vectorAdd(&pair.a, &pair.b, &result_ab);
            simd.vectorAdd(&pair.b, &pair.a, &result_ba);

            return assert.slicesApproxEqual(&result_ab, &result_ba, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "vector addition is associative: (a + b) + c == a + (b + c)" {
    const gen = generators.vectorTripleF32(8);

    const result = forAll(property.VectorTriple(8), gen, TestConfig, struct {
        fn check(triple: property.VectorTriple(8)) bool {
            // (a + b) + c
            var ab: [8]f32 = undefined;
            var ab_c: [8]f32 = undefined;
            simd.vectorAdd(&triple.a, &triple.b, &ab);
            simd.vectorAdd(&ab, &triple.c, &ab_c);

            // a + (b + c)
            var bc: [8]f32 = undefined;
            var a_bc: [8]f32 = undefined;
            simd.vectorAdd(&triple.b, &triple.c, &bc);
            simd.vectorAdd(&triple.a, &bc, &a_bc);

            return assert.slicesApproxEqual(&ab_c, &a_bc, EPSILON_LOOSE);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "vector addition with zero is identity" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            const zero = [_]f32{0} ** 8;
            var result_arr: [8]f32 = undefined;

            simd.vectorAdd(&vec, &zero, &result_arr);

            return assert.slicesApproxEqual(&result_arr, &vec, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// L2 Norm Properties
// ============================================================================

test "L2 norm is non-negative" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            const norm = simd.vectorL2Norm(&vec);
            return norm >= 0.0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "L2 norm is zero only for zero vector" {
    const gen = generators.nonZeroVector(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            const norm = simd.vectorL2Norm(&vec);
            // Non-zero vector should have positive norm
            return norm > 0.0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "L2 norm scales linearly with scalar multiplication" {
    const gen = generators.scalarVectorF32(8);

    const result = forAll(property.ScalarVectorPair(8), gen, TestConfig, struct {
        fn check(pair: property.ScalarVectorPair(8)) bool {
            const norm_v = simd.vectorL2Norm(&pair.vector);

            // Scale the vector
            var scaled: [8]f32 = undefined;
            for (&scaled, pair.vector) |*s, v| {
                s.* = pair.scalar * v;
            }
            const norm_scaled = simd.vectorL2Norm(&scaled);

            // |scalar * v| = |scalar| * |v|
            const expected = @abs(pair.scalar) * norm_v;

            return assert.approxEqual(norm_scaled, expected, EPSILON_LOOSE);
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Normalization Properties
// ============================================================================

test "normalized vectors have unit length" {
    const gen = generators.nonZeroVector(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            // Normalize manually
            const norm = simd.vectorL2Norm(&vec);
            if (norm < 1e-10) return true; // Skip near-zero vectors

            var normalized: [8]f32 = undefined;
            for (&normalized, vec) |*n, v| {
                n.* = v / norm;
            }

            const unit_norm = simd.vectorL2Norm(&normalized);
            return assert.approxEqual(unit_norm, 1.0, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "normalization preserves direction (dot product with original is positive)" {
    const gen = generators.nonZeroVector(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            const norm = simd.vectorL2Norm(&vec);
            if (norm < 1e-10) return true;

            var normalized: [8]f32 = undefined;
            for (&normalized, vec) |*n, v| {
                n.* = v / norm;
            }

            const dot = simd.vectorDot(&normalized, &vec);
            return dot >= 0.0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Cosine Similarity Properties
// ============================================================================

test "cosine similarity is symmetric: cos(a,b) == cos(b,a)" {
    const gen = generators.vectorPairF32(8);

    const result = forAll(property.VectorPair(8), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(8)) bool {
            const cos_ab = simd.cosineSimilarity(&pair.a, &pair.b);
            const cos_ba = simd.cosineSimilarity(&pair.b, &pair.a);
            return assert.approxEqual(cos_ab, cos_ba, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "cosine similarity is in range [-1, 1]" {
    const gen = generators.vectorPairF32(8);

    const result = forAll(property.VectorPair(8), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(8)) bool {
            const cos = simd.cosineSimilarity(&pair.a, &pair.b);

            // Handle edge cases (zero vectors give 0.0)
            if (std.math.isNan(cos)) return false;

            return cos >= -1.0 - EPSILON and cos <= 1.0 + EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "cosine similarity with self is 1.0" {
    const gen = generators.nonZeroVector(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            const cos = simd.cosineSimilarity(&vec, &vec);
            return assert.approxEqual(cos, 1.0, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "cosine similarity with negation is -1.0" {
    const gen = generators.nonZeroVector(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            var neg: [8]f32 = undefined;
            for (&neg, vec) |*n, v| {
                n.* = -v;
            }

            const cos = simd.cosineSimilarity(&vec, &neg);
            return assert.approxEqual(cos, -1.0, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Distance Properties
// ============================================================================

test "L2 distance is symmetric: dist(a,b) == dist(b,a)" {
    const gen = generators.vectorPairF32(8);

    const result = forAll(property.VectorPair(8), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(8)) bool {
            const dist_ab = simd.l2Distance(&pair.a, &pair.b);
            const dist_ba = simd.l2Distance(&pair.b, &pair.a);
            return assert.approxEqual(dist_ab, dist_ba, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "L2 distance is non-negative" {
    const gen = generators.vectorPairF32(8);

    const result = forAll(property.VectorPair(8), gen, TestConfig, struct {
        fn check(pair: property.VectorPair(8)) bool {
            const dist = simd.l2Distance(&pair.a, &pair.b);
            return dist >= 0.0;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "L2 distance with self is zero" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            const dist = simd.l2Distance(&vec, &vec);
            return assert.approxEqual(dist, 0.0, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "triangle inequality: dist(a,c) <= dist(a,b) + dist(b,c)" {
    const gen = generators.vectorTripleF32(8);

    const result = forAll(property.VectorTriple(8), gen, TestConfig, struct {
        fn check(triple: property.VectorTriple(8)) bool {
            const dist_ac = simd.l2Distance(&triple.a, &triple.c);
            const dist_ab = simd.l2Distance(&triple.a, &triple.b);
            const dist_bc = simd.l2Distance(&triple.b, &triple.c);

            // Allow small tolerance for floating point
            return dist_ac <= dist_ab + dist_bc + EPSILON;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Numerical Stability Properties
// ============================================================================

test "operations produce finite results for normal inputs" {
    const gen = generators.vectorPairF32(8);

    const result = forAll(property.VectorPair(8), gen, HighPrecisionConfig, struct {
        fn check(pair: property.VectorPair(8)) bool {
            // All operations should produce finite results
            const dot = simd.vectorDot(&pair.a, &pair.b);
            const cos = simd.cosineSimilarity(&pair.a, &pair.b);
            const dist = simd.l2Distance(&pair.a, &pair.b);
            const norm_a = simd.vectorL2Norm(&pair.a);
            const norm_b = simd.vectorL2Norm(&pair.b);

            return assert.isFinite(dot) and
                assert.isFinite(cos) and
                assert.isFinite(dist) and
                assert.isFinite(norm_a) and
                assert.isFinite(norm_b);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "softmax produces valid probability distribution" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            var data = vec;
            simd.softmaxInPlace(&data);

            // Check all values are in [0, 1]
            for (data) |v| {
                if (v < 0.0 or v > 1.0 + EPSILON) return false;
            }

            // Check sum is approximately 1.0
            var total: f32 = 0.0;
            for (data) |v| {
                total += v;
            }

            return assert.approxEqual(total, 1.0, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "softmax preserves relative ordering" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            var data = vec;
            simd.softmaxInPlace(&data);

            // If input[i] > input[j], then output[i] >= output[j] - epsilon
            for (0..8) |i| {
                for (0..8) |j| {
                    if (vec[i] > vec[j] + EPSILON) {
                        if (data[i] < data[j] - EPSILON) return false;
                    }
                }
            }
            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Activation Function Properties
// ============================================================================

test "ReLU outputs are non-negative" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            var data = vec;
            simd.reluInPlace(&data);

            for (data) |v| {
                if (v < 0.0) return false;
            }
            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "ReLU is idempotent: relu(relu(x)) == relu(x)" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            var data1 = vec;
            simd.reluInPlace(&data1);

            var data2 = data1;
            simd.reluInPlace(&data2);

            return assert.slicesApproxEqual(&data1, &data2, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "SiLU(0) == 0" {
    const gen = generators.vectorF32(8);
    _ = gen;

    // Direct test with zero
    var zero = [_]f32{0} ** 8;
    simd.siluInPlace(&zero);

    for (zero) |v| {
        try std.testing.expect(assert.approxEqual(v, 0.0, EPSILON));
    }
}

test "leaky ReLU preserves sign for negative inputs" {
    const gen = generators.vectorF32(8);

    const result = forAll([8]f32, gen, TestConfig, struct {
        fn check(vec: [8]f32) bool {
            var data = vec;
            simd.leakyReluInPlace(&data, 0.1);

            for (data, vec) |output, input| {
                if (input < 0.0) {
                    // Output should be negative but smaller in magnitude
                    if (output > 0.0) return false;
                } else {
                    // Positive inputs unchanged
                    if (!assert.approxEqual(output, input, EPSILON)) return false;
                }
            }
            return true;
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Matrix Multiplication Properties
// ============================================================================

test "matrix multiplication produces finite results" {
    // Test with small matrices to keep test fast
    const dim = 4;
    const MatrixPair = struct {
        a: [dim * dim]f32,
        b: [dim * dim]f32,
    };

    const gen = Generator(MatrixPair){
        .generateFn = struct {
            fn generate(prng: *std.Random.DefaultPrng, _: usize) MatrixPair {
                var result: MatrixPair = undefined;
                for (&result.a) |*v| {
                    v.* = prng.random().float(f32) * 2.0 - 1.0;
                }
                for (&result.b) |*v| {
                    v.* = prng.random().float(f32) * 2.0 - 1.0;
                }
                return result;
            }
        }.generate,
        .shrinkFn = null,
    };

    const result = forAll(MatrixPair, gen, TestConfig, struct {
        fn check(pair: MatrixPair) bool {
            var c: [dim * dim]f32 = undefined;
            simd.matrixMultiply(&pair.a, &pair.b, &c, dim, dim, dim);

            return assert.allFinite(&c);
        }
    }.check);

    try std.testing.expect(result.passed);
}

// ============================================================================
// Reduction Properties
// ============================================================================

test "sum reduction matches manual sum" {
    const gen = generators.vectorF32(16);

    const result = forAll([16]f32, gen, TestConfig, struct {
        fn check(vec: [16]f32) bool {
            const simd_sum = simd.sum(&vec);

            var manual_sum: f32 = 0.0;
            for (vec) |v| {
                manual_sum += v;
            }

            return assert.approxEqual(simd_sum, manual_sum, EPSILON_LOOSE);
        }
    }.check);

    try std.testing.expect(result.passed);
}

test "max reduction finds correct maximum" {
    const gen = generators.vectorF32(16);

    const result = forAll([16]f32, gen, TestConfig, struct {
        fn check(vec: [16]f32) bool {
            const simd_max = simd.maxValue(&vec);

            var manual_max: f32 = vec[0];
            for (vec[1..]) |v| {
                manual_max = @max(manual_max, v);
            }

            return assert.approxEqual(simd_max, manual_max, EPSILON);
        }
    }.check);

    try std.testing.expect(result.passed);
}
