//! SIMD vector operations
//!
//! Provides high-performance vectorized operations using SIMD instructions
//! when available (AVX-512, NEON, WASM SIMD).
//!
//! # Safety Requirements
//! All functions require input vectors to be properly sized:
//! - All input vectors must have matching lengths where applicable
//! - Empty vectors are not allowed (undefined behavior)
//! - Result buffers must be pre-allocated with correct size
//!
//! # Performance Notes
//! - Functions use @Vector for SIMD acceleration when VectorSize > 1
//! - Debug builds include additional bounds checking via std.debug.assert
//! - Release builds rely on loop bounds for safety (no debug.assert overhead)

const std = @import("std");

const VectorSize = std.simd.suggestVectorLength(f32) orelse 4;

/// Vector addition using SIMD when available
/// @param a First input vector
/// @param b Second input vector (must have same length as a)
/// @param result Output buffer (must have same length as inputs, caller-owned)
pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len > 0);
    std.debug.assert(b.len > 0);
    std.debug.assert(a.len == b.len and a.len == result.len);

    const len = a.len;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            result[i..][0..VectorSize].* = va + vb;
        }
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// Vector dot product using SIMD when available
/// @param a First input vector
/// @param b Second input vector (must have same length as a)
/// @return Scalar dot product
pub fn vectorDot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len > 0);
    std.debug.assert(b.len > 0);
    std.debug.assert(a.len == b.len);

    const len = a.len;
    var dot_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            vec_sum += va * vb;
        }

        // Use @reduce for efficient horizontal sum (Zig 0.16+)
        dot_sum += @reduce(.Add, vec_sum);
    }

    while (i < len) : (i += 1) {
        dot_sum += a[i] * b[i];
    }

    return dot_sum;
}

/// Vector L2 norm using SIMD when available
/// @param v Input vector (must have len > 0)
/// @return Euclidean norm of the vector
pub fn vectorL2Norm(v: []const f32) f32 {
    std.debug.assert(v.len > 0);
    const len = v.len;
    var norm_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const vv: Vec = v[i..][0..VectorSize].*;
            vec_sum += vv * vv;
        }

        // Use @reduce for efficient horizontal sum (Zig 0.16+)
        norm_sum += @reduce(.Add, vec_sum);
    }

    while (i < len) : (i += 1) {
        norm_sum += v[i] * v[i];
    }

    return @sqrt(norm_sum);
}

/// Cosine similarity using SIMD operations
/// @param a First input vector (must not be empty)
/// @param b Second input vector (must not be empty and same length as a)
/// @return Cosine similarity in range [-1, 1], or 0.0 for zero-length vectors
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len == 0 or b.len == 0) return 0.0;
    if (a.len != b.len) return 0.0;

    const dot_product = vectorDot(a, b);
    const norm_a = vectorL2Norm(a);
    const norm_b = vectorL2Norm(b);

    if (norm_a == 0.0 or norm_b == 0.0) {
        return 0.0;
    }

    return dot_product / (norm_a * norm_b);
}

/// Batch cosine similarity computation with pre-computed query norm
/// Fast version that avoids redundant query norm computation
/// @param query Query vector (must not be empty)
/// @param query_norm Pre-computed L2 norm of query (must be > 0)
/// @param vectors Array of database vectors
/// @param results Output array (must have same length as vectors, caller-owned)
pub fn batchCosineSimilarityFast(
    query: []const f32,
    query_norm: f32,
    vectors: []const []const f32,
    results: []f32,
) void {
    std.debug.assert(query.len > 0);
    std.debug.assert(query_norm > 0.0);
    std.debug.assert(vectors.len == results.len);

    for (vectors, results) |vector, *result| {
        const dot = vectorDot(query, vector);
        const vec_norm = vectorL2Norm(vector);
        result.* = if (query_norm > 0 and vec_norm > 0)
            dot / (query_norm * vec_norm)
        else
            0;
    }
}

/// Batch cosine similarity computation for database searches
/// Computes cosine similarity between a query vector and multiple database vectors
/// @param query Query vector (must not be empty)
/// @param vectors Array of database vectors
/// @param results Output array (must have same length as vectors, caller-owned)
pub fn batchCosineSimilarity(
    query: []const f32,
    vectors: []const []const f32,
    results: []f32,
) void {
    std.debug.assert(query.len > 0);
    std.debug.assert(vectors.len == results.len);

    const query_norm = vectorL2Norm(query);
    batchCosineSimilarityFast(query, query_norm, vectors, results);
}

/// Batch cosine similarity with pre-computed norms for maximum performance
/// Use this when database vector norms are pre-computed and stored
/// @param query Query vector (must not be empty)
/// @param query_norm Pre-computed L2 norm of query (must be > 0)
/// @param vectors Array of database vectors
/// @param vector_norms Pre-computed L2 norms of database vectors (same length as vectors)
/// @param results Output array (must have same length as vectors, caller-owned)
pub fn batchCosineSimilarityPrecomputed(
    query: []const f32,
    query_norm: f32,
    vectors: []const []const f32,
    vector_norms: []const f32,
    results: []f32,
) void {
    std.debug.assert(query.len > 0);
    std.debug.assert(query_norm > 0.0);
    std.debug.assert(vectors.len == results.len);
    std.debug.assert(vectors.len == vector_norms.len);

    for (vectors, vector_norms, results) |vector, vec_norm, *result| {
        const dot = vectorDot(query, vector);
        result.* = if (query_norm > 0 and vec_norm > 0)
            dot / (query_norm * vec_norm)
        else
            0;
    }
}

/// Batch dot‑product computation.
/// Computes the dot product of a single `query` vector against each vector in `vectors`.
/// Results are stored in the `results` slice, which must have the same length as `vectors`.
pub fn batchDotProduct(
    query: []const f32,
    vectors: []const []const f32,
    results: []f32,
) void {
    std.debug.assert(query.len > 0);
    std.debug.assert(vectors.len == results.len);
    for (vectors, results) |vec, *res| {
        res.* = vectorDot(query, vec);
    }
}

/// Vector reduction operations with SIMD acceleration
/// @param op Reduction operation: sum, max, or min
/// @param v Input vector
/// @return Reduced value (0.0 for sum on empty, undefined for min/max on empty)
pub fn vectorReduce(op: enum { sum, max, min }, v: []const f32) f32 {
    if (v.len == 0) return 0.0;

    const len = v.len;
    var i: usize = 0;
    var result: f32 = if (op == .sum) 0.0 else v[0];

    if (comptime VectorSize > 1 and len >= VectorSize) {
        const Vec = @Vector(VectorSize, f32);
        var vec_result: Vec = @splat(result);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const vv: Vec = v[i..][0..VectorSize].*;
            switch (op) {
                .sum => vec_result += vv,
                .max => vec_result = @max(vec_result, vv),
                .min => vec_result = @min(vec_result, vv),
            }
        }

        // Use @reduce for efficient horizontal reduction (Zig 0.16+)
        switch (op) {
            .sum => result += @reduce(.Add, vec_result),
            .max => result = @max(result, @reduce(.Max, vec_result)),
            .min => result = @min(result, @reduce(.Min, vec_result)),
        }
    }

    while (i < len) : (i += 1) {
        switch (op) {
            .sum => result += v[i],
            .max => result = @max(result, v[i]),
            .min => result = @min(result, v[i]),
        }
    }

    return result;
}

/// Check if SIMD is available at compile time
pub fn hasSimdSupport() bool {
    return VectorSize > 1;
}

// ============================================================================
// Activation Functions (SIMD-accelerated)
// ============================================================================

/// SiLU (Swish) activation: x * sigmoid(x)
/// In-place modification for memory efficiency
pub fn siluInPlace(data: []f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const one: Vec = @splat(1.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            // sigmoid(x) = 1 / (1 + exp(-x))
            const neg_x = -x;
            const exp_neg = @exp(neg_x);
            const sigmoid = one / (one + exp_neg);
            data[i..][0..VectorSize].* = x * sigmoid;
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        const x = data[i];
        const sig = 1.0 / (1.0 + @exp(-x));
        data[i] = x * sig;
    }
}

/// GELU activation (approximate): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
/// In-place modification for memory efficiency
pub fn geluInPlace(data: []f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const sqrt_2_pi: Vec = @splat(0.7978845608); // sqrt(2/pi)
        const coeff: Vec = @splat(0.044715);
        const half: Vec = @splat(0.5);
        const one: Vec = @splat(1.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            const x3 = x * x * x;
            const inner = sqrt_2_pi * (x + coeff * x3);
            // tanh approximation or use @tanh
            const tanh_val = tanhVec(inner);
            data[i..][0..VectorSize].* = half * x * (one + tanh_val);
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        const x = data[i];
        const x3 = x * x * x;
        const inner = 0.7978845608 * (x + 0.044715 * x3);
        const tanh_val = std.math.tanh(inner);
        data[i] = 0.5 * x * (1.0 + tanh_val);
    }
}

/// ReLU activation: max(0, x)
/// In-place modification
pub fn reluInPlace(data: []f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const zero: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = @max(zero, x);
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        data[i] = @max(0.0, data[i]);
    }
}

/// Leaky ReLU activation: x if x > 0 else alpha * x
/// In-place modification
pub fn leakyReluInPlace(data: []f32, alpha: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const zero: Vec = @splat(0.0);
        const alpha_vec: Vec = @splat(alpha);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            // leaky_relu = x if x > 0 else alpha * x
            // = max(x, alpha * x) when alpha < 1
            const scaled = alpha_vec * x;
            const mask = x > zero;
            data[i..][0..VectorSize].* = @select(f32, mask, x, scaled);
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        const x = data[i];
        data[i] = if (x > 0) x else alpha * x;
    }
}

// ============================================================================
// Softmax Operations (SIMD-accelerated)
// ============================================================================

/// Find maximum value in array using SIMD
pub fn maxValue(data: []const f32) f32 {
    if (data.len == 0) return -std.math.inf(f32);

    var i: usize = 0;
    var max_val: f32 = data[0];

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var max_vec: Vec = @splat(data[0]);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            max_vec = @max(max_vec, v);
        }

        // Horizontal max
        max_val = @reduce(.Max, max_vec);
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        max_val = @max(max_val, data[i]);
    }

    return max_val;
}

/// Compute exp(x - max) in-place for numerical stability
pub fn expSubtractMax(data: []f32, max_val: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const max_vec: Vec = @splat(max_val);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            const diff: Vec = x - max_vec;
            const exp_result: Vec = @exp(diff);
            data[i..][0..VectorSize].* = exp_result;
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        data[i] = @exp(data[i] - max_val);
    }
}

/// Sum all values in array using SIMD
pub fn sum(data: []const f32) f32 {
    if (data.len == 0) return 0.0;

    var i: usize = 0;
    var total: f32 = 0.0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            sum_vec += v;
        }

        total = @reduce(.Add, sum_vec);
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        total += data[i];
    }

    return total;
}

/// Divide all values by a scalar using SIMD
pub fn divideByScalar(data: []f32, divisor: f32) void {
    if (data.len == 0 or divisor == 0.0) return;

    const inv = 1.0 / divisor;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const inv_vec: Vec = @splat(inv);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = v * inv_vec;
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        data[i] *= inv;
    }
}

/// Complete softmax operation in-place
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
pub fn softmaxInPlace(data: []f32) void {
    if (data.len == 0) return;

    // Step 1: Find max for numerical stability
    const max_val = maxValue(data);

    // Step 2: Compute exp(x - max) in-place
    expSubtractMax(data, max_val);

    // Step 3: Compute sum
    const total = sum(data);

    // Step 4: Normalize
    if (total > 0.0) {
        divideByScalar(data, total);
    }
}

/// Log-softmax for cross-entropy loss: log(softmax(x))
/// More numerically stable than log(softmax(x))
pub fn logSoftmaxInPlace(data: []f32) void {
    if (data.len == 0) return;

    // log_softmax(x)_i = x_i - max(x) - log(sum(exp(x - max(x))))
    const max_val = maxValue(data);

    // Compute sum of exp(x - max) using SIMD
    var exp_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const max_vec: Vec = @splat(max_val);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            sum_vec += @exp(x - max_vec);
        }

        exp_sum = @reduce(.Add, sum_vec);
    }

    // Scalar tail for exp sum
    while (i < data.len) : (i += 1) {
        exp_sum += @exp(data[i] - max_val);
    }

    const log_sum = @log(exp_sum);

    // Apply log-softmax in-place using SIMD
    i = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const max_vec: Vec = @splat(max_val);
        const log_sum_vec: Vec = @splat(log_sum);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = x - max_vec - log_sum_vec;
        }
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        data[i] = data[i] - max_val - log_sum;
    }
}

// ============================================================================
// Normalization Operations (SIMD-accelerated)
// ============================================================================

/// Compute sum of squares using SIMD (for RMSNorm)
pub fn squaredSum(data: []const f32) f32 {
    if (data.len == 0) return 0.0;

    var i: usize = 0;
    var total: f32 = 0.0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            sum_vec += v * v;
        }

        total = @reduce(.Add, sum_vec);
    }

    // Scalar tail
    while (i < data.len) : (i += 1) {
        total += data[i] * data[i];
    }

    return total;
}

/// RMSNorm: x / sqrt(mean(x^2) + eps)
/// Modifies data in-place and optionally applies weights
pub fn rmsNormInPlace(data: []f32, weights: ?[]const f32, eps: f32) void {
    if (data.len == 0) return;

    // Compute RMS
    const sq_sum = squaredSum(data);
    const mean_sq = sq_sum / @as(f32, @floatFromInt(data.len));
    const rms = @sqrt(mean_sq + eps);
    const inv_rms = 1.0 / rms;

    var i: usize = 0;

    if (weights) |w| {
        std.debug.assert(w.len == data.len);

        if (comptime VectorSize > 1) {
            const Vec = @Vector(VectorSize, f32);
            const inv_rms_vec: Vec = @splat(inv_rms);

            while (i + VectorSize <= data.len) : (i += VectorSize) {
                const x: Vec = data[i..][0..VectorSize].*;
                const weight: Vec = w[i..][0..VectorSize].*;
                data[i..][0..VectorSize].* = x * inv_rms_vec * weight;
            }
        }

        // Scalar tail
        while (i < data.len) : (i += 1) {
            data[i] = data[i] * inv_rms * w[i];
        }
    } else {
        if (comptime VectorSize > 1) {
            const Vec = @Vector(VectorSize, f32);
            const inv_rms_vec: Vec = @splat(inv_rms);

            while (i + VectorSize <= data.len) : (i += VectorSize) {
                const x: Vec = data[i..][0..VectorSize].*;
                data[i..][0..VectorSize].* = x * inv_rms_vec;
            }
        }

        // Scalar tail
        while (i < data.len) : (i += 1) {
            data[i] *= inv_rms;
        }
    }
}

/// LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta
/// Modifies data in-place
pub fn layerNormInPlace(data: []f32, gamma: ?[]const f32, beta: ?[]const f32, eps: f32) void {
    if (data.len == 0) return;

    // Compute mean
    const mean = sum(data) / @as(f32, @floatFromInt(data.len));

    // Compute variance
    var variance: f32 = 0.0;
    for (data) |x| {
        const diff = x - mean;
        variance += diff * diff;
    }
    variance /= @as(f32, @floatFromInt(data.len));

    const inv_std = 1.0 / @sqrt(variance + eps);

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const mean_vec: Vec = @splat(mean);
        const inv_std_vec: Vec = @splat(inv_std);

        if (gamma != null and beta != null) {
            const g = gamma.?;
            const b = beta.?;
            while (i + VectorSize <= data.len) : (i += VectorSize) {
                const x: Vec = data[i..][0..VectorSize].*;
                const g_vec: Vec = g[i..][0..VectorSize].*;
                const b_vec: Vec = b[i..][0..VectorSize].*;
                data[i..][0..VectorSize].* = (x - mean_vec) * inv_std_vec * g_vec + b_vec;
            }
        } else {
            while (i + VectorSize <= data.len) : (i += VectorSize) {
                const x: Vec = data[i..][0..VectorSize].*;
                data[i..][0..VectorSize].* = (x - mean_vec) * inv_std_vec;
            }
        }
    }

    // Scalar tail
    if (gamma != null and beta != null) {
        const g = gamma.?;
        const b = beta.?;
        while (i < data.len) : (i += 1) {
            data[i] = (data[i] - mean) * inv_std * g[i] + b[i];
        }
    } else {
        while (i < data.len) : (i += 1) {
            data[i] = (data[i] - mean) * inv_std;
        }
    }
}

// ============================================================================
// Distance Calculations (SIMD-accelerated)
// ============================================================================

/// L2 (Euclidean) squared distance using SIMD
/// Returns sum((a - b)^2) - use sqrt for actual distance
pub fn l2DistanceSquared(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    if (a.len == 0) return 0.0;

    var i: usize = 0;
    var total: f32 = 0.0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= a.len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            const diff = va - vb;
            sum_vec += diff * diff;
        }

        total = @reduce(.Add, sum_vec);
    }

    // Scalar tail
    while (i < a.len) : (i += 1) {
        const diff = a[i] - b[i];
        total += diff * diff;
    }

    return total;
}

/// L2 distance (actual Euclidean distance)
pub fn l2Distance(a: []const f32, b: []const f32) f32 {
    return @sqrt(l2DistanceSquared(a, b));
}

/// Inner product (dot product) - already have vectorDot, this is an alias
pub const innerProduct = vectorDot;

// ============================================================================
// Helper Functions
// ============================================================================

/// Fast vectorized tanh approximation
/// Uses the identity: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
fn tanhVec(x: @Vector(VectorSize, f32)) @Vector(VectorSize, f32) {
    const Vec = @Vector(VectorSize, f32);
    const one: Vec = @splat(1.0);
    const two: Vec = @splat(2.0);

    const exp2x = @exp(two * x);
    return (exp2x - one) / (exp2x + one);
}

/// SIMD instruction set capabilities detected at compile time
pub const SimdCapabilities = struct {
    vector_size: usize,
    has_simd: bool,
    arch: Arch,

    pub const Arch = enum {
        x86_64,
        aarch64,
        wasm,
        generic,
    };
};

/// Get SIMD capabilities for current platform
pub fn getSimdCapabilities() SimdCapabilities {
    const builtin = @import("builtin");
    const arch: SimdCapabilities.Arch = switch (builtin.cpu.arch) {
        .x86_64 => .x86_64,
        .aarch64 => .aarch64,
        .wasm32, .wasm64 => .wasm,
        else => .generic,
    };

    return .{
        .vector_size = VectorSize,
        .has_simd = VectorSize > 1,
        .arch = arch,
    };
}

/// Matrix multiplication with blocking/tiling for cache efficiency and SIMD acceleration
/// Computes result[m][n] = a[m][k] * b[k][n]
/// @param a Matrix A (size m x k, row-major order)
/// @param b Matrix B (size k x n, row-major order)
/// @param result Output matrix (size m x n, caller-owned, must be pre-zeroed or will be overwritten)
/// @param m Number of rows in A and result
/// @param n Number of columns in B and result
/// @param k Number of columns in A, rows in B (must match)
pub fn matrixMultiply(
    a: []const f32,
    b: []const f32,
    result: []f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    std.debug.assert(m > 0 and n > 0 and k > 0);
    std.debug.assert(a.len == m * k);
    std.debug.assert(b.len == k * n);
    std.debug.assert(result.len == m * n);

    @memset(result, 0);

    const BLOCK_SIZE = 32;
    var i: usize = 0;

    while (i < m) : (i += BLOCK_SIZE) {
        const i_end = @min(i + BLOCK_SIZE, m);
        var j: usize = 0;
        while (j < n) : (j += BLOCK_SIZE) {
            const j_end = @min(j + BLOCK_SIZE, n);
            var l: usize = 0;
            while (l < k) : (l += BLOCK_SIZE) {
                const l_end = @min(l + BLOCK_SIZE, k);

                var ii: usize = i;
                while (ii < i_end) : (ii += 1) {
                    // SIMD-vectorized inner loop: process VectorSize columns at a time
                    var jj: usize = j;
                    if (comptime VectorSize > 1) {
                        const Vec = @Vector(VectorSize, f32);
                        while (jj + VectorSize <= j_end) : (jj += VectorSize) {
                            var vec_sum: Vec = result[ii * n + jj ..][0..VectorSize].*;
                            var ll: usize = l;
                            while (ll < l_end) : (ll += 1) {
                                const a_val: Vec = @splat(a[ii * k + ll]);
                                const b_vec: Vec = b[ll * n + jj ..][0..VectorSize].*;
                                vec_sum += a_val * b_vec;
                            }
                            result[ii * n + jj ..][0..VectorSize].* = vec_sum;
                        }
                    }
                    // Scalar fallback for remaining columns
                    while (jj < j_end) : (jj += 1) {
                        var acc: f32 = result[ii * n + jj];
                        var ll: usize = l;
                        while (ll < l_end) : (ll += 1) {
                            acc += a[ii * k + ll] * b[ll * n + jj];
                        }
                        result[ii * n + jj] = acc;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Integer Vector Operations (Zig 0.16 @Vector)
// ============================================================================

/// Vector size for i32 operations
const VectorSizeI32 = std.simd.suggestVectorLength(i32) orelse 4;

/// Vector size for u8 operations (useful for quantization)
const VectorSizeU8 = std.simd.suggestVectorLength(u8) orelse 16;

/// SIMD integer addition
pub fn vectorAddI32(a: []const i32, b: []const i32, result: []i32) void {
    std.debug.assert(a.len > 0);
    std.debug.assert(a.len == b.len and a.len == result.len);

    var i: usize = 0;
    if (comptime VectorSizeI32 > 1) {
        const Vec = @Vector(VectorSizeI32, i32);
        while (i + VectorSizeI32 <= a.len) : (i += VectorSizeI32) {
            const va: Vec = a[i..][0..VectorSizeI32].*;
            const vb: Vec = b[i..][0..VectorSizeI32].*;
            result[i..][0..VectorSizeI32].* = va + vb;
        }
    }
    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// SIMD integer sum reduction
pub fn sumI32(data: []const i32) i64 {
    if (data.len == 0) return 0;

    var i: usize = 0;
    var total: i64 = 0;

    if (comptime VectorSizeI32 > 1) {
        const Vec = @Vector(VectorSizeI32, i32);
        var sum_vec: Vec = @splat(0);

        while (i + VectorSizeI32 <= data.len) : (i += VectorSizeI32) {
            const v: Vec = data[i..][0..VectorSizeI32].*;
            sum_vec += v;
        }

        // Horizontal sum using @reduce
        total = @reduce(.Add, sum_vec);
    }

    while (i < data.len) : (i += 1) {
        total += data[i];
    }

    return total;
}

/// SIMD max for i32
pub fn maxI32(data: []const i32) i32 {
    if (data.len == 0) return std.math.minInt(i32);

    var i: usize = 0;
    var max_val: i32 = data[0];

    if (comptime VectorSizeI32 > 1) {
        const Vec = @Vector(VectorSizeI32, i32);
        var max_vec: Vec = @splat(data[0]);

        while (i + VectorSizeI32 <= data.len) : (i += VectorSizeI32) {
            const v: Vec = data[i..][0..VectorSizeI32].*;
            max_vec = @max(max_vec, v);
        }

        max_val = @reduce(.Max, max_vec);
    }

    while (i < data.len) : (i += 1) {
        max_val = @max(max_val, data[i]);
    }

    return max_val;
}

/// SIMD min for i32
pub fn minI32(data: []const i32) i32 {
    if (data.len == 0) return std.math.maxInt(i32);

    var i: usize = 0;
    var min_val: i32 = data[0];

    if (comptime VectorSizeI32 > 1) {
        const Vec = @Vector(VectorSizeI32, i32);
        var min_vec: Vec = @splat(data[0]);

        while (i + VectorSizeI32 <= data.len) : (i += VectorSizeI32) {
            const v: Vec = data[i..][0..VectorSizeI32].*;
            min_vec = @min(min_vec, v);
        }

        min_val = @reduce(.Min, min_vec);
    }

    while (i < data.len) : (i += 1) {
        min_val = @min(min_val, data[i]);
    }

    return min_val;
}

// ============================================================================
// Fused Multiply-Add Operations (FMA)
// ============================================================================

/// Fused multiply-add: result = a * b + c
/// Uses SIMD FMA when available for better precision and performance
pub fn fma(a: []const f32, b: []const f32, c: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and b.len == c.len and c.len == result.len);
    if (a.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= a.len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            const vc: Vec = c[i..][0..VectorSize].*;
            result[i..][0..VectorSize].* = @mulAdd(Vec, va, vb, vc);
        }
    }

    while (i < a.len) : (i += 1) {
        result[i] = @mulAdd(f32, a[i], b[i], c[i]);
    }
}

/// Scalar-vector fused multiply-add: result = scalar * a + b
pub fn fmaScalar(scalar: f32, a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and b.len == result.len);
    if (a.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const s: Vec = @splat(scalar);

        while (i + VectorSize <= a.len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            result[i..][0..VectorSize].* = @mulAdd(Vec, s, va, vb);
        }
    }

    while (i < a.len) : (i += 1) {
        result[i] = @mulAdd(f32, scalar, a[i], b[i]);
    }
}

// ============================================================================
// Vector Scaling Operations
// ============================================================================

/// Multiply vector by scalar in-place
pub fn scaleInPlace(data: []f32, scalar: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const s: Vec = @splat(scalar);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = v * s;
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] *= scalar;
    }
}

/// Add scalar to vector in-place
pub fn addScalarInPlace(data: []f32, scalar: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const s: Vec = @splat(scalar);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = v + s;
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] += scalar;
    }
}

// ============================================================================
// Element-wise Operations
// ============================================================================

/// Element-wise multiplication (Hadamard product)
pub fn hadamard(a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and a.len == result.len);
    if (a.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= a.len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            result[i..][0..VectorSize].* = va * vb;
        }
    }

    while (i < a.len) : (i += 1) {
        result[i] = a[i] * b[i];
    }
}

/// Element-wise absolute value
pub fn absInPlace(data: []f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = @abs(v);
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] = @abs(data[i]);
    }
}

/// Element-wise clamp
pub fn clampInPlace(data: []f32, min_val: f32, max_val: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const min_vec: Vec = @splat(min_val);
        const max_vec: Vec = @splat(max_val);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            data[i..][0..VectorSize].* = @max(min_vec, @min(max_vec, v));
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] = @max(min_val, @min(max_val, data[i]));
    }
}

// ============================================================================
// Comparison Operations (returning masks)
// ============================================================================

/// Count elements greater than threshold
pub fn countGreaterThan(data: []const f32, threshold: f32) usize {
    if (data.len == 0) return 0;

    var count: usize = 0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const thresh: Vec = @splat(threshold);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            const mask = v > thresh;
            // Count true values in mask
            const ones: @Vector(VectorSize, u1) = @bitCast(mask);
            count += @reduce(.Add, @as(@Vector(VectorSize, usize), ones));
        }
    }

    while (i < data.len) : (i += 1) {
        if (data[i] > threshold) count += 1;
    }

    return count;
}

// ============================================================================
// Memory Operations
// ============================================================================

/// Copy with SIMD acceleration
pub fn copyF32(src: []const f32, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    if (src.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);

        while (i + VectorSize <= src.len) : (i += VectorSize) {
            const v: Vec = src[i..][0..VectorSize].*;
            dst[i..][0..VectorSize].* = v;
        }
    }

    while (i < src.len) : (i += 1) {
        dst[i] = src[i];
    }
}

/// Fill array with value using SIMD
pub fn fillF32(data: []f32, value: f32) void {
    if (data.len == 0) return;

    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const v: Vec = @splat(value);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            data[i..][0..VectorSize].* = v;
        }
    }

    while (i < data.len) : (i += 1) {
        data[i] = value;
    }
}

// ============================================================================
// v2.0 SIMD Kernels
// ============================================================================

/// Euclidean distance: sqrt(sum((a[i] - b[i])^2))
/// Single-pass SIMD computation of L2 distance between two vectors.
/// @param a First input vector (must not be empty)
/// @param b Second input vector (must have same length as a)
/// @return Euclidean distance between the two vectors
pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len > 0);
    std.debug.assert(b.len > 0);
    std.debug.assert(a.len == b.len);

    const len = a.len;
    var dist_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var vec_sum: Vec = @splat(0.0);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const va: Vec = a[i..][0..VectorSize].*;
            const vb: Vec = b[i..][0..VectorSize].*;
            const diff = va - vb;
            vec_sum += diff * diff;
        }

        dist_sum += @reduce(.Add, vec_sum);
    }

    while (i < len) : (i += 1) {
        const diff = a[i] - b[i];
        dist_sum += diff * diff;
    }

    return @sqrt(dist_sum);
}

/// Numerically stable softmax with separate output buffer.
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
/// @param data Input logits (must not be empty)
/// @param out Output probabilities (must have same length as data, caller-owned)
pub fn softmax(data: []const f32, out: []f32) void {
    std.debug.assert(data.len > 0);
    std.debug.assert(data.len == out.len);

    // Step 1: Find max for numerical stability
    const max_val = maxValue(data);

    // Step 2: Compute exp(x - max) into output buffer
    var exp_sum: f32 = 0.0;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const max_vec: Vec = @splat(max_val);
        var sum_vec: Vec = @splat(0.0);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const x: Vec = data[i..][0..VectorSize].*;
            const exp_val: Vec = @exp(x - max_vec);
            out[i..][0..VectorSize].* = exp_val;
            sum_vec += exp_val;
        }

        exp_sum += @reduce(.Add, sum_vec);
    }

    while (i < data.len) : (i += 1) {
        const exp_val = @exp(data[i] - max_val);
        out[i] = exp_val;
        exp_sum += exp_val;
    }

    // Step 3: Normalize
    if (exp_sum > 0.0) {
        divideByScalar(out, exp_sum);
    }
}

/// SAXPY: y[i] += a * x[i] (BLAS Level 1)
/// Performs the scalar-alpha-x-plus-y operation in-place on y.
/// @param a Scalar multiplier
/// @param x Input vector (must not be empty)
/// @param y Input/output vector, modified in-place (must have same length as x)
pub fn saxpy(a: f32, x: []const f32, y: []f32) void {
    std.debug.assert(x.len > 0);
    std.debug.assert(x.len == y.len);

    const len = x.len;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const va: Vec = @splat(a);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const vx: Vec = x[i..][0..VectorSize].*;
            const vy: Vec = y[i..][0..VectorSize].*;
            y[i..][0..VectorSize].* = @mulAdd(Vec, va, vx, vy);
        }
    }

    while (i < len) : (i += 1) {
        y[i] = @mulAdd(f32, a, x[i], y[i]);
    }
}

/// Reduce sum: sum all elements in a vector.
/// Alias for `sum` — provided for semantic consistency with `reduceMin`/`reduceMax`.
pub const reduceSum = sum;

/// Reduce min: find the minimum element in a vector.
/// @param data Input vector
/// @return Minimum value, or positive infinity for empty input
pub fn reduceMin(data: []const f32) f32 {
    if (data.len == 0) return std.math.inf(f32);

    var i: usize = 0;
    var result: f32 = data[0];

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var min_vec: Vec = @splat(data[0]);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            min_vec = @min(min_vec, v);
        }

        result = @min(result, @reduce(.Min, min_vec));
    }

    while (i < data.len) : (i += 1) {
        result = @min(result, data[i]);
    }

    return result;
}

/// Reduce max: find the maximum element in a vector.
/// @param data Input vector
/// @return Maximum value, or negative infinity for empty input
pub fn reduceMax(data: []const f32) f32 {
    if (data.len == 0) return -std.math.inf(f32);

    var i: usize = 0;
    var result: f32 = data[0];

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        var max_vec: Vec = @splat(data[0]);

        while (i + VectorSize <= data.len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            max_vec = @max(max_vec, v);
        }

        result = @max(result, @reduce(.Max, max_vec));
    }

    while (i < data.len) : (i += 1) {
        result = @max(result, data[i]);
    }

    return result;
}

/// Scale vector into a separate output buffer: out[i] = data[i] * scalar
/// Unlike scaleInPlace which modifies in-place, this writes to a distinct output slice.
/// @param data Input vector (must not be empty)
/// @param scalar Scalar multiplier
/// @param out Output buffer (must have same length as data, caller-owned)
pub fn scale(data: []const f32, scalar: f32, out: []f32) void {
    std.debug.assert(data.len > 0);
    std.debug.assert(data.len == out.len);

    const len = data.len;
    var i: usize = 0;

    if (comptime VectorSize > 1) {
        const Vec = @Vector(VectorSize, f32);
        const s: Vec = @splat(scalar);

        while (i + VectorSize <= len) : (i += VectorSize) {
            const v: Vec = data[i..][0..VectorSize].*;
            out[i..][0..VectorSize].* = v * s;
        }
    }

    while (i < len) : (i += 1) {
        out[i] = data[i] * scalar;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "vector addition works" {
    var a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var b = [_]f32{ 0.5, 1.5, 2.5, 3.5 };
    var result: [4]f32 = undefined;

    vectorAdd(&a, &b, &result);

    try std.testing.expectEqual(@as(f32, 1.5), result[0]);
    try std.testing.expectEqual(@as(f32, 3.5), result[1]);
    try std.testing.expectEqual(@as(f32, 5.5), result[2]);
    try std.testing.expectEqual(@as(f32, 7.5), result[3]);
}

test "vector dot product works" {
    var a = [_]f32{ 1.0, 2.0, 3.0 };
    var b = [_]f32{ 4.0, 5.0, 6.0 };

    const result = vectorDot(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), result, 1e-6);
}

test "vector L2 norm works" {
    var v = [_]f32{ 3.0, 4.0 };

    const result = vectorL2Norm(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result, 1e-6);
}

test "cosine similarity works" {
    var a = [_]f32{ 1.0, 0.0 };
    var b = [_]f32{ 0.0, 1.0 };

    const result = cosineSimilarity(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result, 1e-6);
}

test "matrix multiplication works" {
    // 2x3 * 3x2 = 2x2
    // A = [1 2 3]   B = [7  8 ]   Result = [58  64 ]
    //     [4 5 6]       [9  10]            [139 154]
    //                   [11 12]
    var a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var result: [4]f32 = undefined;

    matrixMultiply(&a, &b, &result, 2, 2, 3);

    try std.testing.expectApproxEqAbs(@as(f32, 58.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 139.0), result[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 154.0), result[3], 1e-5);
}

test "matrix multiplication larger" {
    // Test with a matrix size that exercises SIMD paths (8x8)
    const size = 8;
    var a: [size * size]f32 = undefined;
    var b: [size * size]f32 = undefined;
    var result: [size * size]f32 = undefined;

    // Initialize with simple values for verification
    for (0..size) |i| {
        for (0..size) |j| {
            a[i * size + j] = @floatFromInt(i + j);
            b[i * size + j] = @floatFromInt(i * j);
        }
    }

    matrixMultiply(&a, &b, &result, size, size, size);

    // Verify result[0][0] = sum(a[0][k] * b[k][0]) for k=0..7
    // = 0*0 + 1*0 + 2*0 + 3*0 + 4*0 + 5*0 + 6*0 + 7*0 = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 1e-4);

    // Verify result[1][1] = sum(a[1][k] * b[k][1]) for k=0..7
    // = 1*0 + 2*1 + 3*2 + 4*3 + 5*4 + 6*5 + 7*6 + 8*7 = 0 + 2 + 6 + 12 + 20 + 30 + 42 + 56 = 168
    try std.testing.expectApproxEqAbs(@as(f32, 168.0), result[size + 1], 1e-4);
}

// ============================================================================
// Tests for Activation Functions
// ============================================================================

test "siluInPlace works" {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    siluInPlace(&data);

    // SiLU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-6);
    // SiLU(x) > 0 for x > 0
    try std.testing.expect(data[3] > 0);
    try std.testing.expect(data[4] > 0);
    // SiLU(x) < 0 for x < 0
    try std.testing.expect(data[0] < 0);
    try std.testing.expect(data[1] < 0);
}

test "geluInPlace works" {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    geluInPlace(&data);

    // GELU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[2], 1e-6);
    // GELU(x) > 0 for x > 0
    try std.testing.expect(data[3] > 0);
    try std.testing.expect(data[4] > 0);
}

test "reluInPlace works" {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    reluInPlace(&data);

    try std.testing.expectEqual(@as(f32, 0.0), data[0]);
    try std.testing.expectEqual(@as(f32, 0.0), data[1]);
    try std.testing.expectEqual(@as(f32, 0.0), data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), data[3]);
    try std.testing.expectEqual(@as(f32, 2.0), data[4]);
}

test "leakyReluInPlace works" {
    var data = [_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 };
    leakyReluInPlace(&data, 0.1);

    try std.testing.expectApproxEqAbs(@as(f32, -0.2), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.1), data[1], 1e-6);
    try std.testing.expectEqual(@as(f32, 0.0), data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), data[3]);
    try std.testing.expectEqual(@as(f32, 2.0), data[4]);
}

// ============================================================================
// Tests for Softmax Operations
// ============================================================================

test "softmaxInPlace works" {
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    softmaxInPlace(&data);

    // Sum should be 1.0
    var total: f32 = 0.0;
    for (data) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);

    // Values should be in ascending order (since inputs were)
    try std.testing.expect(data[0] < data[1]);
    try std.testing.expect(data[1] < data[2]);

    // All values should be positive
    for (data) |v| {
        try std.testing.expect(v > 0);
    }
}

test "maxValue works" {
    const data = [_]f32{ 1.0, 5.0, 3.0, 2.0, 4.0 };
    const max_val = maxValue(&data);
    try std.testing.expectEqual(@as(f32, 5.0), max_val);
}

test "sum works" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const total = sum(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), total, 1e-6);
}

// ============================================================================
// Tests for Normalization Operations
// ============================================================================

test "rmsNormInPlace works" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const original_sq_sum = squaredSum(&data);
    rmsNormInPlace(&data, null, 1e-6);

    // After RMSNorm, the RMS should be approximately 1
    const new_sq_sum = squaredSum(&data);
    const new_rms = @sqrt(new_sq_sum / 4.0);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), new_rms, 1e-4);

    // Verify original was not 1 (sanity check)
    try std.testing.expect(original_sq_sum > 1.0);
}

test "squaredSum works" {
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const result = squaredSum(&data);
    // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
    try std.testing.expectApproxEqAbs(@as(f32, 14.0), result, 1e-6);
}

// ============================================================================
// Tests for Distance Calculations
// ============================================================================

test "l2DistanceSquared works" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 2.0, 2.0 };
    const dist_sq = l2DistanceSquared(&a, &b);
    // sqrt(1 + 4 + 4) = sqrt(9) = 3, so squared = 9
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), dist_sq, 1e-6);
}

test "l2Distance works" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 2.0, 2.0 };
    const dist = l2Distance(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), dist, 1e-6);
}

test "SIMD activation large array" {
    // Test with larger array to exercise SIMD paths
    var data: [64]f32 = undefined;
    for (0..64) |i| {
        data[i] = @as(f32, @floatFromInt(i)) / 32.0 - 1.0; // Range: -1 to 1
    }

    var silu_data = data;
    siluInPlace(&silu_data);

    var gelu_data = data;
    geluInPlace(&gelu_data);

    var relu_data = data;
    reluInPlace(&relu_data);

    // Verify all outputs are reasonable (no NaN/Inf)
    for (silu_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
    for (gelu_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
    for (relu_data) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

// ============================================================================
// Tests for Integer SIMD Operations
// ============================================================================

test "integer addition works" {
    const a = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const b = [_]i32{ 10, 20, 30, 40, 50, 60, 70, 80 };
    var result: [8]i32 = undefined;

    vectorAddI32(&a, &b, &result);

    try std.testing.expectEqual(@as(i32, 11), result[0]);
    try std.testing.expectEqual(@as(i32, 22), result[1]);
    try std.testing.expectEqual(@as(i32, 88), result[7]);
}

test "integer sum works" {
    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const total = sumI32(&data);
    try std.testing.expectEqual(@as(i64, 36), total);
}

test "integer max works" {
    const data = [_]i32{ 1, 5, 3, 9, 2, 8, 4, 7 };
    const max_val = maxI32(&data);
    try std.testing.expectEqual(@as(i32, 9), max_val);
}

test "integer min works" {
    const data = [_]i32{ 5, 3, 9, 1, 8, 4, 7, 2 };
    const min_val = minI32(&data);
    try std.testing.expectEqual(@as(i32, 1), min_val);
}

// ============================================================================
// Tests for FMA Operations
// ============================================================================

test "fma works" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const c = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var result: [4]f32 = undefined;

    fma(&a, &b, &c, &result);

    // a*b+c = [2.5, 6.5, 12.5, 20.5]
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.5), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.5), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 20.5), result[3], 1e-6);
}

test "fmaScalar works" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    var result: [4]f32 = undefined;

    fmaScalar(2.0, &a, &b, &result);

    // 2*a+b = [2.5, 4.5, 6.5, 8.5]
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.5), result[1], 1e-6);
}

// ============================================================================
// Tests for Scaling Operations
// ============================================================================

test "scaleInPlace works" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    scaleInPlace(&data, 2.0);

    try std.testing.expectEqual(@as(f32, 2.0), data[0]);
    try std.testing.expectEqual(@as(f32, 4.0), data[1]);
    try std.testing.expectEqual(@as(f32, 6.0), data[2]);
    try std.testing.expectEqual(@as(f32, 8.0), data[3]);
}

test "addScalarInPlace works" {
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    addScalarInPlace(&data, 10.0);

    try std.testing.expectEqual(@as(f32, 11.0), data[0]);
    try std.testing.expectEqual(@as(f32, 12.0), data[1]);
}

// ============================================================================
// Tests for Element-wise Operations
// ============================================================================

test "hadamard product works" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    var result: [4]f32 = undefined;

    hadamard(&a, &b, &result);

    try std.testing.expectEqual(@as(f32, 2.0), result[0]);
    try std.testing.expectEqual(@as(f32, 6.0), result[1]);
    try std.testing.expectEqual(@as(f32, 12.0), result[2]);
    try std.testing.expectEqual(@as(f32, 20.0), result[3]);
}

test "absInPlace works" {
    var data = [_]f32{ -1.0, 2.0, -3.0, 4.0 };
    absInPlace(&data);

    try std.testing.expectEqual(@as(f32, 1.0), data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), data[1]);
    try std.testing.expectEqual(@as(f32, 3.0), data[2]);
    try std.testing.expectEqual(@as(f32, 4.0), data[3]);
}

test "clampInPlace works" {
    var data = [_]f32{ -5.0, 0.5, 1.5, 10.0 };
    clampInPlace(&data, 0.0, 1.0);

    try std.testing.expectEqual(@as(f32, 0.0), data[0]);
    try std.testing.expectEqual(@as(f32, 0.5), data[1]);
    try std.testing.expectEqual(@as(f32, 1.0), data[2]);
    try std.testing.expectEqual(@as(f32, 1.0), data[3]);
}

test "countGreaterThan works" {
    const data = [_]f32{ 0.1, 0.6, 0.3, 0.8, 0.2, 0.9, 0.4, 0.7 };
    const count = countGreaterThan(&data, 0.5);
    try std.testing.expectEqual(@as(usize, 4), count);
}

// ============================================================================
// Tests for Memory Operations
// ============================================================================

test "copyF32 works" {
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var dst: [4]f32 = undefined;

    copyF32(&src, &dst);

    try std.testing.expectEqual(@as(f32, 1.0), dst[0]);
    try std.testing.expectEqual(@as(f32, 4.0), dst[3]);
}

test "fillF32 works" {
    var data: [8]f32 = undefined;
    fillF32(&data, 3.14);

    for (data) |v| {
        try std.testing.expectApproxEqAbs(@as(f32, 3.14), v, 1e-6);
    }
}

// ============================================================================
// Tests for v2.0 SIMD Kernels
// ============================================================================

test "euclideanDistance works" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 2.0, 2.0 };
    const dist = euclideanDistance(&a, &b);
    // sqrt(1 + 4 + 4) = sqrt(9) = 3
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), dist, 1e-6);
}

test "euclideanDistance identical vectors is zero" {
    const a = [_]f32{ 3.0, 4.0, 5.0, 6.0 };
    const dist = euclideanDistance(&a, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dist, 1e-6);
}

test "euclideanDistance large array" {
    var a: [64]f32 = undefined;
    var b: [64]f32 = undefined;
    for (0..64) |i| {
        a[i] = @floatFromInt(i);
        b[i] = @as(f32, @floatFromInt(i)) + 1.0;
    }
    const dist = euclideanDistance(&a, &b);
    // Each diff is 1.0, so sum of squares = 64, sqrt(64) = 8
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), dist, 1e-5);
}

test "softmax output sums to one" {
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    var out: [3]f32 = undefined;
    softmax(&data, &out);

    var total: f32 = 0.0;
    for (out) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);

    // Values should be in ascending order (since inputs were)
    try std.testing.expect(out[0] < out[1]);
    try std.testing.expect(out[1] < out[2]);

    // All values should be positive
    for (out) |v| {
        try std.testing.expect(v > 0);
    }
}

test "softmax numerical stability with large values" {
    const data = [_]f32{ 1000.0, 1001.0, 1002.0 };
    var out: [3]f32 = undefined;
    softmax(&data, &out);

    var total: f32 = 0.0;
    for (out) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);

    // No NaN or Inf
    for (out) |v| {
        try std.testing.expect(!std.math.isNan(v));
        try std.testing.expect(!std.math.isInf(v));
    }
}

test "saxpy basic operation" {
    var y = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const x = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    saxpy(2.0, &x, &y);

    // y = 2*x + y = [21, 42, 63, 84]
    try std.testing.expectApproxEqAbs(@as(f32, 21.0), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), y[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 63.0), y[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 84.0), y[3], 1e-6);
}

test "saxpy with zero alpha" {
    const original = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var y = original;
    const x = [_]f32{ 10.0, 20.0, 30.0, 40.0 };
    saxpy(0.0, &x, &y);

    // y should be unchanged
    for (y, original) |actual, expected| {
        try std.testing.expectEqual(expected, actual);
    }
}

test "saxpy large array" {
    var y: [64]f32 = undefined;
    var x: [64]f32 = undefined;
    for (0..64) |i| {
        x[i] = 1.0;
        y[i] = @floatFromInt(i);
    }
    saxpy(3.0, &x, &y);

    for (0..64) |i| {
        const expected = @as(f32, @floatFromInt(i)) + 3.0;
        try std.testing.expectApproxEqAbs(expected, y[i], 1e-5);
    }
}

test "reduceSum works" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const total = reduceSum(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 15.0), total, 1e-6);
}

test "reduceSum empty returns zero" {
    const data = [_]f32{};
    const total = reduceSum(&data);
    try std.testing.expectEqual(@as(f32, 0.0), total);
}

test "reduceSum large array" {
    var data: [64]f32 = undefined;
    for (0..64) |i| {
        data[i] = 1.0;
    }
    const total = reduceSum(&data);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), total, 1e-4);
}

test "reduceMin works" {
    const data = [_]f32{ 5.0, 3.0, 9.0, 1.0, 7.0 };
    const result = reduceMin(&data);
    try std.testing.expectEqual(@as(f32, 1.0), result);
}

test "reduceMin empty returns positive infinity" {
    const data = [_]f32{};
    const result = reduceMin(&data);
    try std.testing.expect(std.math.isInf(result));
    try std.testing.expect(result > 0);
}

test "reduceMin negative values" {
    const data = [_]f32{ -1.0, -5.0, -2.0, -3.0 };
    const result = reduceMin(&data);
    try std.testing.expectEqual(@as(f32, -5.0), result);
}

test "reduceMax works" {
    const data = [_]f32{ 5.0, 3.0, 9.0, 1.0, 7.0 };
    const result = reduceMax(&data);
    try std.testing.expectEqual(@as(f32, 9.0), result);
}

test "reduceMax empty returns negative infinity" {
    const data = [_]f32{};
    const result = reduceMax(&data);
    try std.testing.expect(std.math.isInf(result));
    try std.testing.expect(result < 0);
}

test "reduceMax negative values" {
    const data = [_]f32{ -1.0, -5.0, -2.0, -3.0 };
    const result = reduceMax(&data);
    try std.testing.expectEqual(@as(f32, -1.0), result);
}

test "scale out-of-place works" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    scale(&data, 3.0, &out);

    try std.testing.expectEqual(@as(f32, 3.0), out[0]);
    try std.testing.expectEqual(@as(f32, 6.0), out[1]);
    try std.testing.expectEqual(@as(f32, 9.0), out[2]);
    try std.testing.expectEqual(@as(f32, 12.0), out[3]);
}

test "scale preserves original" {
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var out: [4]f32 = undefined;
    scale(&data, 5.0, &out);

    // Original should be unchanged
    try std.testing.expectEqual(@as(f32, 1.0), data[0]);
    try std.testing.expectEqual(@as(f32, 2.0), data[1]);
}

test "scale large array" {
    var data: [64]f32 = undefined;
    var out: [64]f32 = undefined;
    for (0..64) |i| {
        data[i] = @floatFromInt(i);
    }
    scale(&data, 2.0, &out);

    for (0..64) |i| {
        const expected = @as(f32, @floatFromInt(i)) * 2.0;
        try std.testing.expectApproxEqAbs(expected, out[i], 1e-5);
    }
}
