/// SIMD-accelerated operations for high-performance computing
// Simple utility functions for tests and basic API
// Distance between two equal-length slices
pub fn distance(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) {
        @panic("distance requires equal-length slices");
    }
    var sum: f32 = 0.0;
    for (a, b) |va, vb| {
        const diff = va - vb;
        sum += diff * diff;
    }
    return std.math.sqrt(sum);
}

// Cosine similarity between two equal-length slices
pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
    if (a.len != b.len) {
        @panic("cosineSimilarity requires equal-length slices");
    }
    var dot: f32 = 0.0;
    var normA: f32 = 0.0;
    var normB: f32 = 0.0;
    for (a, b) |va, vb| {
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }
    const denom = std.math.sqrt(normA) * std.math.sqrt(normB);
    if (denom == 0.0) return 0.0;
    return dot / denom;
}

// Dummy performance monitor struct for tests
pub const PerformanceMonitor = struct {};
pub fn getPerformanceMonitor() PerformanceMonitor { return .{}; }
pub const SIMDOpts = struct {
    // Add SIMD options here
    // TODO: Implement SIMD options
};
//! Provides vectorized operations for AI/ML workloads

const std = @import("std");

/// SIMD vector width (4 for most architectures, can be optimized per platform)
const SIMD_WIDTH = 4;

/// SIMD-accelerated vector operations
pub const VectorOps = struct {
    /// SIMD-accelerated dot product
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0.0;
        var i: usize = 0;

        // SIMD processing in chunks of SIMD_WIDTH
        while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
            const va = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(a[i .. i + SIMD_WIDTH].ptr)).*);
            const vb = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(b[i .. i + SIMD_WIDTH].ptr)).*);
            const product = va * vb;
            sum += @reduce(.Add, product);
        }

        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// SIMD-accelerated matrix-vector multiplication
    pub fn matrixVectorMultiply(matrix: []const f32, vector: []const f32, result: []f32, rows: usize, cols: usize) void {
        for (0..rows) |i| {
            const row_start = i * cols;
            result[i] = dotProduct(matrix[row_start .. row_start + cols], vector);
        }
    }

    /// SIMD-accelerated ReLU activation
    pub fn vectorizedRelu(data: []f32) void {
        var i: usize = 0;

        // SIMD processing
        while (i + SIMD_WIDTH <= data.len) : (i += SIMD_WIDTH) {
            var vec = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(data[i .. i + SIMD_WIDTH].ptr)).*);
            vec = @max(vec, @as(@Vector(SIMD_WIDTH, f32), @splat(0.0)));
            @memcpy(data[i .. i + SIMD_WIDTH], &@as([SIMD_WIDTH]f32, vec));
        }

        // Handle remaining elements
        while (i < data.len) : (i += 1) {
            data[i] = @max(data[i], 0.0);
        }
    }

    /// SIMD-accelerated Leaky ReLU activation
    pub fn vectorizedLeakyRelu(data: []f32, slope: f32) void {
        var i: usize = 0;

        // SIMD processing
        while (i + SIMD_WIDTH <= data.len) : (i += SIMD_WIDTH) {
            var vec = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(data[i .. i + SIMD_WIDTH].ptr)).*);

            // Create mask for negative values
            const zero_vec = @as(@Vector(SIMD_WIDTH, f32), @splat(0.0));
            const slope_vec = @as(@Vector(SIMD_WIDTH, f32), @splat(slope));

            // Apply: max(x, x * slope) for x < 0, x for x >= 0
            const mask = vec < zero_vec;
            const leaky_part = vec * slope_vec;
            vec = @select(f32, mask, leaky_part, vec);

            @memcpy(data[i .. i + SIMD_WIDTH], &@as([SIMD_WIDTH]f32, vec));
        }

        // Handle remaining elements
        while (i < data.len) : (i += 1) {
            data[i] = if (data[i] > 0.0) data[i] else slope * data[i];
        }
    }

    /// SIMD-accelerated element-wise maximum
    pub fn vectorMax(a: []const f32, b: []const f32, result: []f32) void {
        var i: usize = 0;

        while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
            const va = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(a[i .. i + SIMD_WIDTH].ptr)).*);
            const vb = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(b[i .. i + SIMD_WIDTH].ptr)).*);
            const vmax = @max(va, vb);
            @memcpy(result[i .. i + SIMD_WIDTH], &@as([SIMD_WIDTH]f32, vmax));
        }

        while (i < a.len) : (i += 1) {
            result[i] = @max(a[i], b[i]);
        }
    }

    /// SIMD-accelerated element-wise addition
    pub fn vectorAdd(a: []const f32, b: []const f32, result: []f32) void {
        var i: usize = 0;

        while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
            const va = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(a[i .. i + SIMD_WIDTH].ptr)).*);
            const vb = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(b[i .. i + SIMD_WIDTH].ptr)).*);
            const vsum = va + vb;
            @memcpy(result[i .. i + SIMD_WIDTH], &@as([SIMD_WIDTH]f32, vsum));
        }

        while (i < a.len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    /// SIMD-accelerated element-wise multiplication
    pub fn vectorMul(a: []const f32, b: []const f32, result: []f32) void {
        var i: usize = 0;

        while (i + SIMD_WIDTH <= a.len) : (i += SIMD_WIDTH) {
            const va = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(a[i .. i + SIMD_WIDTH].ptr)).*);
            const vb = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(b[i .. i + SIMD_WIDTH].ptr)).*);
            const vprod = va * vb;
            @memcpy(result[i .. i + SIMD_WIDTH], &@as([SIMD_WIDTH]f32, vprod));
        }

        while (i < a.len) : (i += 1) {
            result[i] = a[i] * b[i];
        }
    }

    /// SIMD-accelerated vector normalization (L2 norm)
    pub fn normalize(data: []f32, result: []f32) void {
        // Calculate L2 norm (magnitude)
        var sum_sq: f32 = 0.0;
        var i: usize = 0;

        // SIMD sum of squares
        while (i + SIMD_WIDTH <= data.len) : (i += SIMD_WIDTH) {
            const vec = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(data[i .. i + SIMD_WIDTH].ptr)).*);
            const sq = vec * vec;
            sum_sq += @reduce(.Add, sq);
        }

        // Handle remaining elements
        while (i < data.len) : (i += 1) {
            sum_sq += data[i] * data[i];
        }

        const magnitude = std.math.sqrt(sum_sq);
        if (magnitude == 0.0) return; // Avoid division by zero

        const inv_magnitude = 1.0 / magnitude;
        const scale_vec = @as(@Vector(SIMD_WIDTH, f32), @splat(inv_magnitude));

        i = 0;
        // SIMD normalization
        while (i + SIMD_WIDTH <= data.len) : (i += SIMD_WIDTH) {
            const vec = @as(@Vector(SIMD_WIDTH, f32), @as(*const [SIMD_WIDTH]f32, @ptrCast(data[i .. i + SIMD_WIDTH].ptr)).*) * scale_vec;
            @memcpy(result[i .. i + SIMD_WIDTH], &@as([SIMD_WIDTH]f32, vec));
        }

        // Handle remaining elements
        while (i < data.len) : (i += 1) {
            result[i] = data[i] * inv_magnitude;
        }
    }

    /// SIMD-accelerated matrix multiplication (basic implementation)
    /// result: output matrix (rows x cols)
    /// a: first matrix (rows x cols)
    /// b: second matrix (cols x cols)
    /// rows: number of rows in result
    /// cols: number of columns in result and rows in b
    /// inner_dim: number of columns in a and rows in b
    pub fn matrixMultiply(result: []f32, a: []const f32, b: []const f32, rows: usize, cols: usize, inner_dim: usize) void {
        // Basic matrix multiplication - can be optimized with SIMD
        for (0..rows) |i| {
            for (0..cols) |j| {
                var sum: f32 = 0.0;
                for (0..inner_dim) |k| {
                    sum += a[i * inner_dim + k] * b[k * cols + j];
                }
                result[i * cols + j] = sum;
            }
        }
        // TODO: Implement SIMD-accelerated matrix multiplication
    }

    /// Check if SIMD is available and beneficial for given size
    pub fn shouldUseSimd(size: usize) bool {
        return size >= SIMD_WIDTH * 2; // Only use SIMD for reasonably sized arrays
    }
};
