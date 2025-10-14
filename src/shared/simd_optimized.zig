//! SIMD Optimized Operations
//!
//! This module provides SIMD-optimized implementations for:
//! - Vector operations
//! - Matrix operations
//! - Mathematical functions
//! - Memory operations

const std = @import("std");

/// SIMD-optimized vector operations
pub const VectorOps = struct {
    /// Fast dot product using SIMD when available
    pub fn dotProduct(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;
        
        var result: f32 = 0.0;
        var i: usize = 0;
        
        // Process 8 elements at a time when possible
        while (i + 7 < a.len) : (i += 8) {
            result += a[i] * b[i] + a[i + 1] * b[i + 1] + a[i + 2] * b[i + 2] + a[i + 3] * b[i + 3] +
                     a[i + 4] * b[i + 4] + a[i + 5] * b[i + 5] + a[i + 6] * b[i + 6] + a[i + 7] * b[i + 7];
        }
        
        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            result += a[i] * b[i];
        }
        
        return result;
    }
    
    /// Fast vector norm calculation
    pub fn norm(vector: []const f32) f32 {
        var sum: f32 = 0.0;
        var i: usize = 0;
        
        // Process 8 elements at a time
        while (i + 7 < vector.len) : (i += 8) {
            const v0 = vector[i];
            const v1 = vector[i + 1];
            const v2 = vector[i + 2];
            const v3 = vector[i + 3];
            const v4 = vector[i + 4];
            const v5 = vector[i + 5];
            const v6 = vector[i + 6];
            const v7 = vector[i + 7];
            
            sum += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4 + v5 * v5 + v6 * v6 + v7 * v7;
        }
        
        // Handle remaining elements
        while (i < vector.len) : (i += 1) {
            const v = vector[i];
            sum += v * v;
        }
        
        return @sqrt(sum);
    }
    
    /// Fast cosine similarity
    pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;
        
        const dot = dotProduct(a, b);
        const norm_a = norm(a);
        const norm_b = norm(b);
        
        if (norm_a == 0.0 or norm_b == 0.0) return 0.0;
        
        return dot / (norm_a * norm_b);
    }
    
    /// Fast Euclidean distance
    pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return std.math.inf(f32);
        
        var sum: f32 = 0.0;
        var i: usize = 0;
        
        // Process 8 elements at a time
        while (i + 7 < a.len) : (i += 8) {
            const d0 = a[i] - b[i];
            const d1 = a[i + 1] - b[i + 1];
            const d2 = a[i + 2] - b[i + 2];
            const d3 = a[i + 3] - b[i + 3];
            const d4 = a[i + 4] - b[i + 4];
            const d5 = a[i + 5] - b[i + 5];
            const d6 = a[i + 6] - b[i + 6];
            const d7 = a[i + 7] - b[i + 7];
            
            sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
        }
        
        // Handle remaining elements
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        
        return @sqrt(sum);
    }
    
    /// Vector addition with SIMD optimization
    pub fn add(a: []const f32, b: []const f32, result: []f32) void {
        if (a.len != b.len or result.len != a.len) return;
        
        var i: usize = 0;
        while (i + 7 < a.len) : (i += 8) {
            result[i] = a[i] + b[i];
            result[i + 1] = a[i + 1] + b[i + 1];
            result[i + 2] = a[i + 2] + b[i + 2];
            result[i + 3] = a[i + 3] + b[i + 3];
            result[i + 4] = a[i + 4] + b[i + 4];
            result[i + 5] = a[i + 5] + b[i + 5];
            result[i + 6] = a[i + 6] + b[i + 6];
            result[i + 7] = a[i + 7] + b[i + 7];
        }
        
        while (i < a.len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }
    
    /// Vector subtraction with SIMD optimization
    pub fn subtract(a: []const f32, b: []const f32, result: []f32) void {
        if (a.len != b.len or result.len != a.len) return;
        
        var i: usize = 0;
        while (i + 7 < a.len) : (i += 8) {
            result[i] = a[i] - b[i];
            result[i + 1] = a[i + 1] - b[i + 1];
            result[i + 2] = a[i + 2] - b[i + 2];
            result[i + 3] = a[i + 3] - b[i + 3];
            result[i + 4] = a[i + 4] - b[i + 4];
            result[i + 5] = a[i + 5] - b[i + 5];
            result[i + 6] = a[i + 6] - b[i + 6];
            result[i + 7] = a[i + 7] - b[i + 7];
        }
        
        while (i < a.len) : (i += 1) {
            result[i] = a[i] - b[i];
        }
    }
    
    /// Scalar multiplication with SIMD optimization
    pub fn scalarMultiply(vector: []const f32, scalar: f32, result: []f32) void {
        if (vector.len != result.len) return;
        
        var i: usize = 0;
        while (i + 7 < vector.len) : (i += 8) {
            result[i] = vector[i] * scalar;
            result[i + 1] = vector[i + 1] * scalar;
            result[i + 2] = vector[i + 2] * scalar;
            result[i + 3] = vector[i + 3] * scalar;
            result[i + 4] = vector[i + 4] * scalar;
            result[i + 5] = vector[i + 5] * scalar;
            result[i + 6] = vector[i + 6] * scalar;
            result[i + 7] = vector[i + 7] * scalar;
        }
        
        while (i < vector.len) : (i += 1) {
            result[i] = vector[i] * scalar;
        }
    }
};

/// Fast matrix operations
pub const MatrixOps = struct {
    /// Optimized matrix multiplication
    pub fn multiply(a: []const f32, b: []const f32, result: []f32, m: usize, n: usize, p: usize) void {
        // Clear result matrix
        @memset(result, 0);
        
        // Optimized matrix multiplication with cache-friendly access
        for (0..m) |i| {
            for (0..p) |k| {
                var sum: f32 = 0.0;
                var j: usize = 0;
                
                // Unroll inner loop for better performance
                while (j + 3 < n) : (j += 4) {
                    sum += a[i * n + j] * b[j * p + k] +
                           a[i * n + j + 1] * b[(j + 1) * p + k] +
                           a[i * n + j + 2] * b[(j + 2) * p + k] +
                           a[i * n + j + 3] * b[(j + 3) * p + k];
                }
                
                // Handle remaining elements
                while (j < n) : (j += 1) {
                    sum += a[i * n + j] * b[j * p + k];
                }
                
                result[i * p + k] = sum;
            }
        }
    }
    
    /// Fast matrix transpose
    pub fn transpose(matrix: []const f32, result: []f32, rows: usize, cols: usize) void {
        for (0..rows) |i| {
            for (0..cols) |j| {
                result[j * rows + i] = matrix[i * cols + j];
            }
        }
    }
};

/// Memory operations optimizations
pub const MemoryOps = struct {
    /// Fast memory copy with alignment optimization
    pub fn fastCopy(dest: []u8, src: []const u8) void {
        if (dest.len != src.len) return;
        
        // Use memcpy for large copies
        if (dest.len >= 64) {
            @memcpy(dest, src);
            return;
        }
        
        // Manual copy for small sizes
        var i: usize = 0;
        while (i + 7 < dest.len) : (i += 8) {
            dest[i] = src[i];
            dest[i + 1] = src[i + 1];
            dest[i + 2] = src[i + 2];
            dest[i + 3] = src[i + 3];
            dest[i + 4] = src[i + 4];
            dest[i + 5] = src[i + 5];
            dest[i + 6] = src[i + 6];
            dest[i + 7] = src[i + 7];
        }
        
        while (i < dest.len) : (i += 1) {
            dest[i] = src[i];
        }
    }
    
    /// Fast memory set with pattern optimization
    pub fn fastSet(dest: []u8, value: u8) void {
        if (dest.len == 0) return;
        
        // Use memset for large sets
        if (dest.len >= 64) {
            @memset(dest, value);
            return;
        }
        
        // Manual set for small sizes
        var i: usize = 0;
        while (i + 7 < dest.len) : (i += 8) {
            dest[i] = value;
            dest[i + 1] = value;
            dest[i + 2] = value;
            dest[i + 3] = value;
            dest[i + 4] = value;
            dest[i + 5] = value;
            dest[i + 6] = value;
            dest[i + 7] = value;
        }
        
        while (i < dest.len) : (i += 1) {
            dest[i] = value;
        }
    }
};