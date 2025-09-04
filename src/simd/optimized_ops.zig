//! Optimized SIMD operations for WDBX-AI vector database

const std = @import("std");
const builtin = @import("builtin");
const core = @import("../core/mod.zig");

/// CPU feature detection
pub const CpuFeatures = struct {
    has_sse2: bool,
    has_avx: bool,
    has_avx2: bool,
    has_avx512: bool,
    has_neon: bool,
    
    pub fn detect() CpuFeatures {
        return CpuFeatures{
            .has_sse2 = std.Target.x86.featureSetHas(builtin.cpu.features, .sse2),
            .has_avx = std.Target.x86.featureSetHas(builtin.cpu.features, .avx),
            .has_avx2 = std.Target.x86.featureSetHas(builtin.cpu.features, .avx2),
            .has_avx512 = std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f),
            .has_neon = builtin.cpu.arch == .aarch64,
        };
    }
    
    pub fn getOptimalVectorSize(self: CpuFeatures) usize {
        if (self.has_avx512) return 16;
        if (self.has_avx2 or self.has_avx) return 8;
        if (self.has_sse2 or self.has_neon) return 4;
        return 1;
    }
    
    pub fn print(self: CpuFeatures) !void {
        const stdout = std.io.getStdOut().writer();
        try stdout.print("CPU Features:\n");
        try stdout.print("  SSE2: {}\n", .{self.has_sse2});
        try stdout.print("  AVX: {}\n", .{self.has_avx});
        try stdout.print("  AVX2: {}\n", .{self.has_avx2});
        try stdout.print("  AVX-512: {}\n", .{self.has_avx512});
        try stdout.print("  NEON: {}\n", .{self.has_neon});
        try stdout.print("  Optimal Vector Size: {}\n", .{self.getOptimalVectorSize()});
    }
};

/// Optimized distance calculations
pub const DistanceOps = struct {
    /// Calculate Euclidean distance with optimal SIMD
    pub fn euclideanDistance(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return std.math.inf(f32);
        if (a.len == 0) return 0.0;
        
        const features = CpuFeatures.detect();
        const optimal_size = features.getOptimalVectorSize();
        
        return switch (optimal_size) {
            16 => euclideanDistanceSimd16(a, b),
            8 => euclideanDistanceSimd8(a, b),
            4 => euclideanDistanceSimd4(a, b),
            else => euclideanDistanceScalar(a, b),
        };
    }
    
    /// Calculate cosine similarity with optimal SIMD
    pub fn cosineSimilarity(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return 0.0;
        if (a.len == 0) return 1.0;
        
        const features = CpuFeatures.detect();
        const optimal_size = features.getOptimalVectorSize();
        
        return switch (optimal_size) {
            16 => cosineSimilaritySimd16(a, b),
            8 => cosineSimilaritySimd8(a, b),
            4 => cosineSimilaritySimd4(a, b),
            else => cosineSimilarityScalar(a, b),
        };
    }
    
    /// Calculate Manhattan distance with optimal SIMD
    pub fn manhattanDistance(a: []const f32, b: []const f32) f32 {
        if (a.len != b.len) return std.math.inf(f32);
        if (a.len == 0) return 0.0;
        
        const features = CpuFeatures.detect();
        const optimal_size = features.getOptimalVectorSize();
        
        return switch (optimal_size) {
            16 => manhattanDistanceSimd16(a, b),
            8 => manhattanDistanceSimd8(a, b),
            4 => manhattanDistanceSimd4(a, b),
            else => manhattanDistanceScalar(a, b),
        };
    }
    
    // SIMD implementations for different vector sizes
    fn euclideanDistanceSimd16(a: []const f32, b: []const f32) f32 {
        var sum: @Vector(16, f32) = @splat(0.0);
        var i: usize = 0;
        
        // Process 16 elements at a time
        while (i + 16 <= a.len) : (i += 16) {
            const va: @Vector(16, f32) = a[i..i + 16][0..16].*;
            const vb: @Vector(16, f32) = b[i..i + 16][0..16].*;
            const diff = va - vb;
            sum += diff * diff;
        }
        
        // Handle remaining elements
        var scalar_sum: f32 = @reduce(.Add, sum);
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            scalar_sum += diff * diff;
        }
        
        return std.math.sqrt(scalar_sum);
    }
    
    fn euclideanDistanceSimd8(a: []const f32, b: []const f32) f32 {
        var sum: @Vector(8, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 8 <= a.len) : (i += 8) {
            const va: @Vector(8, f32) = a[i..i + 8][0..8].*;
            const vb: @Vector(8, f32) = b[i..i + 8][0..8].*;
            const diff = va - vb;
            sum += diff * diff;
        }
        
        var scalar_sum: f32 = @reduce(.Add, sum);
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            scalar_sum += diff * diff;
        }
        
        return std.math.sqrt(scalar_sum);
    }
    
    fn euclideanDistanceSimd4(a: []const f32, b: []const f32) f32 {
        var sum: @Vector(4, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 4 <= a.len) : (i += 4) {
            const va: @Vector(4, f32) = a[i..i + 4][0..4].*;
            const vb: @Vector(4, f32) = b[i..i + 4][0..4].*;
            const diff = va - vb;
            sum += diff * diff;
        }
        
        var scalar_sum: f32 = @reduce(.Add, sum);
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            scalar_sum += diff * diff;
        }
        
        return std.math.sqrt(scalar_sum);
    }
    
    fn euclideanDistanceScalar(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0.0;
        for (a, b) |va, vb| {
            const diff = va - vb;
            sum += diff * diff;
        }
        return std.math.sqrt(sum);
    }
    
    fn cosineSimilaritySimd16(a: []const f32, b: []const f32) f32 {
        var dot_product: @Vector(16, f32) = @splat(0.0);
        var norm_a: @Vector(16, f32) = @splat(0.0);
        var norm_b: @Vector(16, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 16 <= a.len) : (i += 16) {
            const va: @Vector(16, f32) = a[i..i + 16][0..16].*;
            const vb: @Vector(16, f32) = b[i..i + 16][0..16].*;
            dot_product += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        
        var scalar_dot: f32 = @reduce(.Add, dot_product);
        var scalar_norm_a: f32 = @reduce(.Add, norm_a);
        var scalar_norm_b: f32 = @reduce(.Add, norm_b);
        
        while (i < a.len) : (i += 1) {
            scalar_dot += a[i] * b[i];
            scalar_norm_a += a[i] * a[i];
            scalar_norm_b += b[i] * b[i];
        }
        
        const magnitude = std.math.sqrt(scalar_norm_a) * std.math.sqrt(scalar_norm_b);
        return if (magnitude > 0) scalar_dot / magnitude else 0.0;
    }
    
    fn cosineSimilaritySimd8(a: []const f32, b: []const f32) f32 {
        var dot_product: @Vector(8, f32) = @splat(0.0);
        var norm_a: @Vector(8, f32) = @splat(0.0);
        var norm_b: @Vector(8, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 8 <= a.len) : (i += 8) {
            const va: @Vector(8, f32) = a[i..i + 8][0..8].*;
            const vb: @Vector(8, f32) = b[i..i + 8][0..8].*;
            dot_product += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        
        var scalar_dot: f32 = @reduce(.Add, dot_product);
        var scalar_norm_a: f32 = @reduce(.Add, norm_a);
        var scalar_norm_b: f32 = @reduce(.Add, norm_b);
        
        while (i < a.len) : (i += 1) {
            scalar_dot += a[i] * b[i];
            scalar_norm_a += a[i] * a[i];
            scalar_norm_b += b[i] * b[i];
        }
        
        const magnitude = std.math.sqrt(scalar_norm_a) * std.math.sqrt(scalar_norm_b);
        return if (magnitude > 0) scalar_dot / magnitude else 0.0;
    }
    
    fn cosineSimilaritySimd4(a: []const f32, b: []const f32) f32 {
        var dot_product: @Vector(4, f32) = @splat(0.0);
        var norm_a: @Vector(4, f32) = @splat(0.0);
        var norm_b: @Vector(4, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 4 <= a.len) : (i += 4) {
            const va: @Vector(4, f32) = a[i..i + 4][0..4].*;
            const vb: @Vector(4, f32) = b[i..i + 4][0..4].*;
            dot_product += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        
        var scalar_dot: f32 = @reduce(.Add, dot_product);
        var scalar_norm_a: f32 = @reduce(.Add, norm_a);
        var scalar_norm_b: f32 = @reduce(.Add, norm_b);
        
        while (i < a.len) : (i += 1) {
            scalar_dot += a[i] * b[i];
            scalar_norm_a += a[i] * a[i];
            scalar_norm_b += b[i] * b[i];
        }
        
        const magnitude = std.math.sqrt(scalar_norm_a) * std.math.sqrt(scalar_norm_b);
        return if (magnitude > 0) scalar_dot / magnitude else 0.0;
    }
    
    fn cosineSimilarityScalar(a: []const f32, b: []const f32) f32 {
        var dot_product: f32 = 0.0;
        var norm_a: f32 = 0.0;
        var norm_b: f32 = 0.0;
        
        for (a, b) |va, vb| {
            dot_product += va * vb;
            norm_a += va * va;
            norm_b += vb * vb;
        }
        
        const magnitude = std.math.sqrt(norm_a) * std.math.sqrt(norm_b);
        return if (magnitude > 0) dot_product / magnitude else 0.0;
    }
    
    fn manhattanDistanceSimd16(a: []const f32, b: []const f32) f32 {
        var sum: @Vector(16, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 16 <= a.len) : (i += 16) {
            const va: @Vector(16, f32) = a[i..i + 16][0..16].*;
            const vb: @Vector(16, f32) = b[i..i + 16][0..16].*;
            const diff = va - vb;
            sum += @abs(diff);
        }
        
        var scalar_sum: f32 = @reduce(.Add, sum);
        while (i < a.len) : (i += 1) {
            scalar_sum += @abs(a[i] - b[i]);
        }
        
        return scalar_sum;
    }
    
    fn manhattanDistanceSimd8(a: []const f32, b: []const f32) f32 {
        var sum: @Vector(8, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 8 <= a.len) : (i += 8) {
            const va: @Vector(8, f32) = a[i..i + 8][0..8].*;
            const vb: @Vector(8, f32) = b[i..i + 8][0..8].*;
            const diff = va - vb;
            sum += @abs(diff);
        }
        
        var scalar_sum: f32 = @reduce(.Add, sum);
        while (i < a.len) : (i += 1) {
            scalar_sum += @abs(a[i] - b[i]);
        }
        
        return scalar_sum;
    }
    
    fn manhattanDistanceSimd4(a: []const f32, b: []const f32) f32 {
        var sum: @Vector(4, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 4 <= a.len) : (i += 4) {
            const va: @Vector(4, f32) = a[i..i + 4][0..4].*;
            const vb: @Vector(4, f32) = b[i..i + 4][0..4].*;
            const diff = va - vb;
            sum += @abs(diff);
        }
        
        var scalar_sum: f32 = @reduce(.Add, sum);
        while (i < a.len) : (i += 1) {
            scalar_sum += @abs(a[i] - b[i]);
        }
        
        return scalar_sum;
    }
    
    fn manhattanDistanceScalar(a: []const f32, b: []const f32) f32 {
        var sum: f32 = 0.0;
        for (a, b) |va, vb| {
            sum += @abs(va - vb);
        }
        return sum;
    }
};

/// Optimized matrix operations
pub const MatrixOps = struct {
    /// Matrix multiplication with SIMD optimization
    pub fn multiply(
        a: []const f32,
        b: []const f32,
        result: []f32,
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) void {
        if (rows_a * cols_b != result.len) return;
        if (cols_a * cols_b != b.len) return;
        if (rows_a * cols_a != a.len) return;
        
        const features = CpuFeatures.detect();
        const optimal_size = features.getOptimalVectorSize();
        
        switch (optimal_size) {
            16 => multiplySimd16(a, b, result, rows_a, cols_a, cols_b),
            8 => multiplySimd8(a, b, result, rows_a, cols_a, cols_b),
            4 => multiplySimd4(a, b, result, rows_a, cols_a, cols_b),
            else => multiplyScalar(a, b, result, rows_a, cols_a, cols_b),
        }
    }
    
    fn multiplySimd16(
        a: []const f32,
        b: []const f32,
        result: []f32,
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) void {
        for (0..rows_a) |row| {
            for (0..cols_b) |col| {
                var sum: @Vector(16, f32) = @splat(0.0);
                var k: usize = 0;
                
                while (k + 16 <= cols_a) : (k += 16) {
                    const va: @Vector(16, f32) = a[row * cols_a + k..row * cols_a + k + 16][0..16].*;
                    var vb: @Vector(16, f32) = undefined;
                    
                    for (0..16) |i| {
                        vb[i] = b[(k + i) * cols_b + col];
                    }
                    
                    sum += va * vb;
                }
                
                var scalar_sum: f32 = @reduce(.Add, sum);
                while (k < cols_a) : (k += 1) {
                    scalar_sum += a[row * cols_a + k] * b[k * cols_b + col];
                }
                
                result[row * cols_b + col] = scalar_sum;
            }
        }
    }
    
    fn multiplySimd8(
        a: []const f32,
        b: []const f32,
        result: []f32,
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) void {
        for (0..rows_a) |row| {
            for (0..cols_b) |col| {
                var sum: @Vector(8, f32) = @splat(0.0);
                var k: usize = 0;
                
                while (k + 8 <= cols_a) : (k += 8) {
                    const va: @Vector(8, f32) = a[row * cols_a + k..row * cols_a + k + 8][0..8].*;
                    var vb: @Vector(8, f32) = undefined;
                    
                    for (0..8) |i| {
                        vb[i] = b[(k + i) * cols_b + col];
                    }
                    
                    sum += va * vb;
                }
                
                var scalar_sum: f32 = @reduce(.Add, sum);
                while (k < cols_a) : (k += 1) {
                    scalar_sum += a[row * cols_a + k] * b[k * cols_b + col];
                }
                
                result[row * cols_b + col] = scalar_sum;
            }
        }
    }
    
    fn multiplySimd4(
        a: []const f32,
        b: []const f32,
        result: []f32,
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) void {
        for (0..rows_a) |row| {
            for (0..cols_b) |col| {
                var sum: @Vector(4, f32) = @splat(0.0);
                var k: usize = 0;
                
                while (k + 4 <= cols_a) : (k += 4) {
                    const va: @Vector(4, f32) = a[row * cols_a + k..row * cols_a + k + 4][0..4].*;
                    var vb: @Vector(4, f32) = undefined;
                    
                    for (0..4) |i| {
                        vb[i] = b[(k + i) * cols_b + col];
                    }
                    
                    sum += va * vb;
                }
                
                var scalar_sum: f32 = @reduce(.Add, sum);
                while (k < cols_a) : (k += 1) {
                    scalar_sum += a[row * cols_a + k] * b[k * cols_b + col];
                }
                
                result[row * cols_b + col] = scalar_sum;
            }
        }
    }
    
    fn multiplyScalar(
        a: []const f32,
        b: []const f32,
        result: []f32,
        rows_a: usize,
        cols_a: usize,
        cols_b: usize,
    ) void {
        for (0..rows_a) |row| {
            for (0..cols_b) |col| {
                var sum: f32 = 0.0;
                for (0..cols_a) |k| {
                    sum += a[row * cols_a + k] * b[k * cols_b + col];
                }
                result[row * cols_b + col] = sum;
            }
        }
    }
    
    /// Transpose matrix with SIMD optimization
    pub fn transpose(input: []const f32, output: []f32, rows: usize, cols: usize) void {
        if (input.len != rows * cols or output.len != rows * cols) return;
        
        // Use cache-friendly block transpose for large matrices
        if (rows > 64 and cols > 64) {
            transposeBlocked(input, output, rows, cols);
        } else {
            transposeSimple(input, output, rows, cols);
        }
    }
    
    fn transposeBlocked(input: []const f32, output: []f32, rows: usize, cols: usize) void {
        const block_size = 32;
        
        var row_block: usize = 0;
        while (row_block < rows) : (row_block += block_size) {
            var col_block: usize = 0;
            while (col_block < cols) : (col_block += block_size) {
                const max_row = @min(row_block + block_size, rows);
                const max_col = @min(col_block + block_size, cols);
                
                for (row_block..max_row) |row| {
                    for (col_block..max_col) |col| {
                        output[col * rows + row] = input[row * cols + col];
                    }
                }
            }
        }
    }
    
    fn transposeSimple(input: []const f32, output: []f32, rows: usize, cols: usize) void {
        for (0..rows) |row| {
            for (0..cols) |col| {
                output[col * rows + row] = input[row * cols + col];
            }
        }
    }
};

/// Vector normalization with SIMD
pub const NormalizationOps = struct {
    /// Normalize vector to unit length
    pub fn normalize(vector: []f32) void {
        const magnitude = calculateMagnitude(vector);
        if (magnitude > 0.0) {
            scaleVector(vector, 1.0 / magnitude);
        }
    }
    
    /// Calculate vector magnitude using SIMD
    pub fn calculateMagnitude(vector: []const f32) f32 {
        const features = CpuFeatures.detect();
        const optimal_size = features.getOptimalVectorSize();
        
        return switch (optimal_size) {
            16 => calculateMagnitudeSimd16(vector),
            8 => calculateMagnitudeSimd8(vector),
            4 => calculateMagnitudeSimd4(vector),
            else => calculateMagnitudeScalar(vector),
        };
    }
    
    /// Scale vector by scalar using SIMD
    pub fn scaleVector(vector: []f32, scalar: f32) void {
        const features = CpuFeatures.detect();
        const optimal_size = features.getOptimalVectorSize();
        
        switch (optimal_size) {
            16 => scaleVectorSimd16(vector, scalar),
            8 => scaleVectorSimd8(vector, scalar),
            4 => scaleVectorSimd4(vector, scalar),
            else => scaleVectorScalar(vector, scalar),
        }
    }
    
    fn calculateMagnitudeSimd16(vector: []const f32) f32 {
        var sum: @Vector(16, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 16 <= vector.len) : (i += 16) {
            const v: @Vector(16, f32) = vector[i..i + 16][0..16].*;
            sum += v * v;
        }
        
        var scalar_sum: f32 = @reduce(.Add, sum);
        while (i < vector.len) : (i += 1) {
            scalar_sum += vector[i] * vector[i];
        }
        
        return std.math.sqrt(scalar_sum);
    }
    
    fn calculateMagnitudeSimd8(vector: []const f32) f32 {
        var sum: @Vector(8, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 8 <= vector.len) : (i += 8) {
            const v: @Vector(8, f32) = vector[i..i + 8][0..8].*;
            sum += v * v;
        }
        
        var scalar_sum: f32 = @reduce(.Add, sum);
        while (i < vector.len) : (i += 1) {
            scalar_sum += vector[i] * vector[i];
        }
        
        return std.math.sqrt(scalar_sum);
    }
    
    fn calculateMagnitudeSimd4(vector: []const f32) f32 {
        var sum: @Vector(4, f32) = @splat(0.0);
        var i: usize = 0;
        
        while (i + 4 <= vector.len) : (i += 4) {
            const v: @Vector(4, f32) = vector[i..i + 4][0..4].*;
            sum += v * v;
        }
        
        var scalar_sum: f32 = @reduce(.Add, sum);
        while (i < vector.len) : (i += 1) {
            scalar_sum += vector[i] * vector[i];
        }
        
        return std.math.sqrt(scalar_sum);
    }
    
    fn calculateMagnitudeScalar(vector: []const f32) f32 {
        var sum: f32 = 0.0;
        for (vector) |v| {
            sum += v * v;
        }
        return std.math.sqrt(sum);
    }
    
    fn scaleVectorSimd16(vector: []f32, scalar: f32) void {
        const scale_vec: @Vector(16, f32) = @splat(scalar);
        var i: usize = 0;
        
        while (i + 16 <= vector.len) : (i += 16) {
            var v: @Vector(16, f32) = vector[i..i + 16][0..16].*;
            v *= scale_vec;
            vector[i..i + 16][0..16].* = v;
        }
        
        while (i < vector.len) : (i += 1) {
            vector[i] *= scalar;
        }
    }
    
    fn scaleVectorSimd8(vector: []f32, scalar: f32) void {
        const scale_vec: @Vector(8, f32) = @splat(scalar);
        var i: usize = 0;
        
        while (i + 8 <= vector.len) : (i += 8) {
            var v: @Vector(8, f32) = vector[i..i + 8][0..8].*;
            v *= scale_vec;
            vector[i..i + 8][0..8].* = v;
        }
        
        while (i < vector.len) : (i += 1) {
            vector[i] *= scalar;
        }
    }
    
    fn scaleVectorSimd4(vector: []f32, scalar: f32) void {
        const scale_vec: @Vector(4, f32) = @splat(scalar);
        var i: usize = 0;
        
        while (i + 4 <= vector.len) : (i += 4) {
            var v: @Vector(4, f32) = vector[i..i + 4][0..4].*;
            v *= scale_vec;
            vector[i..i + 4][0..4].* = v;
        }
        
        while (i < vector.len) : (i += 1) {
            vector[i] *= scalar;
        }
    }
    
    fn scaleVectorScalar(vector: []f32, scalar: f32) void {
        for (vector) |*v| {
            v.* *= scalar;
        }
    }
};

/// Batch operations for processing multiple vectors efficiently
pub const BatchOps = struct {
    /// Calculate distances between query vector and multiple vectors
    pub fn batchDistance(
        query: []const f32,
        vectors: []const []const f32,
        distances: []f32,
        distance_type: enum { euclidean, cosine, manhattan },
    ) void {
        if (vectors.len != distances.len) return;
        
        for (vectors, 0..) |vector, i| {
            distances[i] = switch (distance_type) {
                .euclidean => DistanceOps.euclideanDistance(query, vector),
                .cosine => 1.0 - DistanceOps.cosineSimilarity(query, vector),
                .manhattan => DistanceOps.manhattanDistance(query, vector),
            };
        }
    }
    
    /// Normalize multiple vectors in batch
    pub fn batchNormalize(vectors: [][]f32) void {
        for (vectors) |vector| {
            NormalizationOps.normalize(vector);
        }
    }
    
    /// Parallel batch operations
    pub fn parallelBatchDistance(
        allocator: std.mem.Allocator,
        query: []const f32,
        vectors: []const []const f32,
        distances: []f32,
        distance_type: enum { euclidean, cosine, manhattan },
        thread_count: usize,
    ) !void {
        if (vectors.len != distances.len) return;
        if (vectors.len == 0) return;
        
        const actual_thread_count = @min(thread_count, vectors.len);
        const chunk_size = (vectors.len + actual_thread_count - 1) / actual_thread_count;
        
        var threads = try allocator.alloc(std.Thread, actual_thread_count);
        defer allocator.free(threads);
        
        const WorkerData = struct {
            query: []const f32,
            vectors: []const []const f32,
            distances: []f32,
            distance_type: @TypeOf(distance_type),
            start: usize,
            end: usize,
        };
        
        var worker_data = try allocator.alloc(WorkerData, actual_thread_count);
        defer allocator.free(worker_data);
        
        // Start worker threads
        for (threads, 0..) |*thread, i| {
            const start = i * chunk_size;
            const end = @min(start + chunk_size, vectors.len);
            
            worker_data[i] = WorkerData{
                .query = query,
                .vectors = vectors,
                .distances = distances,
                .distance_type = distance_type,
                .start = start,
                .end = end,
            };
            
            thread.* = try std.Thread.spawn(.{}, batchWorker, .{&worker_data[i]});
        }
        
        // Wait for completion
        for (threads) |thread| {
            thread.join();
        }
    }
    
    fn batchWorker(data: *const anytype) void {
        for (data.start..data.end) |i| {
            data.distances[i] = switch (data.distance_type) {
                .euclidean => DistanceOps.euclideanDistance(data.query, data.vectors[i]),
                .cosine => 1.0 - DistanceOps.cosineSimilarity(data.query, data.vectors[i]),
                .manhattan => DistanceOps.manhattanDistance(data.query, data.vectors[i]),
            };
        }
    }
};

/// Performance benchmarking utilities
pub const BenchmarkOps = struct {
    /// Benchmark distance calculation performance
    pub fn benchmarkDistance(
        allocator: std.mem.Allocator,
        dimension: usize,
        vector_count: usize,
        iterations: usize,
    ) !struct {
        scalar_time_ms: f64,
        simd_time_ms: f64,
        speedup: f64,
    } {
        // Generate test data
        var vectors = try allocator.alloc([]f32, vector_count);
        defer {
            for (vectors) |vector| {
                allocator.free(vector);
            }
            allocator.free(vectors);
        }
        
        for (vectors) |*vector| {
            vector.* = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
        }
        
        const query = try core.random.vector(f32, allocator, dimension, -1.0, 1.0);
        defer allocator.free(query);
        
        // Benchmark scalar implementation
        const scalar_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            for (vectors) |vector| {
                _ = DistanceOps.euclideanDistanceScalar(query, vector);
            }
        }
        const scalar_end = std.time.nanoTimestamp();
        const scalar_time_ms = @as(f64, @floatFromInt(scalar_end - scalar_start)) / 1_000_000.0;
        
        // Benchmark SIMD implementation
        const simd_start = std.time.nanoTimestamp();
        for (0..iterations) |_| {
            for (vectors) |vector| {
                _ = DistanceOps.euclideanDistance(query, vector);
            }
        }
        const simd_end = std.time.nanoTimestamp();
        const simd_time_ms = @as(f64, @floatFromInt(simd_end - simd_start)) / 1_000_000.0;
        
        const speedup = if (simd_time_ms > 0) scalar_time_ms / simd_time_ms else 1.0;
        
        return .{
            .scalar_time_ms = scalar_time_ms,
            .simd_time_ms = simd_time_ms,
            .speedup = speedup,
        };
    }
};

test "CPU feature detection" {
    const features = CpuFeatures.detect();
    const optimal_size = features.getOptimalVectorSize();
    
    // Should detect at least some features on modern systems
    try testing.expect(optimal_size >= 1);
    try testing.expect(optimal_size <= 16);
}

test "Optimized distance calculations" {
    const vector_a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vector_b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    
    const euclidean = DistanceOps.euclideanDistance(&vector_a, &vector_b);
    const cosine = DistanceOps.cosineSimilarity(&vector_a, &vector_b);
    const manhattan = DistanceOps.manhattanDistance(&vector_a, &vector_b);
    
    try testing.expect(euclidean > 0.0);
    try testing.expect(cosine >= 0.0 and cosine <= 1.0);
    try testing.expect(manhattan > 0.0);
    
    // Test consistency with scalar implementations
    const euclidean_scalar = DistanceOps.euclideanDistanceScalar(&vector_a, &vector_b);
    try testing.expect(std.math.approxEqAbs(f32, euclidean, euclidean_scalar, 0.001));
}

test "Matrix operations" {
    const testing_alloc = std.testing.allocator;
    
    // Test 2x3 * 3x2 = 2x2 matrix multiplication
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 }; // 2x3
    const b = [_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 }; // 3x2
    var result = try testing_alloc.alloc(f32, 4); // 2x2
    defer testing_alloc.free(result);
    
    MatrixOps.multiply(&a, &b, result, 2, 3, 2);
    
    // Expected result: [[58, 64], [139, 154]]
    try testing.expect(std.math.approxEqAbs(f32, result[0], 58.0, 0.001));
    try testing.expect(std.math.approxEqAbs(f32, result[1], 64.0, 0.001));
    try testing.expect(std.math.approxEqAbs(f32, result[2], 139.0, 0.001));
    try testing.expect(std.math.approxEqAbs(f32, result[3], 154.0, 0.001));
}

test "Vector normalization" {
    var vector = [_]f32{ 3.0, 4.0, 0.0, 0.0 };
    
    NormalizationOps.normalize(&vector);
    
    const magnitude = NormalizationOps.calculateMagnitude(&vector);
    try testing.expect(std.math.approxEqAbs(f32, magnitude, 1.0, 0.001));
}

test "Batch operations" {
    const testing_alloc = std.testing.allocator;
    
    const query = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const vectors = [_][]const f32{
        &[_]f32{ 1.0, 2.0, 3.0, 4.0 },
        &[_]f32{ 2.0, 3.0, 4.0, 5.0 },
        &[_]f32{ 0.0, 1.0, 2.0, 3.0 },
    };
    
    var distances = try testing_alloc.alloc(f32, vectors.len);
    defer testing_alloc.free(distances);
    
    BatchOps.batchDistance(&query, &vectors, distances, .euclidean);
    
    try testing.expect(distances[0] == 0.0); // Same vector
    try testing.expect(distances[1] > 0.0);   // Different vector
    try testing.expect(distances[2] > 0.0);   // Different vector
}
