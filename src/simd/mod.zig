//! SIMD Module - High-performance vectorized operations
//!
//! This module provides SIMD-optimized implementations for:
//! - Vector arithmetic and distance calculations
//! - Text processing and pattern matching
//! - Matrix operations
//! - Data transformation pipelines
//! - Custom compute kernels

const std = @import("std");
const builtin = @import("builtin");
const core = @import("../core/mod.zig");

/// SIMD configuration based on target architecture
pub const config = struct {
    /// Vector width in bytes
    pub const vector_width: comptime_int = switch (builtin.cpu.arch) {
        .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) 32 else 16,
        .aarch64 => if (std.Target.aarch64.featureSetHas(builtin.cpu.features, .sve)) 64 else 16,
        .wasm32, .wasm64 => 16,
        else => 1,
    };

    /// Whether SIMD is available
    pub const has_simd = vector_width > 1;

    /// Preferred alignment for SIMD operations
    pub const alignment = if (has_simd) vector_width else @alignOf(f32);

    /// SIMD level string
    pub const level = "auto";
};

/// Vector type for a given element type and count
pub fn Vector(comptime T: type, comptime len: comptime_int) type {
    if (config.has_simd and len > 1) {
        return @Vector(len, T);
    } else {
        return [len]T;
    }
}

/// SIMD operations for vectors
pub const ops = struct {
    /// Add two vectors
    pub inline fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        const T = @TypeOf(a);
        if (comptime isSimdVector(T)) {
            return a + b;
        } else {
            var result = a;
            inline for (0..a.len) |i| {
                result[i] = a[i] + b[i];
            }
            return result;
        }
    }

    /// Subtract two vectors
    pub inline fn sub(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        const T = @TypeOf(a);
        if (comptime isSimdVector(T)) {
            return a - b;
        } else {
            var result = a;
            inline for (0..a.len) |i| {
                result[i] = a[i] - b[i];
            }
            return result;
        }
    }

    /// Multiply two vectors element-wise
    pub inline fn mul(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        const T = @TypeOf(a);
        if (comptime isSimdVector(T)) {
            return a * b;
        } else {
            var result = a;
            inline for (0..a.len) |i| {
                result[i] = a[i] * b[i];
            }
            return result;
        }
    }

    /// Divide two vectors element-wise
    pub inline fn div(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
        const T = @TypeOf(a);
        if (comptime isSimdVector(T)) {
            return a / b;
        } else {
            var result = a;
            inline for (0..a.len) |i| {
                result[i] = a[i] / b[i];
            }
            return result;
        }
    }

    /// Fused multiply-add: a * b + c
    pub inline fn fma(a: anytype, b: @TypeOf(a), c: @TypeOf(a)) @TypeOf(a) {
        const T = @TypeOf(a);
        const info = @typeInfo(T);

        if (comptime isSimdVector(T)) {
            const element_type = info.vector.child;
            if (element_type == f32 or element_type == f64) {
                return @mulAdd(T, a, b, c);
            } else {
                return add(mul(a, b), c);
            }
        } else {
            var result = a;
            inline for (0..a.len) |i| {
                result[i] = @mulAdd(@TypeOf(a[i]), a[i], b[i], c[i]);
            }
            return result;
        }
    }

    /// Horizontal sum of vector elements
    pub inline fn hsum(v: anytype) ElementType(@TypeOf(v)) {
        const T = @TypeOf(v);
        if (comptime isSimdVector(T)) {
            return @reduce(.Add, v);
        } else {
            var sum = v[0];
            inline for (1..v.len) |i| {
                sum += v[i];
            }
            return sum;
        }
    }

    /// Horizontal minimum
    pub inline fn hmin(v: anytype) ElementType(@TypeOf(v)) {
        const T = @TypeOf(v);
        if (comptime isSimdVector(T)) {
            return @reduce(.Min, v);
        } else {
            var min = v[0];
            inline for (1..v.len) |i| {
                min = @min(min, v[i]);
            }
            return min;
        }
    }

    /// Horizontal maximum
    pub inline fn hmax(v: anytype) ElementType(@TypeOf(v)) {
        const T = @TypeOf(v);
        if (comptime isSimdVector(T)) {
            return @reduce(.Max, v);
        } else {
            var max = v[0];
            inline for (1..v.len) |i| {
                max = @max(max, v[i]);
            }
            return max;
        }
    }

    /// Dot product of two vectors
    pub inline fn dot(a: anytype, b: @TypeOf(a)) ElementType(@TypeOf(a)) {
        return hsum(mul(a, b));
    }

    /// Square root of vector elements
    pub inline fn sqrt(v: anytype) @TypeOf(v) {
        const T = @TypeOf(v);
        if (comptime isSimdVector(T)) {
            return @sqrt(v);
        } else {
            var result = v;
            inline for (0..v.len) |i| {
                result[i] = @sqrt(v[i]);
            }
            return result;
        }
    }

    /// Absolute value of vector elements
    pub inline fn abs(v: anytype) @TypeOf(v) {
        const T = @TypeOf(v);
        if (comptime isSimdVector(T)) {
            return @abs(v);
        } else {
            var result = v;
            inline for (0..v.len) |i| {
                result[i] = @abs(v[i]);
            }
            return result;
        }
    }

    /// Clamp vector elements between min and max
    pub inline fn clamp(v: anytype, min_val: ElementType(@TypeOf(v)), max_val: ElementType(@TypeOf(v))) @TypeOf(v) {
        const T = @TypeOf(v);
        const min_vec = @as(T, @splat(min_val));
        const max_vec = @as(T, @splat(max_val));

        if (comptime isSimdVector(T)) {
            return @max(min_vec, @min(max_vec, v));
        } else {
            var result = v;
            inline for (0..v.len) |i| {
                result[i] = @max(min_val, @min(max_val, v[i]));
            }
            return result;
        }
    }
};

/// Distance calculations
pub const distance = struct {
    /// Euclidean distance squared
    pub fn euclideanSquared(comptime T: type, a: []const T, b: []const T) T {
        std.debug.assert(a.len == b.len);

        const vec_size = comptime getVectorSize(T);
        var sum: T = 0;
        var i: usize = 0;

        // Process vectorized chunks
        while (i + vec_size <= a.len) : (i += vec_size) {
            const va = loadVector(T, vec_size, a[i..]);
            const vb = loadVector(T, vec_size, b[i..]);
            const diff = ops.sub(va, vb);
            sum += ops.dot(diff, diff);
        }

        // Process remaining elements
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }

        return sum;
    }

    /// Euclidean distance
    pub fn euclidean(comptime T: type, a: []const T, b: []const T) T {
        return @sqrt(euclideanSquared(T, a, b));
    }

    /// Manhattan distance
    pub fn manhattan(comptime T: type, a: []const T, b: []const T) T {
        std.debug.assert(a.len == b.len);

        const vec_size = comptime getVectorSize(T);
        var sum: T = 0;
        var i: usize = 0;

        // Process vectorized chunks
        while (i + vec_size <= a.len) : (i += vec_size) {
            const va = loadVector(T, vec_size, a[i..]);
            const vb = loadVector(T, vec_size, b[i..]);
            const diff = ops.abs(ops.sub(va, vb));
            sum += ops.hsum(diff);
        }

        // Process remaining elements
        while (i < a.len) : (i += 1) {
            sum += @abs(a[i] - b[i]);
        }

        return sum;
    }

    /// Cosine similarity
    pub fn cosineSimilarity(comptime T: type, a: []const T, b: []const T) T {
        std.debug.assert(a.len == b.len);

        const vec_size = comptime getVectorSize(T);
        var dot_product: T = 0;
        var norm_a: T = 0;
        var norm_b: T = 0;
        var i: usize = 0;

        // Process vectorized chunks
        while (i + vec_size <= a.len) : (i += vec_size) {
            const va = loadVector(T, vec_size, a[i..]);
            const vb = loadVector(T, vec_size, b[i..]);

            dot_product += ops.dot(va, vb);
            norm_a += ops.dot(va, va);
            norm_b += ops.dot(vb, vb);
        }

        // Process remaining elements
        while (i < a.len) : (i += 1) {
            dot_product += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        const magnitude = @sqrt(norm_a) * @sqrt(norm_b);
        return if (magnitude > 0) dot_product / magnitude else 0;
    }
};

/// Text processing operations
pub const text = struct {
    /// Count occurrences of a byte in a buffer
    pub fn countByte(haystack: []const u8, needle: u8) usize {
        // Use conservative SIMD approach - fall back to scalar if alignment issues
        if (!config.has_simd or haystack.len < 32) {
            return countByteScalar(haystack, needle);
        }

        const vec_size = @min(32, config.vector_width);
        const alignment = @alignOf(@Vector(vec_size, u8));

        // Check if we can safely use SIMD with current alignment
        if (@intFromPtr(haystack.ptr) % alignment != 0) {
            // Data is not aligned, use scalar approach
            return countByteScalar(haystack, needle);
        }

        var count: usize = 0;
        var i: usize = 0;
        const needle_vec = @as(@Vector(vec_size, u8), @splat(needle));

        // Process SIMD chunks (data is aligned)
        while (i + vec_size <= haystack.len) : (i += vec_size) {
            const chunk = @as(*const @Vector(vec_size, u8), @ptrCast(@alignCast(haystack.ptr + i))).*;
            const matches = chunk == needle_vec;
            count += @popCount(@as(std.meta.Int(.unsigned, vec_size), @bitCast(matches)));
        }

        // Process remaining bytes
        while (i < haystack.len) : (i += 1) {
            if (haystack[i] == needle) count += 1;
        }

        return count;
    }

    fn countByteScalar(haystack: []const u8, needle: u8) usize {
        var count: usize = 0;
        for (haystack) |byte| {
            if (byte == needle) count += 1;
        }
        return count;
    }

    /// Find first occurrence of byte
    pub fn findByte(haystack: []const u8, needle: u8) ?usize {
        // Use conservative SIMD approach - fall back to scalar if alignment issues
        if (!config.has_simd or haystack.len < 32) {
            return std.mem.indexOfScalar(u8, haystack, needle);
        }

        const vec_size = @min(32, config.vector_width);
        const alignment = @alignOf(@Vector(vec_size, u8));

        // Check if we can safely use SIMD with current alignment
        if (@intFromPtr(haystack.ptr) % alignment != 0) {
            // Data is not aligned, use scalar approach
            return std.mem.indexOfScalar(u8, haystack, needle);
        }

        var i: usize = 0;
        const needle_vec = @as(@Vector(vec_size, u8), @splat(needle));

        // Process SIMD chunks (data is aligned)
        while (i + vec_size <= haystack.len) : (i += vec_size) {
            const chunk = @as(*const @Vector(vec_size, u8), @ptrCast(@alignCast(haystack.ptr + i))).*;
            const matches = chunk == needle_vec;
            const match_bits = @as(std.meta.Int(.unsigned, vec_size), @bitCast(matches));

            if (match_bits != 0) {
                const offset = @ctz(match_bits);
                return i + offset;
            }
        }

        // Check remaining bytes
        return std.mem.indexOfScalar(u8, haystack[i..], needle);
    }

    /// Convert ASCII to lowercase
    pub fn toLowerAscii(dst: []u8, src: []const u8) void {
        std.debug.assert(dst.len >= src.len);

        // Use conservative SIMD approach - fall back to scalar if alignment issues
        if (!config.has_simd or src.len < 32) {
            for (src, 0..) |c, i| {
                dst[i] = std.ascii.toLower(c);
            }
            return;
        }

        const vec_size = @min(32, config.vector_width);
        const alignment = @alignOf(@Vector(vec_size, u8));

        // Check if both src and dst are aligned for SIMD operations
        if (@intFromPtr(src.ptr) % alignment != 0 or @intFromPtr(dst.ptr) % alignment != 0) {
            // Data is not aligned, use scalar approach
            for (src, 0..) |c, i| {
                dst[i] = std.ascii.toLower(c);
            }
            return;
        }

        var i: usize = 0;
        const A_vec = @as(@Vector(vec_size, u8), @splat('A'));
        const Z_vec = @as(@Vector(vec_size, u8), @splat('Z'));
        const diff_vec = @as(@Vector(vec_size, u8), @splat('a' - 'A'));

        // Process SIMD chunks (both src and dst are aligned)
        while (i + vec_size <= src.len) : (i += vec_size) {
            const chunk = @as(*const @Vector(vec_size, u8), @ptrCast(@alignCast(src.ptr + i))).*;
            const is_upper = (chunk >= A_vec) & (chunk <= Z_vec);
            const mask = @select(u8, is_upper, diff_vec, @as(@Vector(vec_size, u8), @splat(0)));
            const result = chunk + mask;
            @as(*@Vector(vec_size, u8), @ptrCast(@alignCast(dst.ptr + i))).* = result;
        }

        // Process remaining bytes
        while (i < src.len) : (i += 1) {
            dst[i] = std.ascii.toLower(src[i]);
        }
    }
};

/// Matrix operations
pub const matrix = struct {
    /// Matrix multiplication using SIMD
    pub fn multiply(comptime T: type, c: []T, a: []const T, b: []const T, m: usize, n: usize, k: usize) void {
        std.debug.assert(c.len >= m * n);
        std.debug.assert(a.len >= m * k);
        std.debug.assert(b.len >= k * n);

        // Use tiled matrix multiplication for cache efficiency
        const tile_size = 64;
        var i: usize = 0;

        while (i < m) : (i += tile_size) {
            const i_end = @min(i + tile_size, m);
            var j: usize = 0;

            while (j < n) : (j += tile_size) {
                const j_end = @min(j + tile_size, n);
                var k_idx: usize = 0;

                while (k_idx < k) : (k_idx += tile_size) {
                    const k_end = @min(k_idx + tile_size, k);

                    // Process tile
                    multiplyTile(T, c, a, b, i, i_end, j, j_end, k_idx, k_end, m, n, k);
                }
            }
        }
    }

    fn multiplyTile(comptime T: type, c: []T, a: []const T, b: []const T, i_start: usize, i_end: usize, j_start: usize, j_end: usize, k_start: usize, k_end: usize, m: usize, n: usize, k: usize) void {
        _ = m;
        const vec_size = comptime getVectorSize(T);

        var i = i_start;
        while (i < i_end) : (i += 1) {
            var j = j_start;

            while (j + vec_size <= j_end) : (j += vec_size) {
                var sum = @as(Vector(T, vec_size), @splat(0));

                var k_idx = k_start;
                while (k_idx < k_end) : (k_idx += 1) {
                    const a_scalar = @as(Vector(T, vec_size), @splat(a[i * k + k_idx]));
                    const b_vec = loadVector(T, vec_size, b[k_idx * n + j ..]);
                    sum = ops.fma(a_scalar, b_vec, sum);
                }

                if (k_start == 0) {
                    storeVector(T, vec_size, c[i * n + j ..], sum);
                } else {
                    const c_vec = loadVector(T, vec_size, c[i * n + j ..]);
                    storeVector(T, vec_size, c[i * n + j ..], ops.add(c_vec, sum));
                }
            }

            // Handle remaining columns
            while (j < j_end) : (j += 1) {
                var sum: T = if (k_start == 0) 0 else c[i * n + j];
                var k_idx = k_start;
                while (k_idx < k_end) : (k_idx += 1) {
                    sum += a[i * k + k_idx] * b[k_idx * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }

    /// Transpose matrix
    pub fn transpose(comptime T: type, dst: []T, src: []const T, rows: usize, cols: usize) void {
        std.debug.assert(dst.len >= rows * cols);
        std.debug.assert(src.len >= rows * cols);

        // Use tiled transpose for cache efficiency
        const tile_size = 32;
        var i: usize = 0;

        while (i < rows) : (i += tile_size) {
            const i_end = @min(i + tile_size, rows);
            var j: usize = 0;

            while (j < cols) : (j += tile_size) {
                const j_end = @min(j + tile_size, cols);

                // Transpose tile
                var ii = i;
                while (ii < i_end) : (ii += 1) {
                    var jj = j;
                    while (jj < j_end) : (jj += 1) {
                        dst[jj * rows + ii] = src[ii * cols + jj];
                    }
                }
            }
        }
    }
};

/// Helper functions
fn isSimdVector(comptime T: type) bool {
    return @typeInfo(T) == .vector;
}

fn ElementType(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .vector => |v| v.child,
        .array => |a| a.child,
        else => @compileError("Expected vector or array type"),
    };
}

fn getVectorSize(comptime T: type) comptime_int {
    return switch (T) {
        f32 => @divExact(config.vector_width, 4),
        f64 => @divExact(config.vector_width, 8),
        u8, i8 => config.vector_width,
        u16, i16 => @divExact(config.vector_width, 2),
        u32, i32 => @divExact(config.vector_width, 4),
        u64, i64 => @divExact(config.vector_width, 8),
        else => 1,
    };
}

fn loadVector(comptime T: type, comptime len: comptime_int, data: []const T) Vector(T, len) {
    if (comptime isSimdVector(Vector(T, len))) {
        return @as(*const Vector(T, len), @ptrCast(@alignCast(data.ptr))).*;
    } else {
        var result: [len]T = undefined;
        inline for (0..len) |i| {
            result[i] = data[i];
        }
        return result;
    }
}

fn storeVector(comptime T: type, comptime len: comptime_int, data: []T, vec: Vector(T, len)) void {
    if (comptime isSimdVector(Vector(T, len))) {
        @as(*Vector(T, len), @ptrCast(@alignCast(data.ptr))).* = vec;
    } else {
        inline for (0..len) |i| {
            data[i] = vec[i];
        }
    }
}

/// Custom compute kernel builder
pub const Kernel = struct {
    /// Define a SIMD kernel that operates on slices
    pub fn create(
        comptime name: []const u8,
        comptime T: type,
        comptime operation: fn (Vector(T, getVectorSize(T))) Vector(T, getVectorSize(T)),
    ) type {
        return struct {
            pub const kernel_name = name;

            pub fn execute(dst: []T, src: []const T) void {
                std.debug.assert(dst.len >= src.len);

                const vec_size = comptime getVectorSize(T);
                var i: usize = 0;

                // Process vectorized chunks
                while (i + vec_size <= src.len) : (i += vec_size) {
                    const input = loadVector(T, vec_size, src[i..]);
                    const output = operation(input);
                    storeVector(T, vec_size, dst[i..], output);
                }

                // Process remaining elements
                while (i < src.len) : (i += 1) {
                    var single_input: [1]T = .{src[i]};
                    var single_output: [1]T = undefined;
                    const input = loadVector(T, 1, &single_input);
                    const output = operation(input);
                    storeVector(T, 1, &single_output, output);
                    dst[i] = single_output[0];
                }
            }

            pub fn executeBatch(dsts: [][]T, srcs: []const []const T) void {
                std.debug.assert(dsts.len == srcs.len);
                for (dsts, srcs) |dst, src| {
                    execute(dst, src);
                }
            }
        };
    }
};

test "SIMD vector operations" {
    const a = Vector(f32, 4){ 1.0, 2.0, 3.0, 4.0 };
    const b = Vector(f32, 4){ 5.0, 6.0, 7.0, 8.0 };

    const sum = ops.add(a, b);
    const expected_sum = Vector(f32, 4){ 6.0, 8.0, 10.0, 12.0 };

    if (comptime isSimdVector(@TypeOf(sum))) {
        try std.testing.expectEqual(expected_sum, sum);
    } else {
        for (sum, expected_sum) |actual, expected| {
            try std.testing.expectApproxEqAbs(expected, actual, 0.001);
        }
    }

    const dot_product = ops.dot(a, b);
    try std.testing.expectApproxEqAbs(@as(f32, 70.0), dot_product, 0.001);
}

test "SIMD distance calculations" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    const dist_sq = distance.euclideanSquared(f32, &a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), dist_sq, 0.001);

    const manhattan_dist = distance.manhattan(f32, &a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), manhattan_dist, 0.001);

    const cosine_sim = distance.cosineSimilarity(f32, &a, &b);
    try std.testing.expect(cosine_sim > 0.9);
}

test "SIMD text operations" {
    const haystack = "Hello, World! This is a test string with some repeated letters.";
    const count = text.countByte(haystack, 'l');
    try std.testing.expectEqual(@as(usize, 4), count);

    const pos = text.findByte(haystack, 'W');
    try std.testing.expectEqual(@as(?usize, 7), pos);

    var lower_buf: [64]u8 = undefined;
    text.toLowerAscii(&lower_buf, haystack);
    try std.testing.expect(std.mem.startsWith(u8, &lower_buf, "hello, world!"));
}
