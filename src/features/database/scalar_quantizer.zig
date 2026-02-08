//! Scalar Quantization Module
//!
//! Provides scalar quantization for compressing high-dimensional vectors using
//! min-max scaling to fixed-bit codes (4, 8, or 16 bits per dimension).
//!
//! Memory savings: 4-bit = 8x, 8-bit = 4x, 16-bit = 2x compression.
//! Typical recall loss: <1% for 8-bit quantization.
//!
//! For fixed bit widths, use ScalarQuantizerTyped(BITS) to get compile-time
//! validated quantizer with pre-computed scale factors.
//!
//! References:
//! - Zilliz Quantization: https://zilliz.com/learn/scalar-quantization-and-product-quantization

const std = @import("std");
const simd = @import("../../services/shared/simd.zig");

// ============================================================================
// Comptime Quantization Types
// ============================================================================

/// Comptime-validated scalar quantizer for a fixed bit width.
/// Provides compile-time validation and pre-computed scale factors.
///
/// Example usage:
/// ```zig
/// const Q8 = ScalarQuantizerTyped(8);
/// var quantizer = try Q8.init(allocator, 256);
/// defer quantizer.deinit();
/// ```
pub fn ScalarQuantizerTyped(comptime BITS: u8) type {
    // Compile-time validation of bit width
    comptime {
        if (BITS != 4 and BITS != 8 and BITS != 16) {
            @compileError("BITS must be 4, 8, or 16 for scalar quantization");
        }
    }

    return struct {
        dim: usize,
        min_values: []f32,
        max_values: []f32,
        trained: bool,
        allocator: std.mem.Allocator,

        const Self = @This();

        /// Compile-time known bit width
        pub const BITS_COUNT: u8 = BITS;

        /// Number of quantization levels (comptime)
        pub const LEVELS: u16 = (1 << BITS) - 1;

        /// Scale factor for conversion (comptime)
        pub const SCALE_FACTOR: f32 = @as(f32, LEVELS);

        /// Inverse scale factor (comptime)
        pub const INV_SCALE: f32 = 1.0 / @as(f32, LEVELS);

        /// Bytes per vector for this bit width (comptime calculation)
        pub fn bytesForDimension(dim: usize) usize {
            return switch (BITS) {
                4 => (dim + 1) / 2,
                8 => dim,
                16 => dim * 2,
                else => unreachable,
            };
        }

        /// Initialize a typed scalar quantizer for the given dimension.
        /// No runtime validation of bit width needed - done at comptime.
        pub fn init(allocator: std.mem.Allocator, dim: usize) QuantizationError!Self {
            if (dim == 0) return QuantizationError.InvalidDimension;

            const min_values = allocator.alloc(f32, dim) catch return QuantizationError.OutOfMemory;
            errdefer allocator.free(min_values);
            const max_values = allocator.alloc(f32, dim) catch return QuantizationError.OutOfMemory;

            @memset(min_values, std.math.inf(f32));
            @memset(max_values, -std.math.inf(f32));

            return .{
                .dim = dim,
                .min_values = min_values,
                .max_values = max_values,
                .trained = false,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.min_values);
            self.allocator.free(self.max_values);
            self.* = undefined;
        }

        /// Train the quantizer on a set of vectors.
        pub fn train(self: *Self, vectors: []const []const f32) QuantizationError!void {
            if (vectors.len == 0) return QuantizationError.EmptyData;

            for (vectors) |vector| {
                if (vector.len != self.dim) return QuantizationError.DimensionMismatch;
            }

            @memset(self.min_values, std.math.inf(f32));
            @memset(self.max_values, -std.math.inf(f32));

            for (vectors) |vector| {
                for (vector, 0..) |value, i| {
                    if (value < self.min_values[i]) self.min_values[i] = value;
                    if (value > self.max_values[i]) self.max_values[i] = value;
                }
            }

            self.trained = true;
        }

        /// Encode a vector to quantized codes using compile-time optimized paths.
        pub fn encode(self: *const Self, vector: []const f32, output: []u8) QuantizationError!usize {
            if (!self.trained) return QuantizationError.NotTrained;
            if (vector.len != self.dim) return QuantizationError.DimensionMismatch;

            const bytes_needed = bytesForDimension(self.dim);
            if (output.len < bytes_needed) return QuantizationError.OutOfMemory;

            if (BITS == 8) {
                for (vector, 0..) |value, i| {
                    output[i] = quantizeValueTyped(value, self.min_values[i], self.max_values[i]);
                }
            } else if (BITS == 4) {
                var byte_idx: usize = 0;
                var i: usize = 0;
                while (i < self.dim) {
                    const code1 = quantizeValueTyped(vector[i], self.min_values[i], self.max_values[i]);
                    const code2 = if (i + 1 < self.dim)
                        quantizeValueTyped(vector[i + 1], self.min_values[i + 1], self.max_values[i + 1])
                    else
                        0;
                    output[byte_idx] = (code1 << 4) | code2;
                    byte_idx += 1;
                    i += 2;
                }
            } else if (BITS == 16) {
                for (vector, 0..) |value, i| {
                    const code = quantizeValue16Typed(value, self.min_values[i], self.max_values[i]);
                    output[i * 2] = @truncate(code >> 8);
                    output[i * 2 + 1] = @truncate(code);
                }
            }

            return bytes_needed;
        }

        /// Decode quantized codes back to a vector.
        pub fn decode(self: *const Self, codes: []const u8, output: []f32) QuantizationError!void {
            if (!self.trained) return QuantizationError.NotTrained;
            if (output.len != self.dim) return QuantizationError.DimensionMismatch;

            if (BITS == 8) {
                for (output, 0..) |*value, i| {
                    value.* = dequantizeValueTyped(codes[i], self.min_values[i], self.max_values[i]);
                }
            } else if (BITS == 4) {
                var byte_idx: usize = 0;
                var i: usize = 0;
                while (i < self.dim) {
                    const byte = codes[byte_idx];
                    output[i] = dequantizeValueTyped(byte >> 4, self.min_values[i], self.max_values[i]);
                    if (i + 1 < self.dim) {
                        output[i + 1] = dequantizeValueTyped(byte & 0x0F, self.min_values[i + 1], self.max_values[i + 1]);
                    }
                    byte_idx += 1;
                    i += 2;
                }
            } else if (BITS == 16) {
                for (output, 0..) |*value, i| {
                    const code = (@as(u16, codes[i * 2]) << 8) | @as(u16, codes[i * 2 + 1]);
                    value.* = dequantizeValue16Typed(code, self.min_values[i], self.max_values[i]);
                }
            }
        }

        /// Bytes per vector for this quantizer.
        pub fn bytesPerVector(self: *const Self) usize {
            return bytesForDimension(self.dim);
        }

        // Helper functions using comptime constants
        fn quantizeValueTyped(value: f32, min_val: f32, max_val: f32) u8 {
            if (max_val <= min_val) return 0;
            const normalized = (value - min_val) / (max_val - min_val);
            const scaled = normalized * SCALE_FACTOR;
            const clamped = std.math.clamp(scaled, 0.0, SCALE_FACTOR);
            return @intCast(@min(@as(u16, @intFromFloat(clamped)), LEVELS));
        }

        fn quantizeValue16Typed(value: f32, min_val: f32, max_val: f32) u16 {
            if (max_val <= min_val) return 0;
            const normalized = (value - min_val) / (max_val - min_val);
            const scaled = normalized * 65535.0;
            const clamped = std.math.clamp(scaled, 0.0, 65535.0);
            return @intFromFloat(clamped);
        }

        fn dequantizeValueTyped(code: u8, min_val: f32, max_val: f32) f32 {
            if (max_val <= min_val or LEVELS == 0) return min_val;
            const fraction = @as(f32, @floatFromInt(code)) * INV_SCALE;
            return min_val + fraction * (max_val - min_val);
        }

        fn dequantizeValue16Typed(code: u16, min_val: f32, max_val: f32) f32 {
            if (max_val <= min_val) return min_val;
            const fraction = @as(f32, @floatFromInt(code)) / 65535.0;
            return min_val + fraction * (max_val - min_val);
        }
    };
}

// Pre-instantiated quantizer types for common bit widths
pub const ScalarQuantizer4 = ScalarQuantizerTyped(4);
pub const ScalarQuantizer8 = ScalarQuantizerTyped(8);
pub const ScalarQuantizer16 = ScalarQuantizerTyped(16);

/// Errors that can occur during quantization operations.
pub const QuantizationError = error{
    InvalidDimension,
    InvalidBits,
    InvalidSubvectors,
    EmptyData,
    DimensionMismatch,
    NotTrained,
    OutOfMemory,
};

/// Scalar Quantizer using min-max scaling.
///
/// Compresses f32 vectors to fixed-bit codes (4, 8, or 16 bits).
/// Memory savings: 4-bit = 8x, 8-bit = 4x, 16-bit = 2x compression.
/// Typical recall loss: <1% for 8-bit quantization.
pub const ScalarQuantizer = struct {
    dim: usize,
    bits: u8,
    min_values: []f32,
    max_values: []f32,
    trained: bool,
    allocator: std.mem.Allocator,

    /// Configuration for scalar quantization.
    pub const Config = struct {
        /// Number of bits per dimension (4, 8, or 16).
        bits: u8 = 8,
    };

    /// Statistics about the quantizer.
    pub const Stats = struct {
        dimension: usize,
        bits: u8,
        compression_ratio: f32,
        bytes_per_vector: usize,
    };

    /// Initialize a scalar quantizer for the given dimension.
    pub fn init(allocator: std.mem.Allocator, dim: usize, config: Config) QuantizationError!ScalarQuantizer {
        if (dim == 0) return QuantizationError.InvalidDimension;
        if (config.bits != 4 and config.bits != 8 and config.bits != 16) {
            return QuantizationError.InvalidBits;
        }

        const min_values = allocator.alloc(f32, dim) catch return QuantizationError.OutOfMemory;
        errdefer allocator.free(min_values);
        const max_values = allocator.alloc(f32, dim) catch return QuantizationError.OutOfMemory;

        // Initialize to extreme values
        @memset(min_values, std.math.inf(f32));
        @memset(max_values, -std.math.inf(f32));

        return .{
            .dim = dim,
            .bits = config.bits,
            .min_values = min_values,
            .max_values = max_values,
            .trained = false,
            .allocator = allocator,
        };
    }

    /// Release resources.
    pub fn deinit(self: *ScalarQuantizer) void {
        self.allocator.free(self.min_values);
        self.allocator.free(self.max_values);
        self.* = undefined;
    }

    /// Train the quantizer on a set of vectors to compute min/max statistics.
    /// Uses SIMD to process multiple dimensions at once for improved performance.
    pub fn train(self: *ScalarQuantizer, vectors: []const []const f32) QuantizationError!void {
        if (vectors.len == 0) return QuantizationError.EmptyData;

        // Validate dimensions first
        for (vectors) |vector| {
            if (vector.len != self.dim) return QuantizationError.DimensionMismatch;
        }

        // Reset statistics
        @memset(self.min_values, std.math.inf(f32));
        @memset(self.max_values, -std.math.inf(f32));

        // SIMD-accelerated min/max computation
        const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;

        if (VectorSize > 1 and self.dim >= VectorSize) {
            const Vec = @Vector(VectorSize, f32);
            var i: usize = 0;

            // Process VectorSize dimensions at a time
            while (i + VectorSize <= self.dim) : (i += VectorSize) {
                var min_vec: Vec = @splat(std.math.inf(f32));
                var max_vec: Vec = @splat(-std.math.inf(f32));

                // Find min/max across all vectors for these dimensions
                for (vectors) |vector| {
                    const v: Vec = vector[i..][0..VectorSize].*;
                    min_vec = @min(min_vec, v);
                    max_vec = @max(max_vec, v);
                }

                // Store results
                self.min_values[i..][0..VectorSize].* = min_vec;
                self.max_values[i..][0..VectorSize].* = max_vec;
            }

            // Scalar remainder for dimensions not aligned to VectorSize
            while (i < self.dim) : (i += 1) {
                for (vectors) |vector| {
                    if (vector[i] < self.min_values[i]) self.min_values[i] = vector[i];
                    if (vector[i] > self.max_values[i]) self.max_values[i] = vector[i];
                }
            }
        } else {
            // Fallback to scalar for small dimensions or no SIMD
            for (vectors) |vector| {
                for (vector, 0..) |value, i| {
                    if (value < self.min_values[i]) self.min_values[i] = value;
                    if (value > self.max_values[i]) self.max_values[i] = value;
                }
            }
        }

        self.trained = true;
    }

    /// Encode a vector to quantized codes.
    /// Returns the number of bytes written to output.
    /// Uses SIMD acceleration for 8-bit quantization.
    pub fn encode(self: *const ScalarQuantizer, vector: []const f32, output: []u8) QuantizationError!usize {
        if (!self.trained) return QuantizationError.NotTrained;
        if (vector.len != self.dim) return QuantizationError.DimensionMismatch;

        const bytes_needed = self.bytesPerVector();
        if (output.len < bytes_needed) return QuantizationError.OutOfMemory;

        const num_levels = self.levels();

        switch (self.bits) {
            8 => {
                // SIMD-accelerated 8-bit quantization
                const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;

                if (VectorSize > 1 and self.dim >= VectorSize) {
                    const Vec = @Vector(VectorSize, f32);
                    const levels_f: Vec = @splat(@as(f32, @floatFromInt(num_levels)));
                    const zero: Vec = @splat(0.0);
                    var i: usize = 0;

                    while (i + VectorSize <= self.dim) : (i += VectorSize) {
                        const vals: Vec = vector[i..][0..VectorSize].*;
                        const mins: Vec = self.min_values[i..][0..VectorSize].*;
                        const maxs: Vec = self.max_values[i..][0..VectorSize].*;

                        // Compute range and handle degenerate case (max <= min)
                        const range = maxs - mins;
                        const normalized = (vals - mins) / @max(range, @as(Vec, @splat(1e-10)));
                        const scaled = normalized * levels_f;
                        const clamped = @max(zero, @min(levels_f, scaled));

                        // Convert to u8 (manual per-element since SIMD u8 conversion is complex)
                        const arr: [VectorSize]f32 = clamped;
                        inline for (0..VectorSize) |j| {
                            output[i + j] = @intFromFloat(@min(@as(f32, @floatFromInt(num_levels)), arr[j]));
                        }
                    }

                    // Scalar remainder
                    while (i < self.dim) : (i += 1) {
                        output[i] = quantizeValue(vector[i], self.min_values[i], self.max_values[i], num_levels);
                    }
                } else {
                    // Scalar fallback
                    for (vector, 0..) |value, i| {
                        output[i] = quantizeValue(value, self.min_values[i], self.max_values[i], num_levels);
                    }
                }
            },
            4 => {
                // Pack two 4-bit values per byte
                var byte_idx: usize = 0;
                var i: usize = 0;
                while (i < self.dim) {
                    const code1 = quantizeValue(vector[i], self.min_values[i], self.max_values[i], num_levels);
                    const code2 = if (i + 1 < self.dim)
                        quantizeValue(vector[i + 1], self.min_values[i + 1], self.max_values[i + 1], num_levels)
                    else
                        0;
                    output[byte_idx] = (code1 << 4) | code2;
                    byte_idx += 1;
                    i += 2;
                }
            },
            16 => {
                // Two bytes per dimension
                for (vector, 0..) |value, i| {
                    const code = quantizeValue16(value, self.min_values[i], self.max_values[i]);
                    output[i * 2] = @truncate(code >> 8);
                    output[i * 2 + 1] = @truncate(code);
                }
            },
            else => unreachable,
        }

        return bytes_needed;
    }

    /// Decode quantized codes back to a vector.
    /// Uses SIMD acceleration for 8-bit dequantization.
    pub fn decode(self: *const ScalarQuantizer, codes: []const u8, output: []f32) QuantizationError!void {
        if (!self.trained) return QuantizationError.NotTrained;
        if (output.len != self.dim) return QuantizationError.DimensionMismatch;

        const num_levels = self.levels();

        switch (self.bits) {
            8 => {
                // SIMD-accelerated 8-bit dequantization
                const VectorSize = std.simd.suggestVectorLength(f32) orelse 8;

                if (VectorSize > 1 and self.dim >= VectorSize) {
                    const Vec = @Vector(VectorSize, f32);
                    const levels_f: Vec = @splat(@as(f32, @floatFromInt(num_levels)));
                    var i: usize = 0;

                    while (i + VectorSize <= self.dim) : (i += VectorSize) {
                        const mins: Vec = self.min_values[i..][0..VectorSize].*;
                        const maxs: Vec = self.max_values[i..][0..VectorSize].*;
                        const range = maxs - mins;

                        // Convert codes to float vector (manual per-element)
                        var code_floats: [VectorSize]f32 = undefined;
                        inline for (0..VectorSize) |j| {
                            code_floats[j] = @floatFromInt(codes[i + j]);
                        }
                        const code_vec: Vec = code_floats;

                        // Dequantize: min + (code / levels) * range
                        const fraction = code_vec / levels_f;
                        const result = mins + fraction * range;
                        output[i..][0..VectorSize].* = result;
                    }

                    // Scalar remainder
                    while (i < self.dim) : (i += 1) {
                        output[i] = dequantizeValue(codes[i], self.min_values[i], self.max_values[i], num_levels);
                    }
                } else {
                    // Scalar fallback
                    for (output, 0..) |*value, i| {
                        value.* = dequantizeValue(codes[i], self.min_values[i], self.max_values[i], num_levels);
                    }
                }
            },
            4 => {
                var byte_idx: usize = 0;
                var i: usize = 0;
                while (i < self.dim) {
                    const byte = codes[byte_idx];
                    output[i] = dequantizeValue(byte >> 4, self.min_values[i], self.max_values[i], num_levels);
                    if (i + 1 < self.dim) {
                        output[i + 1] = dequantizeValue(byte & 0x0F, self.min_values[i + 1], self.max_values[i + 1], num_levels);
                    }
                    byte_idx += 1;
                    i += 2;
                }
            },
            16 => {
                for (output, 0..) |*value, i| {
                    const code = (@as(u16, codes[i * 2]) << 8) | @as(u16, codes[i * 2 + 1]);
                    value.* = dequantizeValue16(code, self.min_values[i], self.max_values[i]);
                }
            },
            else => unreachable,
        }
    }

    /// Compute approximate distance between a query and encoded vector.
    /// Uses asymmetric distance computation (ADC) for better accuracy.
    pub fn asymmetricDistance(self: *const ScalarQuantizer, query: []const f32, codes: []const u8) QuantizationError!f32 {
        if (!self.trained) return QuantizationError.NotTrained;
        if (query.len != self.dim) return QuantizationError.DimensionMismatch;

        // Decode and compute distance
        const decoded = self.allocator.alloc(f32, self.dim) catch return QuantizationError.OutOfMemory;
        defer self.allocator.free(decoded);

        try self.decode(codes, decoded);

        // Return L2 distance squared
        var dist: f32 = 0.0;
        for (query, decoded) |q, d| {
            const diff = q - d;
            dist += diff * diff;
        }
        return dist;
    }

    /// Compute cosine similarity between query and encoded vector.
    pub fn cosineSimilarity(self: *const ScalarQuantizer, query: []const f32, codes: []const u8) QuantizationError!f32 {
        if (!self.trained) return QuantizationError.NotTrained;
        if (query.len != self.dim) return QuantizationError.DimensionMismatch;

        const decoded = self.allocator.alloc(f32, self.dim) catch return QuantizationError.OutOfMemory;
        defer self.allocator.free(decoded);

        try self.decode(codes, decoded);
        return simd.cosineSimilarity(query, decoded);
    }

    /// Number of bytes needed to store one encoded vector.
    pub fn bytesPerVector(self: *const ScalarQuantizer) usize {
        return switch (self.bits) {
            4 => (self.dim + 1) / 2,
            8 => self.dim,
            16 => self.dim * 2,
            else => unreachable,
        };
    }

    /// Get statistics about this quantizer.
    pub fn getStats(self: *const ScalarQuantizer) Stats {
        const original_bytes = self.dim * @sizeOf(f32);
        const quantized_bytes = self.bytesPerVector();
        return .{
            .dimension = self.dim,
            .bits = self.bits,
            .compression_ratio = @as(f32, @floatFromInt(original_bytes)) / @as(f32, @floatFromInt(quantized_bytes)),
            .bytes_per_vector = quantized_bytes,
        };
    }

    fn levels(self: *const ScalarQuantizer) u16 {
        return (@as(u16, 1) << @as(u4, @intCast(self.bits))) - 1;
    }
};

// ============================================================================
// Helper functions
// ============================================================================

/// Thread-local decode buffer for avoiding allocation in hot path.
pub const DecodeBuffer = struct {
    buffer: []f32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, dim: usize) !DecodeBuffer {
        return .{
            .buffer = try allocator.alloc(f32, dim),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DecodeBuffer) void {
        self.allocator.free(self.buffer);
    }

    pub fn get(self: *DecodeBuffer) []f32 {
        return self.buffer;
    }
};

pub fn quantizeValue(value: f32, min_val: f32, max_val: f32, num_levels: u16) u8 {
    if (max_val <= min_val) return 0;
    const normalized = (value - min_val) / (max_val - min_val);
    const scaled = normalized * @as(f32, @floatFromInt(num_levels));
    const clamped = std.math.clamp(scaled, 0.0, @as(f32, @floatFromInt(num_levels)));
    return @intCast(@min(@as(u16, @intFromFloat(clamped)), num_levels));
}

pub fn quantizeValue16(value: f32, min_val: f32, max_val: f32) u16 {
    if (max_val <= min_val) return 0;
    const normalized = (value - min_val) / (max_val - min_val);
    const scaled = normalized * 65535.0;
    const clamped = std.math.clamp(scaled, 0.0, 65535.0);
    return @intFromFloat(clamped);
}

pub fn dequantizeValue(code: u8, min_val: f32, max_val: f32, num_levels: u16) f32 {
    if (max_val <= min_val or num_levels == 0) return min_val;
    const fraction = @as(f32, @floatFromInt(code)) / @as(f32, @floatFromInt(num_levels));
    return min_val + fraction * (max_val - min_val);
}

pub fn dequantizeValue16(code: u16, min_val: f32, max_val: f32) f32 {
    if (max_val <= min_val) return min_val;
    const fraction = @as(f32, @floatFromInt(code)) / 65535.0;
    return min_val + fraction * (max_val - min_val);
}

// ============================================================================
// Tests
// ============================================================================

test "scalar quantizer 8-bit basic" {
    const allocator = std.testing.allocator;

    var sq = try ScalarQuantizer.init(allocator, 4, .{ .bits = 8 });
    defer sq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 1.0, -1.0, 0.5 },
        &[_]f32{ 0.5, 0.0, 0.0, 1.0 },
        &[_]f32{ 1.0, -1.0, 1.0, 0.0 },
    };

    try sq.train(&vectors);

    const stats = sq.getStats();
    try std.testing.expectEqual(@as(usize, 4), stats.dimension);
    try std.testing.expectEqual(@as(u8, 8), stats.bits);
    try std.testing.expectEqual(@as(f32, 4.0), stats.compression_ratio);

    // Encode and decode
    var encoded: [4]u8 = undefined;
    const bytes_written = try sq.encode(&vectors[0], &encoded);
    try std.testing.expectEqual(@as(usize, 4), bytes_written);

    var decoded: [4]f32 = undefined;
    try sq.decode(&encoded, &decoded);

    // Check approximate reconstruction
    for (vectors[0], decoded) |orig, dec| {
        try std.testing.expect(@abs(orig - dec) < 0.02); // Allow small quantization error
    }
}

test "scalar quantizer 4-bit compression" {
    const allocator = std.testing.allocator;

    var sq = try ScalarQuantizer.init(allocator, 8, .{ .bits = 4 });
    defer sq.deinit();

    const vectors = [_][]const f32{
        &[_]f32{ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 },
        &[_]f32{ 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0 },
    };

    try sq.train(&vectors);

    const stats = sq.getStats();
    try std.testing.expectEqual(@as(usize, 4), stats.bytes_per_vector); // 8 dims / 2 = 4 bytes
    try std.testing.expectEqual(@as(f32, 8.0), stats.compression_ratio);

    var encoded: [4]u8 = undefined;
    _ = try sq.encode(&vectors[0], &encoded);

    var decoded: [8]f32 = undefined;
    try sq.decode(&encoded, &decoded);

    // 4-bit has larger quantization error
    for (vectors[0], decoded) |orig, dec| {
        try std.testing.expect(@abs(orig - dec) < 0.1);
    }
}
