//! Enhanced SIMD Vector Module
//! Advanced vector operations with modern Zig patterns and performance optimizations

const std = @import("std");
const core = @import("../core/mod.zig");
const Allocator = core.Allocator;
const Timer = core.Timer;

/// Vector types with enhanced features
pub const VectorType = enum {
    f32,
    f64,
    i32,
    i64,
    u32,
    u64,
};

/// SIMD configuration for different architectures
pub const SIMDConfig = struct {
    vector_width: usize,
    alignment: usize,
    has_avx2: bool,
    has_avx512: bool,
    has_sse4: bool,
    has_neon: bool,

    pub fn detect() SIMDConfig {
        const builtin = @import("builtin");
        const target = builtin.target;

        return switch (target.cpu.arch) {
            .x86_64 => .{
                .vector_width = if (std.Target.x86.featureSetHas(target.cpu.features, .avx512f)) 64 else if (std.Target.x86.featureSetHas(target.cpu.features, .avx2)) 32 else 16,
                .alignment = if (std.Target.x86.featureSetHas(target.cpu.features, .avx512f)) 64 else if (std.Target.x86.featureSetHas(target.cpu.features, .avx2)) 32 else 16,
                .has_avx2 = std.Target.x86.featureSetHas(target.cpu.features, .avx2),
                .has_avx512 = std.Target.x86.featureSetHas(target.cpu.features, .avx512f),
                .has_sse4 = std.Target.x86.featureSetHas(target.cpu.features, .sse4_1),
                .has_neon = false,
            },
            .aarch64 => .{
                .vector_width = if (std.Target.aarch64.featureSetHas(target.cpu.features, .sve)) 64 else 16,
                .alignment = if (std.Target.aarch64.featureSetHas(target.cpu.features, .sve)) 64 else 16,
                .has_avx2 = false,
                .has_avx512 = false,
                .has_sse4 = false,
                .has_neon = true,
            },
            .wasm32, .wasm64 => .{
                .vector_width = 16,
                .alignment = 16,
                .has_avx2 = false,
                .has_avx512 = false,
                .has_sse4 = false,
                .has_neon = false,
            },
            else => .{
                .vector_width = 1,
                .alignment = @sizeOf(f32),
                .has_avx2 = false,
                .has_avx512 = false,
                .has_sse4 = false,
                .has_neon = false,
            },
        };
    }
};

/// Enhanced vector structure
pub fn Vector(comptime T: type) type {
    return struct {
        data: []T,
        allocator: Allocator,
        config: SIMDConfig,

        const Self = @This();
        const VectorT = @Vector(4, T);

        /// Initialize vector with size
        pub fn init(allocator: Allocator, size: usize) !Self {
            const config = SIMDConfig.detect();
            const aligned_size = (size + config.vector_width - 1) & ~(config.vector_width - 1);
            const data = try allocator.alloc(T, aligned_size);

            return Self{
                .data = data,
                .allocator = allocator,
                .config = config,
            };
        }

        /// Initialize vector with data
        pub fn initWithData(allocator: Allocator, data: []const T) !Self {
            const config = SIMDConfig.detect();
            const aligned_size = (data.len + config.vector_width - 1) & ~(config.vector_width - 1);
            const vec_data = try allocator.alloc(T, aligned_size);

            @memcpy(vec_data[0..data.len], data);
            if (aligned_size > data.len) {
                @memset(vec_data[data.len..], 0);
            }

            return Self{
                .data = vec_data,
                .allocator = allocator,
                .config = config,
            };
        }

        /// Deinitialize vector
        pub fn deinit(self: *Self) void {
            self.allocator.free(self.data);
        }

        /// Get vector length
        pub fn len(self: *const Self) usize {
            return self.data.len;
        }

        /// Get element at index
        pub fn get(self: *const Self, index: usize) T {
            return self.data[index];
        }

        /// Set element at index
        pub fn set(self: *Self, index: usize, value: T) void {
            self.data[index] = value;
        }

        /// Fill vector with value
        pub fn fill(self: *Self, value: T) void {
            @memset(self.data, value);
        }

        /// Copy data from another vector
        pub fn copyFrom(self: *Self, other: *const Self) void {
            const min_len = @min(self.data.len, other.data.len);
            @memcpy(self.data[0..min_len], other.data[0..min_len]);
        }

        /// Add another vector element-wise
        pub fn add(self: *Self, other: *const Self) void {
            const min_len = @min(self.data.len, other.data.len);
            const simd_len = (min_len / 4) * 4;

            // SIMD operations
            var i: usize = 0;
            while (i < simd_len) : (i += 4) {
                const a: VectorT = self.data[i..][0..4].*;
                const b: VectorT = other.data[i..][0..4].*;
                const result = a + b;
                self.data[i..][0..4].* = result;
            }

            // Scalar remainder
            while (i < min_len) : (i += 1) {
                self.data[i] += other.data[i];
            }
        }

        /// Subtract another vector element-wise
        pub fn sub(self: *Self, other: *const Self) void {
            const min_len = @min(self.data.len, other.data.len);
            const simd_len = (min_len / 4) * 4;

            // SIMD operations
            var i: usize = 0;
            while (i < simd_len) : (i += 4) {
                const a: VectorT = self.data[i..][0..4].*;
                const b: VectorT = other.data[i..][0..4].*;
                const result = a - b;
                self.data[i..][0..4].* = result;
            }

            // Scalar remainder
            while (i < min_len) : (i += 1) {
                self.data[i] -= other.data[i];
            }
        }

        /// Multiply by scalar
        pub fn scale(self: *Self, scalar: T) void {
            const simd_len = (self.data.len / 4) * 4;

            // SIMD operations
            var i: usize = 0;
            while (i < simd_len) : (i += 4) {
                const a: VectorT = self.data[i..][0..4].*;
                const scalar_vec: VectorT = @splat(scalar);
                const result = a * scalar_vec;
                self.data[i..][0..4].* = result;
            }

            // Scalar remainder
            while (i < self.data.len) : (i += 1) {
                self.data[i] *= scalar;
            }
        }

        /// Element-wise multiplication
        pub fn mul(self: *Self, other: *const Self) void {
            const min_len = @min(self.data.len, other.data.len);
            const simd_len = (min_len / 4) * 4;

            // SIMD operations
            var i: usize = 0;
            while (i < simd_len) : (i += 4) {
                const a: VectorT = self.data[i..][0..4].*;
                const b: VectorT = other.data[i..][0..4].*;
                const result = a * b;
                self.data[i..][0..4].* = result;
            }

            // Scalar remainder
            while (i < min_len) : (i += 1) {
                self.data[i] *= other.data[i];
            }
        }

        /// Dot product with another vector
        pub fn dot(self: *const Self, other: *const Self) T {
            const min_len = @min(self.data.len, other.data.len);
            const simd_len = (min_len / 4) * 4;

            var sum: T = 0;

            // SIMD operations
            var i: usize = 0;
            while (i < simd_len) : (i += 4) {
                const a: VectorT = self.data[i..][0..4].*;
                const b: VectorT = other.data[i..][0..4].*;
                const product = a * b;
                sum += @reduce(.Add, product);
            }

            // Scalar remainder
            while (i < min_len) : (i += 1) {
                sum += self.data[i] * other.data[i];
            }

            return sum;
        }

        /// Compute magnitude (L2 norm)
        pub fn magnitude(self: *const Self) T {
            const dot_product = self.dot(self);
            return @sqrt(dot_product);
        }

        /// Normalize vector to unit length
        pub fn normalize(self: *Self) void {
            const mag = self.magnitude();
            if (mag > 0) {
                self.scale(1.0 / mag);
            }
        }

        /// Apply function to each element
        pub fn map(self: *Self, func: fn (T) T) void {
            for (self.data) |*element| {
                element.* = func(element.*);
            }
        }

        /// Apply function to each element with index
        pub fn mapIndexed(self: *Self, func: fn (usize, T) T) void {
            for (self.data, 0..) |*element, i| {
                element.* = func(i, element.*);
            }
        }

        /// Reduce vector to single value
        pub fn reduce(self: *const Self, initial: T, func: fn (T, T) T) T {
            var result = initial;
            for (self.data) |element| {
                result = func(result, element);
            }
            return result;
        }

        /// Find minimum value
        pub fn min(self: *const Self) T {
            return self.reduce(self.data[0], std.math.min);
        }

        /// Find maximum value
        pub fn max(self: *const Self) T {
            return self.reduce(self.data[0], std.math.max);
        }

        /// Compute sum of all elements
        pub fn sum(self: *const Self) T {
            return self.reduce(0, struct {
                fn add(a: T, b: T) T {
                    return a + b;
                }
            }.add);
        }

        /// Compute mean of all elements
        pub fn mean(self: *const Self) T {
            return self.sum() / @as(T, @floatFromInt(self.data.len));
        }

        /// Compute variance
        pub fn variance(self: *const Self) T {
            const mean_val = self.mean();
            var sum_sq: T = 0;

            for (self.data) |element| {
                const diff = element - mean_val;
                sum_sq += diff * diff;
            }

            return sum_sq / @as(T, @floatFromInt(self.data.len));
        }

        /// Compute standard deviation
        pub fn stdDev(self: *const Self) T {
            return @sqrt(self.variance());
        }

        /// Apply activation function
        pub fn activate(self: *Self, activation: ActivationFunction) void {
            switch (activation) {
                .relu => self.map(struct {
                    fn relu(x: T) T {
                        return @max(0, x);
                    }
                }.relu),
                .sigmoid => self.map(struct {
                    fn sigmoid(x: T) T {
                        return 1.0 / (1.0 + std.math.exp(-x));
                    }
                }.sigmoid),
                .tanh => self.map(struct {
                    fn tanh(x: T) T {
                        return std.math.tanh(x);
                    }
                }.tanh),
                .softplus => self.map(struct {
                    fn softplus(x: T) T {
                        return std.math.log(1.0 + std.math.exp(x));
                    }
                }.softplus),
            }
        }

        /// Generate random values
        pub fn randomize(self: *Self, rng: std.rand.Random, min_val: T, max_val: T) void {
            for (self.data) |*element| {
                element.* = min_val + (max_val - min_val) * rng.float(T);
            }
        }

        /// Create slice view
        pub fn slice(self: *Self, start: usize, end: usize) []T {
            return self.data[start..end];
        }

        /// Resize vector
        pub fn resize(self: *Self, new_size: usize) !void {
            if (new_size != self.data.len) {
                const new_data = try self.allocator.realloc(self.data, new_size);
                self.data = new_data;
            }
        }
    };
}

/// Activation functions
pub const ActivationFunction = enum {
    relu,
    sigmoid,
    tanh,
    softplus,
};

/// Vector operations namespace
pub const VectorOps = struct {
    /// Element-wise addition
    pub fn add(comptime T: type, a: []T, b: []T, result: []T) void {
        const min_len = @min(@min(a.len, b.len), result.len);
        const simd_len = (min_len / 4) * 4;

        // SIMD operations
        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const va: @Vector(4, T) = a[i..][0..4].*;
            const vb: @Vector(4, T) = b[i..][0..4].*;
            const sum = va + vb;
            result[i..][0..4].* = sum;
        }

        // Scalar remainder
        while (i < min_len) : (i += 1) {
            result[i] = a[i] + b[i];
        }
    }

    /// Element-wise subtraction
    pub fn sub(comptime T: type, a: []T, b: []T, result: []T) void {
        const min_len = @min(@min(a.len, b.len), result.len);
        const simd_len = (min_len / 4) * 4;

        // SIMD operations
        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const va: @Vector(4, T) = a[i..][0..4].*;
            const vb: @Vector(4, T) = b[i..][0..4].*;
            const diff = va - vb;
            result[i..][0..4].* = diff;
        }

        // Scalar remainder
        while (i < min_len) : (i += 1) {
            result[i] = a[i] - b[i];
        }
    }

    /// Element-wise multiplication
    pub fn mul(comptime T: type, a: []T, b: []T, result: []T) void {
        const min_len = @min(@min(a.len, b.len), result.len);
        const simd_len = (min_len / 4) * 4;

        // SIMD operations
        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const va: @Vector(4, T) = a[i..][0..4].*;
            const vb: @Vector(4, T) = b[i..][0..4].*;
            const product = va * vb;
            result[i..][0..4].* = product;
        }

        // Scalar remainder
        while (i < min_len) : (i += 1) {
            result[i] = a[i] * b[i];
        }
    }

    /// Scalar multiplication
    pub fn scale(comptime T: type, a: []T, scalar: T, result: []T) void {
        const min_len = @min(a.len, result.len);
        const simd_len = (min_len / 4) * 4;

        // SIMD operations
        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const va: @Vector(4, T) = a[i..][0..4].*;
            const scalar_vec: @Vector(4, T) = @splat(scalar);
            const product = va * scalar_vec;
            result[i..][0..4].* = product;
        }

        // Scalar remainder
        while (i < min_len) : (i += 1) {
            result[i] = a[i] * scalar;
        }
    }

    /// Dot product
    pub fn dot(comptime T: type, a: []T, b: []T) T {
        const min_len = @min(a.len, b.len);
        const simd_len = (min_len / 4) * 4;

        var sum: T = 0;

        // SIMD operations
        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const va: @Vector(4, T) = a[i..][0..4].*;
            const vb: @Vector(4, T) = b[i..][0..4].*;
            const product = va * vb;
            sum += @reduce(.Add, product);
        }

        // Scalar remainder
        while (i < min_len) : (i += 1) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    /// Cosine similarity
    pub fn cosineSimilarity(comptime T: type, a: []T, b: []T) T {
        const dot_product = dot(T, a, b);
        const norm_a = @sqrt(dot(T, a, a));
        const norm_b = @sqrt(dot(T, b, b));

        if (norm_a == 0 or norm_b == 0) return 0;
        return dot_product / (norm_a * norm_b);
    }

    /// Euclidean distance
    pub fn euclideanDistance(comptime T: type, a: []T, b: []T) T {
        const min_len = @min(a.len, b.len);
        const simd_len = (min_len / 4) * 4;

        var sum_sq: T = 0;

        // SIMD operations
        var i: usize = 0;
        while (i < simd_len) : (i += 4) {
            const va: @Vector(4, T) = a[i..][0..4].*;
            const vb: @Vector(4, T) = b[i..][0..4].*;
            const diff = va - vb;
            const diff_sq = diff * diff;
            sum_sq += @reduce(.Add, diff_sq);
        }

        // Scalar remainder
        while (i < min_len) : (i += 1) {
            const diff = a[i] - b[i];
            sum_sq += diff * diff;
        }

        return @sqrt(sum_sq);
    }
};

test "enhanced vector basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var vec = try Vector(f32).init(allocator, 8);
    defer vec.deinit();

    // Test initialization
    try testing.expectEqual(@as(usize, 8), vec.len());

    // Test set/get
    vec.set(0, 1.0);
    vec.set(1, 2.0);
    try testing.expectEqual(@as(f32, 1.0), vec.get(0));
    try testing.expectEqual(@as(f32, 2.0), vec.get(1));

    // Test fill
    vec.fill(5.0);
    try testing.expectEqual(@as(f32, 5.0), vec.get(0));
    try testing.expectEqual(@as(f32, 5.0), vec.get(7));
}

test "enhanced vector arithmetic" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var vec1 = try Vector(f32).initWithData(allocator, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer vec1.deinit();

    var vec2 = try Vector(f32).initWithData(allocator, &[_]f32{ 5.0, 6.0, 7.0, 8.0 });
    defer vec2.deinit();

    // Test addition
    vec1.add(&vec2);
    try testing.expectEqual(@as(f32, 6.0), vec1.get(0));
    try testing.expectEqual(@as(f32, 8.0), vec1.get(1));
    try testing.expectEqual(@as(f32, 10.0), vec1.get(2));
    try testing.expectEqual(@as(f32, 12.0), vec1.get(3));

    // Test scaling
    vec1.scale(0.5);
    try testing.expectEqual(@as(f32, 3.0), vec1.get(0));
    try testing.expectEqual(@as(f32, 4.0), vec1.get(1));
}

test "enhanced vector operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var vec1 = try Vector(f32).initWithData(allocator, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer vec1.deinit();

    var vec2 = try Vector(f32).initWithData(allocator, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer vec2.deinit();

    // Test dot product
    const dot_result = vec1.dot(&vec2);
    try testing.expectApproxEqAbs(@as(f32, 30.0), dot_result, 0.001);

    // Test magnitude
    const magnitude = vec1.magnitude();
    try testing.expectApproxEqAbs(@as(f32, 5.477), magnitude, 0.001);

    // Test normalization
    vec1.normalize();
    const normalized_magnitude = vec1.magnitude();
    try testing.expectApproxEqAbs(@as(f32, 1.0), normalized_magnitude, 0.001);
}
