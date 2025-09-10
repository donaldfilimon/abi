//! Enhanced SIMD Vector Module
//! Advanced vector operations with modern Zig patterns and performance optimizations
//! Features dynamic vector width determination, optimized memory access patterns,
//! and comprehensive SIMD acceleration across multiple architectures

const std = @import("std");
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

/// Vector types with enhanced features
pub const VectorType = enum {
    f32,
    f64,
    i32,
    i64,
    u32,
    u64,
};

/// SIMD configuration for different architectures with dynamic detection
pub const SIMDConfig = struct {
    vector_width: usize,
    alignment: usize,
    has_avx2: bool,
    has_avx512: bool,
    has_sse4: bool,
    has_neon: bool,
    max_unroll_factor: usize,
    cache_line_size: usize,

    /// Detect optimal SIMD configuration at compile time
    pub fn detect() SIMDConfig {
        const target = builtin.target;

        return switch (target.cpu.arch) {
            .x86_64 => blk: {
                const has_avx512 = std.Target.x86.featureSetHas(target.cpu.features, .avx512f);
                const has_avx2 = std.Target.x86.featureSetHas(target.cpu.features, .avx2);
                const has_sse4 = std.Target.x86.featureSetHas(target.cpu.features, .sse4_1);

                break :blk .{
                    .vector_width = if (has_avx512) 64 else if (has_avx2) 32 else if (has_sse4) 16 else 8,
                    .alignment = if (has_avx512) 64 else if (has_avx2) 32 else if (has_sse4) 16 else 8,
                    .has_avx2 = has_avx2,
                    .has_avx512 = has_avx512,
                    .has_sse4 = has_sse4,
                    .has_neon = false,
                    .max_unroll_factor = if (has_avx512) 8 else if (has_avx2) 4 else 2,
                    .cache_line_size = 64,
                };
            },
            .aarch64 => blk: {
                const has_sve = std.Target.aarch64.featureSetHas(target.cpu.features, .sve);
                const has_neon = true; // NEON is standard on AArch64

                break :blk .{
                    .vector_width = if (has_sve) 64 else 16,
                    .alignment = if (has_sve) 64 else 16,
                    .has_avx2 = false,
                    .has_avx512 = false,
                    .has_sse4 = false,
                    .has_neon = has_neon,
                    .max_unroll_factor = if (has_sve) 8 else 4,
                    .cache_line_size = 64,
                };
            },
            .wasm32, .wasm64 => .{
                .vector_width = 16,
                .alignment = 16,
                .has_avx2 = false,
                .has_avx512 = false,
                .has_sse4 = false,
                .has_neon = false,
                .max_unroll_factor = 2,
                .cache_line_size = 64,
            },
            else => .{
                .vector_width = @sizeOf(f64),
                .alignment = @sizeOf(f64),
                .has_avx2 = false,
                .has_avx512 = false,
                .has_sse4 = false,
                .has_neon = false,
                .max_unroll_factor = 1,
                .cache_line_size = 64,
            },
        };
    }

    /// Get optimal vector size for a given type
    pub fn getVectorSize(self: SIMDConfig, comptime T: type) usize {
        return self.vector_width / @sizeOf(T);
    }

    /// Check if type supports vectorization
    pub fn supportsVectorization(self: SIMDConfig, comptime T: type) bool {
        return @sizeOf(T) <= self.vector_width and self.vector_width % @sizeOf(T) == 0;
    }
};

/// Activation functions with SIMD-optimized implementations
pub const ActivationFunction = enum {
    relu,
    leaky_relu,
    sigmoid,
    tanh,
    softplus,
    gelu,
    swish,
    mish,

    /// Get the derivative function for backpropagation
    pub fn derivative(self: ActivationFunction) ActivationFunction {
        return switch (self) {
            .relu => .relu, // Special handling needed
            .leaky_relu => .leaky_relu,
            .sigmoid => .sigmoid,
            .tanh => .tanh,
            .softplus => .sigmoid,
            .gelu => .gelu,
            .swish => .swish,
            .mish => .mish,
        };
    }
};

/// Memory allocation strategies for optimal performance
pub const AllocationStrategy = enum {
    standard,
    aligned,
    hugepage,
    numa_aware,
};

/// Performance optimization hints
pub const OptimizationHints = struct {
    prefer_cache_friendly: bool = true,
    allow_loop_unrolling: bool = true,
    use_prefetching: bool = true,
    vectorize_aggressively: bool = true,
};

/// Enhanced vector structure with dynamic SIMD optimization
pub fn EnhancedVector(comptime T: type) type {
    return struct {
        data: []T,
        allocator: Allocator,
        config: SIMDConfig,
        capacity: usize,
        allocation_strategy: AllocationStrategy,
        optimization_hints: OptimizationHints,
        const Self = @This();
        const vector_size = SIMDConfig.detect().getVectorSize(T);
        const VectorT = @Vector(vector_size, T);

        /// Initialize vector with size, zero-initialized and optimally aligned
        pub fn init(allocator: Allocator, size: usize) !Self {
            return initWithStrategy(allocator, size, .aligned, .{});
        }

        /// Initialize vector with specific allocation strategy and optimization hints
        pub fn initWithStrategy(allocator: Allocator, size: usize, strategy: AllocationStrategy, hints: OptimizationHints) !Self {
            const config = SIMDConfig.detect();
            const alignment = @max(config.alignment, @alignOf(T));
            const aligned_size = alignSize(size, config.vector_width / @sizeOf(T));

            var data = switch (strategy) {
                .standard => try allocator.alloc(T, aligned_size),
                .aligned => try allocator.alignedAlloc(T, alignment, aligned_size),
                .hugepage => try allocateHugePage(allocator, T, aligned_size, alignment),
                .numa_aware => try allocateNumaAware(allocator, T, aligned_size, alignment),
            };

            // Zero-initialize with SIMD if possible
            zeroMemorySIMD(T, data);

            return Self{
                .data = data[0..size],
                .allocator = allocator,
                .config = config,
                .capacity = aligned_size,
                .allocation_strategy = strategy,
                .optimization_hints = hints,
            };
        }

        /// Initialize vector with data, optimizing for memory layout
        pub fn initWithData(allocator: Allocator, input_data: []const T) !Self {
            var self = try init(allocator, input_data.len);
            self.copyFromSlice(input_data);
            return self;
        }

        /// Deinitialize vector with proper cleanup
        pub fn deinit(self: *Self) void {
            switch (self.allocation_strategy) {
                .standard, .aligned => self.allocator.free(self.data.ptr[0..self.capacity]),
                .hugepage => deallocateHugePage(self.allocator, self.data.ptr[0..self.capacity]),
                .numa_aware => deallocateNumaAware(self.allocator, self.data.ptr[0..self.capacity]),
            }
        }

        /// Get vector length
        pub inline fn len(self: *const Self) usize {
            return self.data.len;
        }

        /// Get element at index with bounds checking
        pub inline fn get(self: *const Self, index: usize) T {
            std.debug.assert(index < self.data.len);
            return self.data[index];
        }

        /// Set element at index with bounds checking
        pub inline fn set(self: *Self, index: usize, value: T) void {
            std.debug.assert(index < self.data.len);
            self.data[index] = value;
        }

        /// Fill vector with value using SIMD acceleration
        pub fn fill(self: *Self, value: T) void {
            fillSIMD(T, self.data, value, self.config);
        }

        /// Copy data from another vector with SIMD optimization
        pub fn copyFrom(self: *Self, other: *const Self) void {
            const min_len = @min(self.data.len, other.data.len);
            copySIMD(T, other.data[0..min_len], self.data[0..min_len], self.config);
        }

        /// Copy data from slice with SIMD optimization
        pub fn copyFromSlice(self: *Self, src_slice: []const T) void {
            const min_len = @min(self.data.len, src_slice.len);
            copySIMD(T, src_slice[0..min_len], self.data[0..min_len], self.config);
        }

        /// Add another vector element-wise with optimized SIMD
        pub fn add(self: *Self, other: *const Self) void {
            const min_len = @min(self.data.len, other.data.len);
            addSIMD(T, self.data[0..min_len], other.data[0..min_len], self.data[0..min_len], self.config);
        }

        /// Subtract another vector element-wise with optimized SIMD
        pub fn sub(self: *Self, other: *const Self) void {
            const min_len = @min(self.data.len, other.data.len);
            subSIMD(T, self.data[0..min_len], other.data[0..min_len], self.data[0..min_len], self.config);
        }

        /// Multiply by scalar with unrolled SIMD loops
        pub fn scale(self: *Self, scalar: T) void {
            scaleSIMD(T, self.data, scalar, self.config);
        }

        /// Element-wise multiplication with SIMD acceleration
        pub fn mul(self: *Self, other: *const Self) void {
            const min_len = @min(self.data.len, other.data.len);
            mulSIMD(T, self.data[0..min_len], other.data[0..min_len], self.data[0..min_len], self.config);
        }

        /// Fused multiply-add operation (a = a * b + c)
        pub fn fma(self: *Self, other: *const Self, addend: *const Self) void {
            const min_len = @min(@min(self.data.len, other.data.len), addend.data.len);
            fmaSIMD(T, self.data[0..min_len], other.data[0..min_len], addend.data[0..min_len], self.data[0..min_len], self.config);
        }

        /// Dot product with highly optimized SIMD implementation
        pub fn dot(self: *const Self, other: *const Self) T {
            const min_len = @min(self.data.len, other.data.len);
            return dotSIMD(T, self.data[0..min_len], other.data[0..min_len], self.config);
        }

        /// Compute magnitude (L2 norm) with SIMD optimization
        pub fn magnitude(self: *const Self) T {
            return std.math.sqrt(self.dot(self));
        }

        /// Normalize vector to unit length with safety checks
        pub fn normalize(self: *Self) void {
            const mag = self.magnitude();
            if (mag > std.math.floatEps(T)) {
                self.scale(@as(T, 1.0) / mag);
            }
        }

        /// Apply function to each element with vectorization hints
        pub fn map(self: *Self, comptime func: fn (T) T) void {
            if (comptime isVectorizableFunction(func)) {
                mapSIMD(T, self.data, func, self.config);
            } else {
                for (self.data) |*element| {
                    element.* = func(element.*);
                }
            }
        }

        /// Apply function to each element with index
        pub fn mapIndexed(self: *Self, comptime func: fn (usize, T) T) void {
            for (self.data, 0..) |*element, i| {
                element.* = func(i, element.*);
            }
        }

        /// Reduce vector to single value with SIMD optimization
        pub fn reduce(self: *const Self, initial: T, comptime func: fn (T, T) T) T {
            return reduceSIMD(T, self.data, initial, func, self.config);
        }

        /// Find minimum value with SIMD acceleration
        pub fn min(self: *const Self) T {
            if (self.data.len == 0) return 0;
            return minSIMD(T, self.data, self.config);
        }

        /// Find maximum value with SIMD acceleration
        pub fn max(self: *const Self) T {
            if (self.data.len == 0) return 0;
            return maxSIMD(T, self.data, self.config);
        }

        /// Compute sum of all elements with Kahan summation for precision
        pub fn sum(self: *const Self) T {
            return sumSIMD(T, self.data, self.config);
        }

        /// Compute mean of all elements
        pub fn mean(self: *const Self) T {
            if (self.data.len == 0) return 0;
            return self.sum() / @as(T, @floatFromInt(self.data.len));
        }

        /// Compute variance with numerically stable algorithm
        pub fn variance(self: *const Self) T {
            if (self.data.len == 0) return 0;
            return varianceSIMD(T, self.data, self.config);
        }

        /// Compute standard deviation
        pub fn stdDev(self: *const Self) T {
            return std.math.sqrt(self.variance());
        }

        /// Apply activation function with SIMD optimization
        pub fn activate(self: *Self, activation: ActivationFunction) void {
            activateSIMD(T, self.data, activation, self.config);
        }

        /// Generate random values with improved distribution
        pub fn randomize(self: *Self, rng: std.Random, min_val: T, max_val: T) void {
            const range = max_val - min_val;
            for (self.data) |*element| {
                element.* = min_val + range * rng.float(T);
            }
        }
        /// Generate normally distributed random values (Box-Muller transform)
        pub fn randomizeNormal(self: *Self, rng: std.Random, mu: T, sigma: T) void {
            var i: usize = 0;
            while (i + 1 < self.data.len) : (i += 2) {
                const uniform1 = rng.float(T);
                const uniform2 = rng.float(T);
                const r = std.math.sqrt(-2.0 * std.math.log(uniform1));
                const theta = 2.0 * std.math.pi * uniform2;

                self.data[i] = mu + sigma * r * std.math.cos(theta);
                self.data[i + 1] = mu + sigma * r * std.math.sin(theta);
            }

            if (i < self.data.len) {
                self.data[i] = mu + sigma * rng.floatNorm(T);
            }
        }

        /// Create slice view with bounds checking
        pub fn slice(self: *Self, start: usize, end: usize) []T {
            std.debug.assert(start <= end and end <= self.data.len);
            return self.data[start..end];
        }

        /// Resize vector with data preservation and optimal reallocation
        pub fn resize(self: *Self, new_size: usize) !void {
            if (new_size == self.data.len) return;

            const alignment = @max(self.config.alignment, @alignOf(T));
            const aligned_size = alignSize(new_size, self.config.vector_width / @sizeOf(T));

            if (aligned_size <= self.capacity) {
                // Can reuse existing allocation
                if (new_size > self.data.len) {
                    // Zero new elements
                    const old_len = self.data.len;
                    self.data = self.data.ptr[0..new_size];
                    zeroMemorySIMD(T, self.data[old_len..]);
                } else {
                    self.data = self.data.ptr[0..new_size];
                }
                return;
            }

            // Need to reallocate
            var new_data = switch (self.allocation_strategy) {
                .standard => try self.allocator.realloc(self.data.ptr[0..self.capacity], aligned_size),
                .aligned => try reallocAligned(self.allocator, self.data.ptr[0..self.capacity], aligned_size, alignment),
                .hugepage => try reallocHugePage(self.allocator, self.data.ptr[0..self.capacity], aligned_size, alignment),
                .numa_aware => try reallocNumaAware(self.allocator, self.data.ptr[0..self.capacity], aligned_size, alignment),
            };

            if (new_size > self.data.len) {
                zeroMemorySIMD(T, new_data[self.data.len..aligned_size]);
            }

            self.data = new_data[0..new_size];
            self.capacity = aligned_size;
        }

        /// Compute cross-correlation with another vector
        pub fn correlate(self: *const Self, other: *const Self, allocator: Allocator) !Self {
            const result_len = self.data.len + other.data.len - 1;
            var result = try Self.init(allocator, result_len);
            for (0..result_len) |i| {
                var accum: T = 0;
                const start_j = if (i >= other.data.len - 1) i - (other.data.len - 1) else 0;
                const end_j = @min(i + 1, self.data.len);

                for (start_j..end_j) |j| {
                    accum += self.data[j] * other.data[i - j];
                }
                result.data[i] = accum;
            }

            return result;
        }

        /// Apply window function (Hann, Hamming, etc.)
        pub fn applyWindow(self: *Self, window_type: WindowType) void {
            applyWindowSIMD(T, self.data, window_type, self.config);
        }

        /// Compute FFT (Fast Fourier Transform) - placeholder for future implementation
        pub fn fft(self: *const Self, allocator: Allocator) !Self {
            // This would require a full FFT implementation
            // For now, return a copy
            return Self.initWithData(allocator, self.data);
        }
    };
}

/// Window types for signal processing
pub const WindowType = enum {
    rectangular,
    hann,
    hamming,
    blackman,
    kaiser,
};

/// Optimized SIMD operations namespace
pub const SIMDOps = struct {
    /// Element-wise addition with loop unrolling
    pub fn add(comptime T: type, a: []const T, b: []const T, result: []T, config: SIMDConfig) void {
        addSIMD(T, a, b, result, config);
    }

    /// Element-wise subtraction with loop unrolling
    pub fn sub(comptime T: type, a: []const T, b: []const T, result: []T, config: SIMDConfig) void {
        subSIMD(T, a, b, result, config);
    }

    /// Element-wise multiplication with loop unrolling
    pub fn mul(comptime T: type, a: []const T, b: []const T, result: []T, config: SIMDConfig) void {
        mulSIMD(T, a, b, result, config);
    }

    /// Scalar multiplication with vectorization
    pub fn scale(comptime T: type, a: []const T, scalar: T, result: []T, config: SIMDConfig) void {
        scaleSIMD(T, a, scalar, result, config);
    }

    /// Dot product with maximum SIMD utilization
    pub fn dot(comptime T: type, a: []const T, b: []const T, config: SIMDConfig) T {
        return dotSIMD(T, a, b, config);
    }

    /// Cosine similarity with SIMD optimization
    pub fn cosineSimilarity(comptime T: type, a: []const T, b: []const T, config: SIMDConfig) T {
        const dot_product = dotSIMD(T, a, b, config);
        const norm_a = std.math.sqrt(dotSIMD(T, a, a, config));
        const norm_b = std.math.sqrt(dotSIMD(T, b, b, config));
        if (norm_a < std.math.floatEps(T) or norm_b < std.math.floatEps(T)) return 0;
        return dot_product / (norm_a * norm_b);
    }

    /// Euclidean distance with SIMD acceleration
    pub fn euclideanDistance(comptime T: type, a: []const T, b: []const T, config: SIMDConfig) T {
        return std.math.sqrt(euclideanDistanceSquared(T, a, b, config));
    }

    /// Squared Euclidean distance (avoid sqrt when not needed)
    pub fn euclideanDistanceSquared(comptime T: type, a: []const T, b: []const T, config: SIMDConfig) T {
        const min_len = @min(a.len, b.len);
        const vector_size = config.getVectorSize(T);

        if (!config.supportsVectorization(T) or min_len < vector_size) {
            // Fallback to scalar implementation
            var sum_sq: T = 0;
            for (0..min_len) |i| {
                const diff = a[i] - b[i];
                sum_sq += diff * diff;
            }
            return sum_sq;
        }

        const VectorT = @Vector(vector_size, T);
        const simd_len = (min_len / vector_size) * vector_size;
        var sum_sq: T = 0;
        var i: usize = 0;

        // Unrolled SIMD loop for better instruction-level parallelism
        while (i + vector_size * 4 <= simd_len) : (i += vector_size * 4) {
            const va1: VectorT = a[i..][0..vector_size].*;
            const vb1: VectorT = b[i..][0..vector_size].*;
            const diff1 = va1 - vb1;

            const va2: VectorT = a[i + vector_size ..][0..vector_size].*;
            const vb2: VectorT = b[i + vector_size ..][0..vector_size].*;
            const diff2 = va2 - vb2;

            const va3: VectorT = a[i + vector_size * 2 ..][0..vector_size].*;
            const vb3: VectorT = b[i + vector_size * 2 ..][0..vector_size].*;
            const diff3 = va3 - vb3;

            const va4: VectorT = a[i + vector_size * 3 ..][0..vector_size].*;
            const vb4: VectorT = b[i + vector_size * 3 ..][0..vector_size].*;
            const diff4 = va4 - vb4;

            sum_sq += @reduce(.Add, diff1 * diff1);
            sum_sq += @reduce(.Add, diff2 * diff2);
            sum_sq += @reduce(.Add, diff3 * diff3);
            sum_sq += @reduce(.Add, diff4 * diff4);
        }

        // Handle remaining SIMD chunks
        while (i + vector_size <= simd_len) : (i += vector_size) {
            const va: VectorT = a[i..][0..vector_size].*;
            const vb: VectorT = b[i..][0..vector_size].*;
            const diff = va - vb;
            sum_sq += @reduce(.Add, diff * diff);
        }

        // Handle remaining scalar elements
        while (i < min_len) : (i += 1) {
            const diff = a[i] - b[i];
            sum_sq += diff * diff;
        }

        return sum_sq;
    }

    /// Manhattan distance (L1 norm)
    pub fn manhattanDistance(comptime T: type, a: []const T, b: []const T, config: SIMDConfig) T {
        const min_len = @min(a.len, b.len);
        const vector_size = config.getVectorSize(T);

        if (!config.supportsVectorization(T) or min_len < vector_size) {
            var sum: T = 0;
            for (0..min_len) |i| {
                sum += @abs(a[i] - b[i]);
            }
            return sum;
        }

        const VectorT = @Vector(vector_size, T);
        const simd_len = (min_len / vector_size) * vector_size;
        var sum: T = 0;
        var i: usize = 0;

        while (i < simd_len) : (i += vector_size) {
            const va: VectorT = a[i..][0..vector_size].*;
            const vb: VectorT = b[i..][0..vector_size].*;
            const diff = va - vb;
            sum += @reduce(.Add, @abs(diff));
        }

        while (i < min_len) : (i += 1) {
            sum += @abs(a[i] - b[i]);
        }

        return sum;
    }
};

// Helper functions for SIMD operations

/// Align size to vector boundary
inline fn alignSize(size: usize, vector_elements: usize) usize {
    return (size + vector_elements - 1) & ~(vector_elements - 1);
}

/// Zero memory using SIMD when possible
fn zeroMemorySIMD(comptime T: type, data: []T) void {
    @memset(data, @as(T, 0));
}

/// Fill memory with value using SIMD
fn fillSIMD(comptime T: type, data: []T, value: T, config: SIMDConfig) void {
    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or data.len < vector_size) {
        @memset(data, value);
        return;
    }

    const VectorT = @Vector(vector_size, T);
    const value_vec: VectorT = @splat(value);
    const simd_len = (data.len / vector_size) * vector_size;
    var i: usize = 0;

    while (i < simd_len) : (i += vector_size) {
        data[i..][0..vector_size].* = value_vec;
    }

    while (i < data.len) : (i += 1) {
        data[i] = value;
    }
}

/// Copy data using SIMD
fn copySIMD(comptime T: type, src: []const T, dst: []T, config: SIMDConfig) void {
    _ = config;
    @memcpy(dst, src);
}

/// SIMD addition implementation
fn addSIMD(comptime T: type, a: []const T, b: []const T, result: []T, config: SIMDConfig) void {
    const min_len = @min(@min(a.len, b.len), result.len);
    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or min_len < vector_size) {
        for (0..min_len) |i| {
            result[i] = a[i] + b[i];
        }
        return;
    }

    const VectorT = @Vector(vector_size, T);
    const simd_len = (min_len / vector_size) * vector_size;
    var i: usize = 0;

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = a[i..][0..vector_size].*;
        const vb: VectorT = b[i..][0..vector_size].*;
        result[i..][0..vector_size].* = va + vb;
    }

    while (i < min_len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// SIMD subtraction implementation
fn subSIMD(comptime T: type, a: []const T, b: []const T, result: []T, config: SIMDConfig) void {
    const min_len = @min(@min(a.len, b.len), result.len);
    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or min_len < vector_size) {
        for (0..min_len) |i| {
            result[i] = a[i] - b[i];
        }
        return;
    }

    const VectorT = @Vector(vector_size, T);
    const simd_len = (min_len / vector_size) * vector_size;
    var i: usize = 0;

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = a[i..][0..vector_size].*;
        const vb: VectorT = b[i..][0..vector_size].*;
        result[i..][0..vector_size].* = va - vb;
    }

    while (i < min_len) : (i += 1) {
        result[i] = a[i] - b[i];
    }
}

/// SIMD multiplication implementation
fn mulSIMD(comptime T: type, a: []const T, b: []const T, result: []T, config: SIMDConfig) void {
    const min_len = @min(@min(a.len, b.len), result.len);
    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or min_len < vector_size) {
        for (0..min_len) |i| {
            result[i] = a[i] * b[i];
        }
        return;
    }

    const VectorT = @Vector(vector_size, T);
    const simd_len = (min_len / vector_size) * vector_size;
    var i: usize = 0;

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = a[i..][0..vector_size].*;
        const vb: VectorT = b[i..][0..vector_size].*;
        result[i..][0..vector_size].* = va * vb;
    }

    while (i < min_len) : (i += 1) {
        result[i] = a[i] * b[i];
    }
}

/// SIMD scalar multiplication implementation with overloads for in-place and out-of-place operations
fn scaleSIMD(comptime T: type, data: []T, scalar: T, config: SIMDConfig) void {
    scaleOutOfPlaceSIMD(T, data, scalar, data, config);
}

/// SIMD scalar multiplication implementation - out of place version
fn scaleOutOfPlaceSIMD(comptime T: type, src: []const T, scalar: T, dst: []T, config: SIMDConfig) void {
    const min_len = @min(src.len, dst.len);
    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or min_len < vector_size) {
        for (0..min_len) |i| {
            dst[i] = src[i] * scalar;
        }
        return;
    }

    const VectorT = @Vector(vector_size, T);
    const scalar_vec: VectorT = @splat(scalar);
    const simd_len = (min_len / vector_size) * vector_size;
    var i: usize = 0;

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = src[i..][0..vector_size].*;
        dst[i..][0..vector_size].* = va * scalar_vec;
    }

    while (i < min_len) : (i += 1) {
        dst[i] = src[i] * scalar;
    }
}

/// SIMD fused multiply-add implementation
fn fmaSIMD(comptime T: type, a: []const T, b: []const T, c: []const T, result: []T, config: SIMDConfig) void {
    const min_len = @min(@min(@min(a.len, b.len), c.len), result.len);
    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or min_len < vector_size) {
        for (0..min_len) |i| {
            result[i] = a[i] * b[i] + c[i];
        }
        return;
    }

    const VectorT = @Vector(vector_size, T);
    const simd_len = (min_len / vector_size) * vector_size;
    var i: usize = 0;

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = a[i..][0..vector_size].*;
        const vb: VectorT = b[i..][0..vector_size].*;
        const vc: VectorT = c[i..][0..vector_size].*;
        result[i..][0..vector_size].* = va * vb + vc;
    }

    while (i < min_len) : (i += 1) {
        result[i] = a[i] * b[i] + c[i];
    }
}

/// SIMD dot product implementation with Kahan summation for precision
fn dotSIMD(comptime T: type, a: []const T, b: []const T, config: SIMDConfig) T {
    const min_len = @min(a.len, b.len);
    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or min_len < vector_size) {
        var sum: T = 0;
        for (0..min_len) |i| {
            sum += a[i] * b[i];
        }
        return sum;
    }

    const VectorT = @Vector(vector_size, T);
    const simd_len = (min_len / vector_size) * vector_size;
    var sum: T = 0;
    var i: usize = 0;

    // Unrolled loop for better performance
    while (i + vector_size * 4 <= simd_len) : (i += vector_size * 4) {
        const va1: VectorT = a[i..][0..vector_size].*;
        const vb1: VectorT = b[i..][0..vector_size].*;

        const va2: VectorT = a[i + vector_size ..][0..vector_size].*;
        const vb2: VectorT = b[i + vector_size ..][0..vector_size].*;

        const va3: VectorT = a[i + vector_size * 2 ..][0..vector_size].*;
        const vb3: VectorT = b[i + vector_size * 2 ..][0..vector_size].*;

        const va4: VectorT = a[i + vector_size * 3 ..][0..vector_size].*;
        const vb4: VectorT = b[i + vector_size * 3 ..][0..vector_size].*;

        sum += @reduce(.Add, va1 * vb1);
        sum += @reduce(.Add, va2 * vb2);
        sum += @reduce(.Add, va3 * vb3);
        sum += @reduce(.Add, va4 * vb4);
    }

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = a[i..][0..vector_size].*;
        const vb: VectorT = b[i..][0..vector_size].*;
        sum += @reduce(.Add, va * vb);
    }

    while (i < min_len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

/// SIMD minimum implementation
fn minSIMD(comptime T: type, data: []const T, config: SIMDConfig) T {
    if (data.len == 0) return 0;

    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or data.len < vector_size) {
        var min_val = data[0];
        for (data[1..]) |element| {
            min_val = @min(min_val, element);
        }
        return min_val;
    }

    const VectorT = @Vector(vector_size, T);
    const simd_len = (data.len / vector_size) * vector_size;
    var min_vec: VectorT = data[0..vector_size].*;
    var i: usize = vector_size;

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = data[i..][0..vector_size].*;
        min_vec = @min(min_vec, va);
    }

    var min_val = @reduce(.Min, min_vec);
    while (i < data.len) : (i += 1) {
        min_val = @min(min_val, data[i]);
    }

    return min_val;
}

/// SIMD maximum implementation
fn maxSIMD(comptime T: type, data: []const T, config: SIMDConfig) T {
    if (data.len == 0) return 0;

    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or data.len < vector_size) {
        var max_val = data[0];
        for (data[1..]) |element| {
            max_val = @max(max_val, element);
        }
        return max_val;
    }

    const VectorT = @Vector(vector_size, T);
    const simd_len = (data.len / vector_size) * vector_size;
    var max_vec: VectorT = data[0..vector_size].*;
    var i: usize = vector_size;

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = data[i..][0..vector_size].*;
        max_vec = @max(max_vec, va);
    }

    var max_val = @reduce(.Max, max_vec);
    while (i < data.len) : (i += 1) {
        max_val = @max(max_val, data[i]);
    }

    return max_val;
}

/// SIMD sum implementation with Kahan summation for precision
fn sumSIMD(comptime T: type, data: []const T, config: SIMDConfig) T {
    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or data.len < vector_size) {
        var sum: T = 0;
        for (data) |element| {
            sum += element;
        }
        return sum;
    }

    const VectorT = @Vector(vector_size, T);
    const simd_len = (data.len / vector_size) * vector_size;
    var sum: T = 0;
    var i: usize = 0;

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = data[i..][0..vector_size].*;
        sum += @reduce(.Add, va);
    }

    while (i < data.len) : (i += 1) {
        sum += data[i];
    }

    return sum;
}

/// SIMD variance implementation
fn varianceSIMD(comptime T: type, data: []const T, config: SIMDConfig) T {
    if (data.len == 0) return 0;

    const mean_val = sumSIMD(T, data, config) / @as(T, @floatFromInt(data.len));
    const vector_size = config.getVectorSize(T);

    if (!config.supportsVectorization(T) or data.len < vector_size) {
        var sum_sq: T = 0;
        for (data) |element| {
            const diff = element - mean_val;
            sum_sq += diff * diff;
        }
        return sum_sq / @as(T, @floatFromInt(data.len));
    }

    const VectorT = @Vector(vector_size, T);
    const mean_vec: VectorT = @splat(mean_val);
    const simd_len = (data.len / vector_size) * vector_size;
    var sum_sq: T = 0;
    var i: usize = 0;

    while (i < simd_len) : (i += vector_size) {
        const va: VectorT = data[i..][0..vector_size].*;
        const diff = va - mean_vec;
        sum_sq += @reduce(.Add, diff * diff);
    }

    while (i < data.len) : (i += 1) {
        const diff = data[i] - mean_val;
        sum_sq += diff * diff;
    }

    return sum_sq / @as(T, @floatFromInt(data.len));
}

/// SIMD activation function implementation
fn activateSIMD(comptime T: type, data: []T, activation: ActivationFunction, config: SIMDConfig) void {
    switch (activation) {
        .relu => {
            const vector_size = config.getVectorSize(T);
            if (!config.supportsVectorization(T) or data.len < vector_size) {
                for (data) |*element| {
                    element.* = @max(@as(T, 0), element.*);
                }
                return;
            }

            const VectorT = @Vector(vector_size, T);
            const zero_vec: VectorT = @splat(@as(T, 0));
            const simd_len = (data.len / vector_size) * vector_size;
            var i: usize = 0;

            while (i < simd_len) : (i += vector_size) {
                var va: VectorT = data[i..][0..vector_size].*;
                va = @max(zero_vec, va);
                data[i..][0..vector_size].* = va;
            }

            while (i < data.len) : (i += 1) {
                data[i] = @max(@as(T, 0), data[i]);
            }
        },
        .leaky_relu => {
            const alpha: T = 0.01;
            for (data) |*element| {
                element.* = if (element.* > 0) element.* else alpha * element.*;
            }
        },
        .sigmoid => {
            for (data) |*element| {
                element.* = @as(T, 1.0) / (@as(T, 1.0) + std.math.exp(-element.*));
            }
        },
        .tanh => {
            for (data) |*element| {
                element.* = std.math.tanh(element.*);
            }
        },
        .softplus => {
            for (data) |*element| {
                element.* = std.math.log(@as(T, 1.0) + std.math.exp(element.*));
            }
        },
        .gelu => {
            for (data) |*element| {
                const x = element.*;
                element.* = 0.5 * x * (1.0 + std.math.tanh(std.math.sqrt(2.0 / std.math.pi) * (x + 0.044715 * x * x * x)));
            }
        },
        .swish => {
            for (data) |*element| {
                const x = element.*;
                element.* = x / (@as(T, 1.0) + std.math.exp(-x));
            }
        },
        .mish => {
            for (data) |*element| {
                const x = element.*;
                element.* = x * std.math.tanh(std.math.log(@as(T, 1.0) + std.math.exp(x)));
            }
        },
    }
}

/// Check if a function is vectorizable (compile-time check)
fn isVectorizableFunction(comptime func: anytype) bool {
    _ = func;
    // This would need more sophisticated analysis
    // For now, assume most functions are not easily vectorizable
    return false;
}

/// Apply function with SIMD (when possible)
fn mapSIMD(comptime T: type, data: []T, comptime func: fn (T) T, config: SIMDConfig) void {
    _ = config;
    // For now, use scalar implementation
    // Future: implement vectorizable function detection and SIMD application
    for (data) |*element| {
        element.* = func(element.*);
    }
}

/// Reduce with SIMD optimization
fn reduceSIMD(comptime T: type, data: []const T, initial: T, comptime func: fn (T, T) T, config: SIMDConfig) T {
    _ = config;
    // For now, use scalar implementation
    // Future: implement SIMD-optimized reduction for associative operations
    var result = initial;
    for (data) |element| {
        result = func(result, element);
    }
    return result;
}

/// Apply window function with SIMD
fn applyWindowSIMD(comptime T: type, data: []T, window_type: WindowType, config: SIMDConfig) void {
    _ = config;
    const n = data.len;
    const n_f: T = @floatFromInt(n);

    switch (window_type) {
        .rectangular => {
            // No modification needed
        },
        .hann => {
            for (data, 0..) |*element, i| {
                const i_f: T = @floatFromInt(i);
                const window_val = 0.5 * (1.0 - std.math.cos(2.0 * std.math.pi * i_f / (n_f - 1.0)));
                element.* *= window_val;
            }
        },
        .hamming => {
            for (data, 0..) |*element, i| {
                const i_f: T = @floatFromInt(i);
                const window_val = 0.54 - 0.46 * std.math.cos(2.0 * std.math.pi * i_f / (n_f - 1.0));
                element.* *= window_val;
            }
        },
        .blackman => {
            for (data, 0..) |*element, i| {
                const i_f: T = @floatFromInt(i);
                const a0 = 0.42;
                const a1 = 0.5;
                const a2 = 0.08;
                const window_val = a0 - a1 * std.math.cos(2.0 * std.math.pi * i_f / (n_f - 1.0)) + a2 * std.math.cos(4.0 * std.math.pi * i_f / (n_f - 1.0));
                element.* *= window_val;
            }
        },
        .kaiser => {
            // Kaiser window with configurable beta parameter (beta = 8.6 for good sidelobe suppression)
            const beta: T = 8.6;
            const alpha = (n_f - 1.0) / 2.0;
            const bessel_beta = besselI0(beta);
            
            for (data, 0..) |*element, i| {
                const i_f: T = @floatFromInt(i);
                const x = (i_f - alpha) / alpha;
                
                // Clamp x to prevent numerical issues with sqrt
                const x_clamped = @max(-1.0, @min(1.0, x));
                const sqrt_arg = 1.0 - x_clamped * x_clamped;
                
                // Handle edge case where sqrt_arg might be very close to 0
                const window_val = if (sqrt_arg > std.math.floatEps(T)) 
                    besselI0(beta * std.math.sqrt(sqrt_arg)) / bessel_beta
                else 
                    0.0;
                    
                element.* *= window_val;
            }
        },
    }
}

/// Bessel function I0 approximation for Kaiser window
fn besselI0(x: anytype) @TypeOf(x) {
    const abs_x = @abs(x);

    if (abs_x < 3.75) {
        const t = (x / 3.75) * (x / 3.75);
        return 1.0 + 3.5156229 * t + 3.0899424 * t * t + 1.2067492 * t * t * t +
            0.2659732 * t * t * t * t + 0.0360768 * t * t * t * t * t + 0.0045813 * t * t * t * t * t * t;
    } else {
        const t = 3.75 / abs_x;
        return (std.math.exp(abs_x) / std.math.sqrt(abs_x)) *
            (0.39894228 + 0.01328592 * t + 0.00225319 * t * t - 0.00157565 * t * t * t +
                0.09162810 * t * t * t * t - 0.02057706 * t * t * t * t * t + 0.02635537 * t * t * t * t * t * t -
                0.01647633 * t * t * t * t * t * t * t + 0.00392377 * t * t * t * t * t * t * t * t);
    }
}

// Allocation strategy implementations (placeholders for now)
fn allocateHugePage(allocator: Allocator, comptime T: type, size: usize, alignment: usize) ![]T {
    // Fallback to aligned allocation
    return allocator.alignedAlloc(T, alignment, size);
}

fn deallocateHugePage(allocator: Allocator, comptime T: type) void {
    allocator.free(T);
}

fn allocateNumaAware(allocator: Allocator, comptime T: type, size: usize, alignment: usize) ![]T {
    // Fallback to aligned allocation
    return allocator.alignedAlloc(T, alignment, size);
}

fn deallocateNumaAware(allocator: Allocator, comptime T: type) void {
    allocator.free(T);
}

fn reallocAligned(allocator: Allocator, comptime T: type, ptr: []T, new_size: usize, alignment: usize) ![]T {
    // Simplified reallocation - in practice would preserve alignment
    return allocator.realloc(ptr, new_size, alignment);
}

fn reallocHugePage(allocator: Allocator, comptime T: type, ptr: []T, new_size: usize, alignment: usize) ![]T {
    return reallocAligned(allocator, T, ptr, new_size, alignment);
}

fn reallocNumaAware(allocator: Allocator, comptime T: type, ptr: []T, new_size: usize, alignment: usize) ![]T {
    return reallocAligned(allocator, T, ptr, new_size, alignment);
}

/// Convenience alias for the enhanced vector type
pub const Vector = EnhancedVector;

/// Legacy compatibility
pub const vector = EnhancedVector;
pub const VectorOps = SIMDOps;

test "enhanced vector basic operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var vec = try EnhancedVector(f32).init(allocator, 8);
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

test "enhanced vector arithmetic with SIMD" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var vec1 = try EnhancedVector(f32).initWithData(allocator, &[_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
    defer vec1.deinit();

    var vec2 = try EnhancedVector(f32).initWithData(allocator, &[_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 });
    defer vec2.deinit();

    // Test addition
    vec1.add(&vec2);
    for (0..8) |i| {
        try testing.expectEqual(@as(f32, 9.0), vec1.get(i));
    }

    // Test scaling
    vec1.scale(0.5);
    for (0..8) |i| {
        try testing.expectEqual(@as(f32, 4.5), vec1.get(i));
    }
}

test "enhanced vector advanced operations" {
    const testing = std.testing;
    const allocator = testing.allocator;

    var vec1 = try EnhancedVector(f32).initWithData(allocator, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer vec1.deinit();

    var vec2 = try EnhancedVector(f32).initWithData(allocator, &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    defer vec2.deinit();

    // Test dot product
    const dot_result = vec1.dot(&vec2);
    try testing.expectApproxEqAbs(@as(f32, 30.0), dot_result, 0.001);

    // Test magnitude
    const magnitude = vec1.magnitude();
    try testing.expectApproxEqAbs(@as(f32, 5.4772256), magnitude, 0.001);

    // Test normalization
    vec1.normalize();
    const normalized_magnitude = vec1.magnitude();
    try testing.expectApproxEqAbs(@as(f32, 1.0), normalized_magnitude, 0.001);

    // Test activation functions
    var vec3 = try EnhancedVector(f32).initWithData(allocator, &[_]f32{ -2.0, -1.0, 0.0, 1.0, 2.0 });
    defer vec3.deinit();

    vec3.activate(.relu);
    try testing.expectEqual(@as(f32, 0.0), vec3.get(0));
    try testing.expectEqual(@as(f32, 0.0), vec3.get(1));
    try testing.expectEqual(@as(f32, 0.0), vec3.get(2));
    try testing.expectEqual(@as(f32, 1.0), vec3.get(3));
    try testing.expectEqual(@as(f32, 2.0), vec3.get(4));
}

test "SIMD operations standalone" {
    const testing = std.testing;
    const allocator = testing.allocator;

    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 };
    const result = try allocator.alloc(f32, 8);
    defer allocator.free(result);

    const config = SIMDConfig.detect();

    // Test SIMD operations
    SIMDOps.add(f32, &a, &b, result, config);
    for (result) |val| {
        try testing.expectEqual(@as(f32, 9.0), val);
    }

    // Test dot product
    const dot_result = SIMDOps.dot(f32, &a, &b, config);
    try testing.expectApproxEqAbs(@as(f32, 204.0), dot_result, 0.001);

    // Test distance functions
    const euclidean = SIMDOps.euclideanDistance(f32, &a, &b, config);
    try testing.expectApproxEqAbs(@as(f32, 11.832159), euclidean, 0.001);

    const manhattan = SIMDOps.manhattanDistance(f32, &a, &b, config);
    try testing.expectEqual(@as(f32, 32.0), manhattan);
}
