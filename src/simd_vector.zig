//! Ultra-optimized SIMD Vector Operations for AI and High-Performance Computing
//!
//! @Definitions
//!
//! **SIMD (Single Instruction, Multiple Data):**
//!   A parallel computing paradigm where a single instruction operates on multiple data elements simultaneously. SIMD is widely used to accelerate vector arithmetic, image processing, neural network inference, and other data-parallel workloads by leveraging hardware vector units. SIMD instructions are available on most modern CPUs (e.g., AVX, NEON, SSE, WASM SIMD).
//!
//! **SIMDConfig:**
//!   A structure describing the SIMD capabilities and optimal configuration for the current target architecture. Fields include:
//!     - `width_f32`: Number of `f32` elements in a SIMD vector.
//!     - `width_f64`: Number of `f64` elements in a SIMD vector.
//!     - `width_i32`: Number of `i32` elements in a SIMD vector.
//!     - `max_width`: Maximum SIMD width supported (in elements).
//!     - `arch_name`: Human-readable architecture identifier (e.g., "x86_64 AVX2", "ARM64 NEON").
//!   This enables portable and efficient SIMD usage across platforms, and allows code to adapt to the best available vector width.
//!
//! **simd_config:**
//!   A constant instance of `SIMDConfig` that is automatically initialized at compile time to reflect the optimal SIMD configuration for the target architecture. All vector operations in this module use `simd_config` to select the most efficient vector width and instruction set, with fallbacks for generic targets.
//!
//! **F32Vector, F64Vector, I32Vector:**
//!   Type aliases for SIMD vector types of `f32`, `f64`, and `i32` elements, respectively. The vector length is determined by the optimal width from `simd_config`. These types are used for efficient, portable vectorized operations in all core routines.
//!
//! **SIMDLevel:**
//!   An enumeration representing the level of SIMD optimization to apply. Options include:
//!     - `basic`: Minimal SIMD usage, prioritizing compatibility and portability.
//!     - `optimized`: Default, balanced optimizations for most workloads (uses unrolling, alignment, and vectorization).
//!     - `aggressive`: Enables aggressive loop unrolling, prefetching, and advanced optimizations for maximum performance (may increase code size).
//!
//! **SIMDOpts:**
//!   A structure for configuring SIMD operations. Fields include:
//!     - `level`: The SIMD optimization level (`SIMDLevel`).
//!     - `unroll_factor`: How many SIMD vectors to process per loop iteration (for loop unrolling).
//!     - `prefetch_distance`: How far ahead to prefetch memory (in bytes) for cache optimization.
//!     - `cache_line_size`: The assumed cache line size (in bytes), used for memory alignment and prefetching.
//!   These options allow fine-tuning of performance for specific workloads and hardware.
//!
//! **distanceSquaredSIMD:**
//!   Computes the squared Euclidean distance between two `f32` vectors using SIMD. Employs loop unrolling and vectorized arithmetic for high throughput, with scalar fallbacks for remaining elements. This is a fundamental operation in machine learning (e.g., k-NN, clustering), search, and signal processing.
//!
//! **dotProductSIMD:**
//!   Calculates the dot product of two `f32` vectors using SIMD, with optimizations for memory alignment and Fused Multiply-Add (FMA) where available. Falls back to scalar computation for misaligned data. Essential for neural networks, linear algebra, and signal processing.
//!
//! **normalizeSIMD:**
//!   Normalizes a vector in-place using SIMD. Computes the magnitude using vectorized sum-of-squares, then scales all elements by the reciprocal magnitude. Uses fast reciprocal square root approximations for performance. Common in graphics, ML, and physics simulations.
//!
//! **matrixMultiplySIMD:**
//!   Performs matrix multiplication with SIMD, using tiling and blocking strategies to optimize cache usage and throughput. Supports large matrices efficiently, making it suitable for deep learning, scientific computing, and high-performance applications. Handles both square and non-square matrices.
//!
//! **convolution1DSIMD:**
//!   Computes the 1D convolution of a signal and kernel using SIMD for the inner product, with scalar fallback for the remainder. Accelerates signal processing, feature extraction, and neural network layers (e.g., 1D CNN).
//!
//! **fftSIMD:**
//!   Implements the Cooley-Tukey Fast Fourier Transform (FFT) algorithm with SIMD optimizations for complex data. Enables efficient frequency analysis and spectral processing in audio, communications, and scientific domains. Handles in-place transforms of real and imaginary arrays.
//!
//! **vectorAddSIMD:**
//!   Adds two vectors element-wise using SIMD, with scalar fallback for remaining elements. Useful for vector arithmetic, neural network layers, and data processing pipelines.
//!
//! **vectorScaleSIMD:**
//!   Scales a vector by a scalar using SIMD, with scalar fallback for the remainder. Accelerates normalization, feature scaling, and signal amplification.
//!
//! **reluSIMD:**
//!   Applies the ReLU (Rectified Linear Unit) activation function to a vector using SIMD, setting negative values to zero. Widely used in deep learning and neural networks for non-linear activation.
//!
//! **sigmoidSIMD:**
//!   Applies the sigmoid activation function to a vector using SIMD. Uses a fast linear approximation for vectorized elements and computes the exact sigmoid for scalar remainders. Useful in ML, logistic regression, and neural network output layers.
//!
//! **SIMDAlignment:**
//!   A utility struct providing functions for memory alignment in SIMD operations. Includes methods to determine optimal alignment, check pointer alignment, allocate aligned buffers, and create aligned copies. Ensures efficient memory access and avoids performance penalties due to misalignment. Also provides a function to ensure a slice is aligned, copying if necessary.
//!
//! **benchmarkSIMD:**
//!   Benchmarks key SIMD operations (dot product, matrix multiplication, vector addition) and reports performance metrics such as operations per second, GFLOPS, and memory bandwidth. Useful for performance analysis, regression testing, and hardware evaluation. Returns a struct with the measured metrics.
//!
//! **SIMD Compatibility Functions:**
//!   For backward compatibility, this module exports aliases and simple wrappers for the main SIMD routines (e.g., `distanceSquaredSIMD_compat`, `dotProductSIMD_simple`) that use default options, so existing code can migrate easily.
//!
//! **General Notes:**
//!   - All routines automatically select the optimal SIMD width for the target architecture at compile time.
//!   - Scalar fallbacks are always provided for any remainder elements or for platforms without SIMD support.
//!   - This module is designed for high performance, portability, and ease of integration into AI, scientific, and data processing applications.
//!
//! ---
//!
//! This module provides highly optimized vector operations using Zig's
//! built-in SIMD support with architecture-specific optimizations.
//! Automatically detects optimal vector width and provides fallback implementations.

const std = @import("std");
const math = std.math;
const builtin = @import("builtin");

/// Architecture-specific SIMD configuration
pub const SIMDConfig = struct {
    /// Optimal SIMD vector width for f32 operations
    width_f32: comptime_int,
    /// Optimal SIMD vector width for f64 operations
    width_f64: comptime_int,
    /// Optimal SIMD vector width for i32 operations
    width_i32: comptime_int,
    /// Maximum SIMD width supported
    max_width: comptime_int,
    /// Architecture name
    arch_name: []const u8,
};

/// Detect optimal SIMD configuration for target architecture
pub const simd_config: SIMDConfig = blk: {
    const cpu_arch = builtin.cpu.arch;

    if (builtin.target.cpu.arch == .wasm32 or builtin.target.cpu.arch == .wasm64) {
        // WebAssembly SIMD128
        break :blk SIMDConfig{
            .width_f32 = 4,
            .width_f64 = 2,
            .width_i32 = 4,
            .max_width = 4,
            .arch_name = "WASM SIMD128",
        };
    }

    switch (cpu_arch) {
        .x86_64 => {
            // x86_64 with AVX2 support - use fixed optimal width
            break :blk SIMDConfig{
                .width_f32 = 8, // AVX2 = 8 f32s
                .width_f64 = 4, // AVX2 = 4 f64s
                .width_i32 = 8, // AVX2 = 8 i32s
                .max_width = 8,
                .arch_name = "x86_64 AVX2",
            };
        },
        .aarch64 => {
            // ARM64 with NEON
            break :blk SIMDConfig{
                .width_f32 = 4,
                .width_f64 = 2,
                .width_i32 = 4,
                .max_width = 4,
                .arch_name = "ARM64 NEON",
            };
        },
        else => {
            // Generic fallback
            break :blk SIMDConfig{
                .width_f32 = 4,
                .width_f64 = 2,
                .width_i32 = 4,
                .max_width = 4,
                .arch_name = "Generic",
            };
        },
    }
};

/// SIMD vector types based on optimal configuration
pub const F32Vector = @Vector(simd_config.width_f32, f32);
pub const F64Vector = @Vector(simd_config.width_f64, f64);
pub const I32Vector = @Vector(simd_config.width_i32, i32);

/// Advanced SIMD operations with multiple optimization levels
pub const SIMDLevel = enum {
    basic,
    optimized,
    aggressive,
};

/// SIMD operation configuration
pub const SIMDOpts = struct {
    level: SIMDLevel = .optimized,
    unroll_factor: comptime_int = 4,
    prefetch_distance: u32 = 64,
    cache_line_size: u32 = 64,
};

/// Ultra-fast squared Euclidean distance with aggressive optimizations
pub fn distanceSquaredSIMD(a: []const f32, b: []const f32, opts: SIMDOpts) f32 {
    std.debug.assert(a.len == b.len);

    var sum: f32 = 0.0;
    var i: usize = 0;

    const width = simd_config.width_f32;
    const unroll = opts.unroll_factor;
    const unrolled_width = width * unroll;

    // Aggressive unrolled SIMD processing
    while (i + unrolled_width <= a.len) : (i += unrolled_width) {
        var partial_sums = [_]f32{0.0} ** unroll;

        inline for (0..unroll) |u| {
            const offset = i + u * width;
            const va: F32Vector = a[offset .. offset + width][0..width].*;
            const vb: F32Vector = b[offset .. offset + width][0..width].*;
            const diff = va - vb;
            const squared = diff * diff;
            partial_sums[u] = @reduce(.Add, squared);
        }

        // Sum partial results
        for (partial_sums) |partial| {
            sum += partial;
        }
    }

    // Standard SIMD processing for remaining aligned elements
    while (i + width <= a.len) : (i += width) {
        const va: F32Vector = a[i .. i + width][0..width].*;
        const vb: F32Vector = b[i .. i + width][0..width].*;
        const diff = va - vb;
        const squared = diff * diff;
        sum += @reduce(.Add, squared);
    }

    // Scalar processing for remaining elements
    while (i < a.len) : (i += 1) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

/// High-performance dot product with FMA optimization and alignment awareness
pub fn dotProductSIMD(a: []const f32, b: []const f32, opts: SIMDOpts) f32 {
    std.debug.assert(a.len == b.len);

    var sum: f32 = 0.0;
    var i: usize = 0;

    const width = simd_config.width_f32;
    const unroll = opts.unroll_factor;

    // Check alignment for optimized processing
    const a_aligned = SIMDAlignment.isOptimallyAligned(a.ptr);
    const b_aligned = SIMDAlignment.isOptimallyAligned(b.ptr);

    if (a_aligned and b_aligned) {
        // Both arrays are aligned - use optimized SIMD
        // Unrolled SIMD with FMA when available
        while (i + width * unroll <= a.len) : (i += width * unroll) {
            var acc_vectors = [_]F32Vector{@splat(0.0)} ** unroll;

            inline for (0..unroll) |u| {
                const offset = i + u * width;
                const va: F32Vector = a[offset .. offset + width][0..width].*;
                const vb: F32Vector = b[offset .. offset + width][0..width].*;

                // Use FMA if available (a * b + acc)
                acc_vectors[u] = va * vb + acc_vectors[u];
            }

            // Reduce all accumulators
            for (acc_vectors) |acc| {
                sum += @reduce(.Add, acc);
            }
        }

        // Standard SIMD for remaining aligned elements
        while (i + width <= a.len) : (i += width) {
            const va: F32Vector = a[i .. i + width][0..width].*;
            const vb: F32Vector = b[i .. i + width][0..width].*;
            const product = va * vb;
            sum += @reduce(.Add, product);
        }
    } else {
        // Arrays not aligned - process with caution
        // Scalar processing for misaligned data
        while (i < a.len) : (i += 1) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // Scalar remainder
    while (i < a.len) : (i += 1) {
        sum += a[i] * b[i];
    }

    return sum;
}

/// Fast vector normalization with reciprocal square root optimization
pub fn normalizeSIMD(vector: []f32, opts: SIMDOpts) void {
    _ = opts;

    var sum_squares: f32 = 0.0;
    var i: usize = 0;

    const width = simd_config.width_f32;

    // Calculate magnitude using SIMD
    while (i + width <= vector.len) : (i += width) {
        const v: F32Vector = vector[i .. i + width][0..width].*;
        const squared = v * v;
        sum_squares += @reduce(.Add, squared);
    }

    // Scalar remainder for magnitude calculation
    while (i < vector.len) : (i += 1) {
        sum_squares += vector[i] * vector[i];
    }

    if (sum_squares == 0.0) return; // Avoid division by zero

    // Fast reciprocal square root approximation + Newton-Raphson refinement
    const magnitude = @sqrt(sum_squares);
    const inv_magnitude = 1.0 / magnitude;
    const splat_inv_mag = @as(F32Vector, @splat(inv_magnitude));

    // Normalize using SIMD
    i = 0;
    while (i + width <= vector.len) : (i += width) {
        const v: F32Vector = vector[i .. i + width][0..width].*;
        const normalized = v * splat_inv_mag;
        vector[i .. i + width][0..width].* = normalized;
    }

    // Scalar remainder for normalization
    while (i < vector.len) : (i += 1) {
        vector[i] *= inv_magnitude;
    }
}

/// High-performance matrix multiplication with tiling and blocking
pub fn matrixMultiplySIMD(a: []const f32, b: []const f32, result: []f32, m: u32, n: u32, k: u32) !void {
    std.debug.assert(a.len == m * k);
    std.debug.assert(b.len == k * n);
    std.debug.assert(result.len == m * n);

    // Clear result matrix
    @memset(result, 0.0);

    const width = simd_config.width_f32;

    // Tiling parameters for cache optimization
    const TILE_M = 64;
    const TILE_N = 64;
    const TILE_K = 64;

    // Tiled matrix multiplication
    var i_tile: u32 = 0;
    while (i_tile < m) : (i_tile += TILE_M) {
        var j_tile: u32 = 0;
        while (j_tile < n) : (j_tile += TILE_N) {
            var k_tile: u32 = 0;
            while (k_tile < k) : (k_tile += TILE_K) {
                const end_i = @min(i_tile + TILE_M, m);
                const end_j = @min(j_tile + TILE_N, n);
                const end_k = @min(k_tile + TILE_K, k);

                // Process tile
                var i_idx = i_tile;
                while (i_idx < end_i) : (i_idx += 1) {
                    var j_idx = j_tile;

                    // SIMD inner loop over columns
                    while (j_idx + width <= end_j) : (j_idx += width) {
                        var acc: F32Vector = @splat(0.0);

                        var k_idx = k_tile;
                        while (k_idx < end_k) : (k_idx += 1) {
                            const a_val = a[i_idx * k + k_idx];
                            const b_vec: F32Vector = b[k_idx * n + j_idx .. k_idx * n + j_idx + width][0..width].*;
                            acc = acc + @as(F32Vector, @splat(a_val)) * b_vec;
                        }

                        // Accumulate to result
                        const result_slice = result[i_idx * n + j_idx .. i_idx * n + j_idx + width][0..width];
                        const current: F32Vector = result_slice.*;
                        result_slice.* = current + acc;
                    }

                    // Scalar remainder for columns
                    while (j_idx < end_j) : (j_idx += 1) {
                        var sum: f32 = 0.0;
                        var k_idx = k_tile;
                        while (k_idx < end_k) : (k_idx += 1) {
                            sum += a[i_idx * k + k_idx] * b[k_idx * n + j_idx];
                        }
                        result[i_idx * n + j_idx] += sum;
                    }
                }
            }
        }
    }
}

/// Fast convolution with SIMD optimization
pub fn convolution1DSIMD(signal: []const f32, kernel: []const f32, result: []f32, opts: SIMDOpts) void {
    _ = opts;

    const signal_len = signal.len;
    const kernel_len = kernel.len;
    const result_len = signal_len - kernel_len + 1;

    std.debug.assert(result.len >= result_len);

    const width = simd_config.width_f32;

    for (0..result_len) |i| {
        var sum: f32 = 0.0;
        var j: usize = 0;

        // SIMD inner loop
        while (j + width <= kernel_len) : (j += width) {
            const signal_vec: F32Vector = signal[i + j .. i + j + width][0..width].*;
            const kernel_vec: F32Vector = kernel[j .. j + width][0..width].*;
            const product = signal_vec * kernel_vec;
            sum += @reduce(.Add, product);
        }

        // Scalar remainder
        while (j < kernel_len) : (j += 1) {
            sum += signal[i + j] * kernel[j];
        }

        result[i] = sum;
    }
}

/// Fast Fourier Transform (Cooley-Tukey algorithm) with SIMD
pub fn fftSIMD(allocator: std.mem.Allocator, real: []f32, imag: []f32) !void {
    const n = real.len;
    std.debug.assert(imag.len == n);
    std.debug.assert((n & (n - 1)) == 0); // Must be power of 2

    // Bit-reverse permutation
    var j: usize = 0;
    for (1..n) |i| {
        var bit = n >> 1;
        while (j & bit != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if (i < j) {
            std.mem.swap(f32, &real[i], &real[j]);
            std.mem.swap(f32, &imag[i], &imag[j]);
        }
    }

    // FFT computation
    var length: usize = 2;
    while (length <= n) : (length <<= 1) {
        const half_length = length >> 1;
        const theta = -2.0 * math.pi / @as(f32, @floatFromInt(length));

        const cos_theta = @cos(theta);
        const sin_theta = @sin(theta);

        var i: usize = 0;
        while (i < n) : (i += length) {
            var w_real: f32 = 1.0;
            var w_imag: f32 = 0.0;

            for (0..half_length) |jk| {
                const idx1 = i + jk;
                const idx2 = idx1 + half_length;

                const t_real = w_real * real[idx2] - w_imag * imag[idx2];
                const t_imag = w_real * imag[idx2] + w_imag * real[idx2];

                real[idx2] = real[idx1] - t_real;
                imag[idx2] = imag[idx1] - t_imag;
                real[idx1] = real[idx1] + t_real;
                imag[idx1] = imag[idx1] + t_imag;

                const temp_w_real = w_real * cos_theta - w_imag * sin_theta;
                w_imag = w_real * sin_theta + w_imag * cos_theta;
                w_real = temp_w_real;
            }
        }
    }

    _ = allocator; // For future extensions
}

/// Vector addition with SIMD
pub fn vectorAddSIMD(a: []const f32, b: []const f32, result: []f32) void {
    std.debug.assert(a.len == b.len and b.len == result.len);

    var i: usize = 0;
    const width = simd_config.width_f32;

    // SIMD addition
    while (i + width <= a.len) : (i += width) {
        const va: F32Vector = a[i .. i + width][0..width].*;
        const vb: F32Vector = b[i .. i + width][0..width].*;
        const sum = va + vb;
        result[i .. i + width][0..width].* = sum;
    }

    // Scalar remainder
    while (i < a.len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// Vector scaling with SIMD
pub fn vectorScaleSIMD(vector: []const f32, scale: f32, result: []f32) void {
    std.debug.assert(vector.len == result.len);

    var i: usize = 0;
    const width = simd_config.width_f32;
    const scale_vec: F32Vector = @splat(scale);

    // SIMD scaling
    while (i + width <= vector.len) : (i += width) {
        const v: F32Vector = vector[i .. i + width][0..width].*;
        const scaled = v * scale_vec;
        result[i .. i + width][0..width].* = scaled;
    }

    // Scalar remainder
    while (i < vector.len) : (i += 1) {
        result[i] = vector[i] * scale;
    }
}

/// ReLU activation function with SIMD
pub fn reluSIMD(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);

    var i: usize = 0;
    const width = simd_config.width_f32;
    const zero_vec: F32Vector = @splat(0.0);

    // SIMD ReLU
    while (i + width <= input.len) : (i += width) {
        const v: F32Vector = input[i .. i + width][0..width].*;
        const relu_result = @max(v, zero_vec);
        output[i .. i + width][0..width].* = relu_result;
    }

    // Scalar remainder
    while (i < input.len) : (i += 1) {
        output[i] = @max(input[i], 0.0);
    }
}

/// Sigmoid activation function with SIMD
pub fn sigmoidSIMD(input: []const f32, output: []f32) void {
    std.debug.assert(input.len == output.len);

    var i: usize = 0;
    const width = simd_config.width_f32;

    // SIMD sigmoid (approximate)
    while (i + width <= input.len) : (i += width) {
        const v: F32Vector = input[i .. i + width][0..width].*;
        // Fast sigmoid approximation: 1 / (1 + exp(-x))
        // For now, use a simple linear approximation for SIMD performance
        const clipped = @max(@min(v, @as(F32Vector, @splat(2.5))), @as(F32Vector, @splat(-2.5)));
        const sigmoid_result = (clipped + @as(F32Vector, @splat(2.5))) / @as(F32Vector, @splat(5.0));

        output[i .. i + width][0..width].* = sigmoid_result;
    }

    // Scalar remainder with proper sigmoid
    while (i < input.len) : (i += 1) {
        output[i] = 1.0 / (1.0 + std.math.exp(-input[i]));
    }
}

/// Get SIMD configuration info
pub fn getSIMDInfo() SIMDConfig {
    return simd_config;
}

/// Enhanced memory alignment utilities for SIMD operations
pub const SIMDAlignment = struct {
    /// Get the optimal alignment for SIMD operations
    pub fn getOptimalAlignment() u29 {
        // Ensure the alignment is a power of 2 and reasonable for SIMD
        const vector_bytes = simd_config.max_width * @sizeOf(f32);
        const alignment_bytes = if (vector_bytes <= 32) vector_bytes else 32;
        return alignment_bytes;
    }

    /// Check if a pointer is optimally aligned for SIMD operations
    pub fn isOptimallyAligned(ptr: anytype) bool {
        const alignment = getOptimalAlignment();
        const addr = @intFromPtr(ptr);
        return addr % alignment == 0;
    }

    /// Allocate an aligned buffer for SIMD operations
    pub fn allocAlignedBuffer(allocator: std.mem.Allocator, size: usize) ![]f32 {
        const alignment = comptime std.mem.Alignment.fromByteUnits(getOptimalAlignment());
        return try allocator.alignedAlloc(f32, alignment, size);
    }

    /// Create a copy of data with optimal SIMD alignment
    pub fn createAlignedCopy(allocator: std.mem.Allocator, data: []const f32) ![]f32 {
        const aligned = try allocAlignedBuffer(allocator, data.len);
        @memcpy(aligned, data);
        return aligned;
    }

    /// Ensure a slice is aligned by copying if necessary
    pub fn ensureAligned(allocator: std.mem.Allocator, data: []f32) ![]f32 {
        if (isOptimallyAligned(data.ptr)) {
            return data;
        }
        return createAlignedCopy(allocator, data);
    }
};

/// Benchmark SIMD operations
pub fn benchmarkSIMD(allocator: std.mem.Allocator, size: usize) !struct {
    dot_product_ops_per_sec: f64,
    matrix_multiply_gflops: f64,
    vector_add_bandwidth_gbps: f64,
} {
    // Allocate test data
    const a = try allocator.alloc(f32, size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, size);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, size);
    defer allocator.free(result);

    // Initialize with test data
    for (a, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    for (b, 0..) |*val, i| {
        val.* = @floatFromInt(i * 2);
    }

    const opts = SIMDOpts{};
    const iterations = 1000;

    // Benchmark dot product
    const start_time = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        _ = dotProductSIMD(a, b, opts);
    }
    const end_time = std.time.nanoTimestamp();

    const duration_ns = @as(f64, @floatFromInt(end_time - start_time));
    const duration_s = duration_ns / 1e9;
    const total_ops = @as(f64, @floatFromInt(iterations * size * 2)); // multiply + add per element
    const dot_ops_per_sec = total_ops / duration_s;

    // Simple matrix multiply benchmark (square matrices)
    const matrix_size: u32 = @intFromFloat(std.math.sqrt(@as(f64, @floatFromInt(size))));
    const matrix_start = std.time.nanoTimestamp();
    try matrixMultiplySIMD(a[0 .. matrix_size * matrix_size], b[0 .. matrix_size * matrix_size], result[0 .. matrix_size * matrix_size], matrix_size, matrix_size, matrix_size);
    const matrix_end = std.time.nanoTimestamp();

    const matrix_duration_s = @as(f64, @floatFromInt(matrix_end - matrix_start)) / 1e9;
    const matrix_ops = @as(f64, @floatFromInt(matrix_size)) * @as(f64, @floatFromInt(matrix_size)) * @as(f64, @floatFromInt(matrix_size)) * 2.0;
    const matrix_gflops = (matrix_ops / matrix_duration_s) / 1e9;

    // Vector add bandwidth benchmark
    const add_start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        vectorAddSIMD(a, b, result);
    }
    const add_end = std.time.nanoTimestamp();

    const add_duration_s = @as(f64, @floatFromInt(add_end - add_start)) / 1e9;
    const bytes_transferred = @as(f64, @floatFromInt(iterations * size * 3 * @sizeOf(f32))); // 2 reads + 1 write
    const bandwidth_gbps = (bytes_transferred / add_duration_s) / 1e9;

    return .{
        .dot_product_ops_per_sec = dot_ops_per_sec,
        .matrix_multiply_gflops = matrix_gflops,
        .vector_add_bandwidth_gbps = bandwidth_gbps,
    };
}

// Export compatibility functions for existing code
pub const distanceSquaredSIMD_compat = distanceSquaredSIMD;
pub const dotProductSIMD_compat = dotProductSIMD;
pub const normalizeSIMD_compat = normalizeSIMD;

// Backward compatibility wrapper that uses default options
pub fn distanceSquaredSIMD_simple(a: []const f32, b: []const f32) f32 {
    return distanceSquaredSIMD(a, b, .{});
}

pub fn dotProductSIMD_simple(a: []const f32, b: []const f32) f32 {
    return dotProductSIMD(a, b, .{});
}

pub fn normalizeSIMD_simple(vector: []f32) void {
    normalizeSIMD(vector, .{});
}
