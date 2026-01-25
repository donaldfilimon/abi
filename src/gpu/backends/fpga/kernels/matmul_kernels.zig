//! FPGA-optimized matrix multiplication kernels for LLM inference
//!
//! Provides hardware-accelerated implementations for:
//! - Quantized matrix multiplication (Q4/Q8 weights)
//! - Batched MatMul operations
//! - Tiled MatMul for memory efficiency
//!
//! Performance targets (per FPGA research roadmap):
//! - Q4 MatMul: 10-20x speedup over CPU
//! - Memory bandwidth: >90% DDR/HBM utilization
//! - Latency: <10us for typical LLM weight matrices

const std = @import("std");
const build_options = @import("build_options");
const distance_kernels = @import("distance_kernels.zig");

/// Configuration for FPGA MatMul kernels
pub const MatMulKernelConfig = struct {
    /// M dimension (rows of A, rows of output)
    m: usize,
    /// N dimension (cols of B, cols of output)
    n: usize,
    /// K dimension (cols of A, rows of B)
    k: usize,
    /// Weight precision
    weight_precision: WeightPrecision = .q8,
    /// Activation precision
    activation_precision: ActivationPrecision = .fp16,
    /// Tile size for M dimension
    tile_m: u32 = 64,
    /// Tile size for N dimension
    tile_n: u32 = 64,
    /// Tile size for K dimension
    tile_k: u32 = 64,
    /// Number of parallel compute units
    compute_units: u32 = 16,
    /// Enable accumulator fusion
    fused_accumulator: bool = true,
    /// Stream weights from DDR (vs preload to BRAM)
    streaming_weights: bool = false,
};

/// Weight quantization formats supported by FPGA
pub const WeightPrecision = enum {
    fp32,
    fp16,
    bf16,
    q8, // INT8 symmetric
    q4, // INT4 symmetric
    q4_k, // K-quant 4-bit (llama.cpp compatible)
    q8_k, // K-quant 8-bit

    pub fn bits(self: WeightPrecision) u8 {
        return switch (self) {
            .fp32 => 32,
            .fp16, .bf16 => 16,
            .q8, .q8_k => 8,
            .q4, .q4_k => 4,
        };
    }

    /// Compute memory savings vs FP32
    pub fn compressionRatio(self: WeightPrecision) f32 {
        return 32.0 / @as(f32, @floatFromInt(self.bits()));
    }
};

/// Activation precision for compute
pub const ActivationPrecision = enum {
    fp32,
    fp16,
    bf16,

    pub fn bits(self: ActivationPrecision) u8 {
        return switch (self) {
            .fp32 => 32,
            .fp16, .bf16 => 16,
        };
    }
};

/// Quantization scale and zero-point for INT quantization
pub const QuantizationParams = struct {
    scale: f32,
    zero_point: i32 = 0,
    /// For block quantization (per-block scales)
    block_size: u32 = 32,
    /// Per-block scales (optional)
    block_scales: ?[]const f32 = null,
};

/// FPGA-accelerated quantized matrix multiplication kernel
pub const QuantizedMatMulKernel = struct {
    config: MatMulKernelConfig,
    allocator: std.mem.Allocator,

    // Pre-allocated buffers for tiled computation
    tile_a_buffer: ?[]f32 = null,
    tile_b_buffer: ?[]f32 = null,
    tile_c_buffer: ?[]f32 = null,

    // Quantization parameters
    weight_params: QuantizationParams = .{ .scale = 1.0 },

    pub fn init(allocator: std.mem.Allocator, config: MatMulKernelConfig) !QuantizedMatMulKernel {
        var kernel = QuantizedMatMulKernel{
            .config = config,
            .allocator = allocator,
        };

        // Pre-allocate tile buffers for streaming
        if (config.streaming_weights) {
            const tile_size = config.tile_m * config.tile_k;
            kernel.tile_a_buffer = try allocator.alloc(f32, tile_size);
            kernel.tile_b_buffer = try allocator.alloc(f32, config.tile_k * config.tile_n);
            kernel.tile_c_buffer = try allocator.alloc(f32, config.tile_m * config.tile_n);
        }

        return kernel;
    }

    pub fn deinit(self: *QuantizedMatMulKernel) void {
        if (self.tile_a_buffer) |buf| self.allocator.free(buf);
        if (self.tile_b_buffer) |buf| self.allocator.free(buf);
        if (self.tile_c_buffer) |buf| self.allocator.free(buf);
    }

    /// Set quantization parameters for weights
    pub fn setWeightQuantization(self: *QuantizedMatMulKernel, params: QuantizationParams) void {
        self.weight_params = params;
    }

    /// Execute quantized matrix multiplication: C = A @ B
    /// A: [M, K] activations (fp16/fp32)
    /// B: [K, N] quantized weights
    /// C: [M, N] output (fp16/fp32)
    pub fn execute(
        self: *QuantizedMatMulKernel,
        activations: []const f32,
        quantized_weights: []const u8,
        output: []f32,
    ) !void {
        const m = self.config.m;
        const n = self.config.n;
        const k = self.config.k;

        std.debug.assert(activations.len == m * k);
        std.debug.assert(output.len == m * n);

        // FPGA implementation would:
        // 1. Stream activations to FPGA DDR/HBM
        // 2. Load quantized weights (already on device or streaming)
        // 3. Execute tiled MatMul on compute units
        // 4. Accumulate results with fused operations
        // 5. Stream output back to host

        // CPU fallback implementation
        try self.executeCpuFallback(activations, quantized_weights, output);
    }

    /// Execute with pre-transposed weights (optimized layout)
    pub fn executeTransposed(
        self: *QuantizedMatMulKernel,
        activations: []const f32,
        quantized_weights_t: []const u8,
        output: []f32,
    ) !void {
        // Transposed layout: B is [N, K] instead of [K, N]
        // More cache-friendly for column-major access
        _ = self;
        _ = quantized_weights_t;
        _ = activations;
        _ = output;

        // FPGA would handle transposed layout directly in hardware
        std.log.info("FPGA MatMul: Using transposed weight layout", .{});
    }

    /// Batched matrix multiplication for multi-head attention
    pub fn executeBatched(
        self: *QuantizedMatMulKernel,
        batch_activations: []const []const f32,
        quantized_weights: []const u8,
        batch_outputs: [][]f32,
    ) !void {
        std.debug.assert(batch_activations.len == batch_outputs.len);

        // FPGA would process all batches in parallel using multiple compute units
        for (batch_activations, batch_outputs) |act, out| {
            try self.execute(act, quantized_weights, out);
        }
    }

    fn executeCpuFallback(
        self: *QuantizedMatMulKernel,
        activations: []const f32,
        quantized_weights: []const u8,
        output: []f32,
    ) !void {
        const m = self.config.m;
        const n = self.config.n;
        const k = self.config.k;
        const scale = self.weight_params.scale;

        // Initialize output to zero
        @memset(output, 0);

        // Dequantize and multiply (tiled for cache efficiency)
        const tile_m = @min(self.config.tile_m, m);
        const tile_n = @min(self.config.tile_n, n);
        const tile_k = @min(self.config.tile_k, k);

        var mi: usize = 0;
        while (mi < m) : (mi += tile_m) {
            const m_end = @min(mi + tile_m, m);

            var ni: usize = 0;
            while (ni < n) : (ni += tile_n) {
                const n_end = @min(ni + tile_n, n);

                var ki: usize = 0;
                while (ki < k) : (ki += tile_k) {
                    const k_end = @min(ki + tile_k, k);

                    // Compute tile
                    for (mi..m_end) |i| {
                        for (ni..n_end) |j| {
                            var sum: f32 = output[i * n + j];

                            for (ki..k_end) |kk| {
                                const a_val = activations[i * k + kk];
                                const w_idx = kk * n + j;

                                // Dequantize weight based on precision
                                const w_val = self.dequantizeWeight(quantized_weights, w_idx);
                                sum += a_val * w_val * scale;
                            }

                            output[i * n + j] = sum;
                        }
                    }
                }
            }
        }
    }

    fn dequantizeWeight(self: *const QuantizedMatMulKernel, weights: []const u8, idx: usize) f32 {
        return switch (self.config.weight_precision) {
            .fp32 => blk: {
                const bytes = weights[idx * 4 ..][0..4];
                break :blk @bitCast(bytes.*);
            },
            .fp16, .bf16 => blk: {
                if (idx * 2 + 1 >= weights.len) break :blk 0.0;
                const bytes = weights[idx * 2 ..][0..2];
                const bits = @as(u16, bytes[0]) | (@as(u16, bytes[1]) << 8);
                // Simple fp16 to fp32 conversion
                break :blk fp16ToF32(bits);
            },
            .q8, .q8_k => blk: {
                if (idx >= weights.len) break :blk 0.0;
                break :blk @as(f32, @floatFromInt(@as(i8, @bitCast(weights[idx]))));
            },
            .q4, .q4_k => blk: {
                const byte_idx = idx / 2;
                if (byte_idx >= weights.len) break :blk 0.0;
                const nibble = if (idx % 2 == 0)
                    weights[byte_idx] & 0x0F
                else
                    weights[byte_idx] >> 4;
                // Sign-extend 4-bit to 8-bit, then to float
                const signed = @as(i8, @intCast(nibble)) - 8;
                break :blk @as(f32, @floatFromInt(signed));
            },
        };
    }

    fn fp16ToF32(bits: u16) f32 {
        const sign: u32 = @as(u32, bits >> 15) << 31;
        const exp: u32 = (bits >> 10) & 0x1F;
        const mant: u32 = bits & 0x3FF;

        if (exp == 0) {
            // Denormal or zero
            if (mant == 0) return @bitCast(sign);
            // Denormal - simplified handling
            return @as(f32, @floatFromInt(@as(i16, @bitCast(bits)))) / 32768.0;
        } else if (exp == 31) {
            // Inf or NaN
            const result: u32 = sign | 0x7F800000 | (@as(u32, mant) << 13);
            return @bitCast(result);
        }

        // Normal number
        const new_exp: u32 = @as(u32, exp) - 15 + 127;
        const result: u32 = sign | (new_exp << 23) | (@as(u32, mant) << 13);
        return @bitCast(result);
    }
};

/// Fused MatMul + Bias + Activation kernel
pub const FusedMatMulKernel = struct {
    matmul: QuantizedMatMulKernel,
    activation_type: ActivationType = .none,
    bias: ?[]const f32 = null,

    pub const ActivationType = enum {
        none,
        relu,
        gelu,
        silu, // SwiGLU component
        tanh,
    };

    pub fn init(allocator: std.mem.Allocator, config: MatMulKernelConfig) !FusedMatMulKernel {
        return FusedMatMulKernel{
            .matmul = try QuantizedMatMulKernel.init(allocator, config),
        };
    }

    pub fn deinit(self: *FusedMatMulKernel) void {
        self.matmul.deinit();
    }

    pub fn setBias(self: *FusedMatMulKernel, bias: []const f32) void {
        self.bias = bias;
    }

    pub fn setActivation(self: *FusedMatMulKernel, activation: ActivationType) void {
        self.activation_type = activation;
    }

    /// Execute fused MatMul + Bias + Activation
    pub fn execute(
        self: *FusedMatMulKernel,
        activations: []const f32,
        quantized_weights: []const u8,
        output: []f32,
    ) !void {
        // FPGA would fuse all operations in hardware
        // to avoid memory round-trips

        try self.matmul.execute(activations, quantized_weights, output);

        const n = self.matmul.config.n;
        const m = self.matmul.config.m;

        // Apply bias
        if (self.bias) |bias| {
            for (0..m) |i| {
                for (0..n) |j| {
                    output[i * n + j] += bias[j];
                }
            }
        }

        // Apply activation
        switch (self.activation_type) {
            .none => {},
            .relu => {
                for (output) |*val| {
                    val.* = @max(0.0, val.*);
                }
            },
            .gelu => {
                for (output) |*val| {
                    val.* = gelu(val.*);
                }
            },
            .silu => {
                for (output) |*val| {
                    val.* = silu(val.*);
                }
            },
            .tanh => {
                for (output) |*val| {
                    val.* = std.math.tanh(val.*);
                }
            },
        }
    }

    fn gelu(x: f32) f32 {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const sqrt_2_over_pi: f32 = 0.7978845608;
        const coeff: f32 = 0.044715;
        const inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        return 0.5 * x * (1.0 + std.math.tanh(inner));
    }

    fn silu(x: f32) f32 {
        // SiLU/Swish: x * sigmoid(x)
        return x / (1.0 + @exp(-x));
    }
};

/// Performance metrics for MatMul operations
pub const MatMulMetrics = struct {
    total_flops: u64 = 0,
    execution_time_ns: u64 = 0,
    memory_bandwidth_gbps: f64 = 0,
    compute_utilization: f32 = 0,

    pub fn computeGflops(self: *const MatMulMetrics) f64 {
        if (self.execution_time_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.total_flops)) / @as(f64, @floatFromInt(self.execution_time_ns));
    }

    pub fn report(self: *const MatMulMetrics) void {
        std.log.info("MatMul Performance:", .{});
        std.log.info("  GFLOPS: {d:.2}", .{self.computeGflops()});
        std.log.info("  Memory BW: {d:.2} GB/s", .{self.memory_bandwidth_gbps});
        std.log.info("  Compute Util: {d:.1}%", .{self.compute_utilization * 100});
    }
};

// Tests

test "quantized matmul basic" {
    const allocator = std.testing.allocator;

    const config = MatMulKernelConfig{
        .m = 2,
        .n = 3,
        .k = 4,
        .weight_precision = .q8,
    };

    var kernel = try QuantizedMatMulKernel.init(allocator, config);
    defer kernel.deinit();

    // 2x4 activations
    const activations = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    // 4x3 = 12 weights as Q8
    const weights = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var output: [6]f32 = undefined;

    try kernel.execute(&activations, &weights, &output);

    // Verify output is computed (basic sanity check)
    for (output) |val| {
        try std.testing.expect(!std.math.isNan(val));
    }
}

test "fused matmul with activation" {
    const allocator = std.testing.allocator;

    const config = MatMulKernelConfig{
        .m = 2,
        .n = 2,
        .k = 2,
        .weight_precision = .fp32,
    };

    var kernel = try FusedMatMulKernel.init(allocator, config);
    defer kernel.deinit();

    kernel.setActivation(.relu);

    const activations = [_]f32{ 1, -1, -1, 1 };
    const weights = [_]u8{0} ** 16; // Placeholder
    var output: [4]f32 = undefined;

    try kernel.execute(&activations, &weights, &output);

    // After ReLU, negative values should be 0
    for (output) |val| {
        try std.testing.expect(val >= 0);
    }
}

test "weight precision compression ratio" {
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), WeightPrecision.q4.compressionRatio(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), WeightPrecision.q8.compressionRatio(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), WeightPrecision.fp16.compressionRatio(), 0.001);
}

/// FPGA-optimized batch matrix multiplication for multi-head attention
/// Designed for LLM inference with multiple attention heads
pub const BatchMatMulKernel = struct {
    config: MatMulKernelConfig,
    allocator: std.mem.Allocator,

    // FPGA-specific configuration
    parallel_heads: u32 = 8, // Number of attention heads processed in parallel
    shared_weights: bool = true, // Share weight matrix across heads for efficiency
    memory_layout: MemoryLayout = .interleaved, // How batch elements are stored

    pub const MemoryLayout = enum {
        contiguous, // All heads stored contiguously
        interleaved, // tiled across compute units
        blocked, // blocked for cache efficiency
    };

    pub fn init(allocator: std.mem.Allocator, config: MatMulKernelConfig) !BatchMatMulKernel {
        return BatchMatMulKernel{
            .config = config,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BatchMatMulKernel) void {
        _ = self;
    }

    /// Set FPGA-specific parameters
    pub fn setFpgaParams(self: *BatchMatMulKernel, parallel_heads: u32, memory_layout: MemoryLayout) void {
        self.parallel_heads = parallel_heads;
        self.memory_layout = memory_layout;
    }

    /// Execute batch matrix multiplication for attention heads
    /// batch_activations: [batch, M, K] activations for each head
    /// quantized_weights: [K, N] shared weights (if shared_weights = true)
    /// batch_outputs: [batch, M, N] outputs for each head
    pub fn executeMultiHead(
        self: *BatchMatMulKernel,
        batch_activations: []const []const f32,
        quantized_weights: []const u8,
        batch_outputs: [][]f32,
    ) !void {
        const batch_size = batch_activations.len;
        std.debug.assert(batch_size == batch_outputs.len);

        // FPGA implementation strategy:
        // 1. Stream batch activations to multiple on-chip buffers
        // 2. Replicate shared weights across compute units (if shared_weights)
        // 3. Execute parallel matmul using spatial architecture
        // 4. Stream results back with minimal synchronization

        if (self.shared_weights) {
            // Optimized path: share weights across all heads
            // FPGA can broadcast weights to all compute units simultaneously

            // Process heads in parallel groups (size = parallel_heads)
            var head_idx: usize = 0;
            while (head_idx < batch_size) {
                const group_end = @min(head_idx + self.parallel_heads, batch_size);

                // FPGA would process this group in parallel
                for (head_idx..group_end) |i| {
                    try self.executeSingleHead(
                        batch_activations[i],
                        quantized_weights,
                        batch_outputs[i],
                    );
                }

                head_idx = group_end;
            }
        } else {
            // Each head has separate weights (different projections)
            // FPGA would need separate weight loading for each compute unit

            for (batch_activations, batch_outputs) |activations, output| {
                try self.executeSingleHead(activations, quantized_weights, output);
            }
        }
    }

    /// Execute single head with FPGA optimizations
    fn executeSingleHead(
        self: *BatchMatMulKernel,
        activations: []const f32,
        quantized_weights: []const u8,
        output: []f32,
    ) !void {
        const m = self.config.m;
        const n = self.config.n;
        const k = self.config.k;

        std.debug.assert(activations.len == m * k);
        std.debug.assert(output.len == m * n);

        // FPGA tiled implementation
        const tile_m = @min(self.config.tile_m, m);
        const tile_n = @min(self.config.tile_n, n);
        const tile_k = @min(self.config.tile_k, k);

        // Initialize output
        @memset(output, 0);

        // FPGA would use spatial architecture with multiple
        // compute units working on different tiles simultaneously

        var mi: usize = 0;
        while (mi < m) : (mi += tile_m) {
            const m_end = @min(mi + tile_m, m);

            var ni: usize = 0;
            while (ni < n) : (ni += tile_n) {
                const n_end = @min(ni + tile_n, n);

                var ki: usize = 0;
                while (ki < k) : (ki += tile_k) {
                    const k_end = @min(ki + tile_k, k);

                    // FPGA compute unit would process this tile
                    for (mi..m_end) |i| {
                        for (ni..n_end) |j| {
                            var sum: f32 = output[i * n + j];

                            for (ki..k_end) |kk| {
                                const a_val = activations[i * k + kk];
                                const w_idx = kk * n + j;

                                const w_val = dequantizeWeightForPrecision(
                                    quantized_weights,
                                    w_idx,
                                    self.config.weight_precision,
                                    1.0, // scale
                                );
                                sum += a_val * w_val;
                            }

                            output[i * n + j] = sum;
                        }
                    }
                }
            }
        }
    }

    /// Performance-optimized for large batches (attention during generation)
    pub fn executeMultiHeadOptimized(
        self: *BatchMatMulKernel,
        batch_activations: []const []const f32,
        batch_weights: ?[]const []const u8, // Optional per-head weights
        batch_outputs: [][]f32,
        metrics: ?*MatMulMetrics,
    ) !void {
        const batch_size = batch_activations.len;

        if (metrics) |m| {
            // Use Timer for Zig 0.16 compatibility (no std.time.nanoTimestamp())
            var timer = std.time.Timer.start() catch {
                // If timer fails, execute without timing
                for (batch_activations, batch_outputs, 0..) |activations, output, i| {
                    const weights = if (batch_weights) |bw| bw[i] else self.allocator.alloc(u8, 0) catch unreachable;
                    defer if (batch_weights != null) {};
                    try self.executeSingleHead(activations, weights, output);
                }
                return;
            };

            // FPGA would have specialized pipeline for batch processing

            for (batch_activations, batch_outputs, 0..) |activations, output, i| {
                const weights = if (batch_weights) |bw| bw[i] else self.allocator.alloc(u8, 0) catch unreachable;
                defer if (batch_weights != null) {};

                try self.executeSingleHead(activations, weights, output);
            }

            m.execution_time_ns = @intCast(timer.read());
            m.total_flops = @as(u64, batch_size) * 2 * self.config.m * self.config.n * self.config.k;
        } else {
            for (batch_activations, batch_outputs, 0..) |activations, output, i| {
                const weights = if (batch_weights) |bw| bw[i] else self.allocator.alloc(u8, 0) catch unreachable;
                defer if (batch_weights != null) {};

                try self.executeSingleHead(activations, weights, output);
            }
        }
    }
};

/// Helper function for weight dequantization
fn dequantizeWeightForPrecision(
    weights: []const u8,
    idx: usize,
    precision: WeightPrecision,
    scale: f32,
) f32 {
    return switch (precision) {
        .fp32 => blk: {
            if (idx * 4 + 3 >= weights.len) break :blk 0.0;
            const bytes = weights[idx * 4 ..][0..4];
            const bits = std.mem.readIntLittle(u32, bytes);
            break :blk @bitCast(bits);
        },
        .fp16, .bf16 => blk: {
            if (idx * 2 + 1 >= weights.len) break :blk 0.0;
            const bytes = weights[idx * 2 ..][0..2];
            const bits = @as(u16, bytes[0]) | (@as(u16, bytes[1]) << 8);
            // Simplified conversion
            break :blk @as(f32, @floatFromInt(bits)) / 65536.0;
        },
        .q8, .q8_k => blk: {
            if (idx >= weights.len) break :blk 0.0;
            const val = @as(i8, @bitCast(weights[idx]));
            break :blk @as(f32, @floatFromInt(val)) * scale;
        },
        .q4, .q4_k => blk: {
            const byte_idx = idx / 2;
            if (byte_idx >= weights.len) break :blk 0.0;
            const nibble = if (idx % 2 == 0)
                weights[byte_idx] & 0x0F
            else
                weights[byte_idx] >> 4;
            const signed = @as(i8, @intCast(nibble)) - 8;
            break :blk @as(f32, @floatFromInt(signed)) * scale;
        },
    };
}
