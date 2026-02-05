//! Quantized Matrix Multiplication Metal Kernels
//!
//! Provides GPU-accelerated implementations for quantized matrix operations on Apple Silicon:
//! - Q4_0 matrix-vector multiplication with fused dequantization
//! - Q8_0 matrix-vector multiplication with fused dequantization
//! - Batched operations for efficient token processing
//!
//! These kernels fuse dequantization and multiplication into a single pass,
//! avoiding intermediate memory allocations and achieving 4x memory efficiency
//! compared to float32 operations.
//!
//! ## Metal-Specific Optimizations
//! - Uses SIMD groups (32 threads) for efficient reductions
//! - Threadgroup memory for warp-level communication
//! - Metal's half-precision support for scale factors
//! - Optimized for Apple Silicon's unified memory architecture

const std = @import("std");
const builtin = @import("builtin");
const metal = @import("../metal.zig");
const types = @import("../../kernel_types.zig");

pub const QuantKernelError = error{
    MetalNotAvailable,
    CompilationFailed,
    KernelLaunchFailed,
    MemoryError,
    NotInitialized,
    InvalidQuantFormat,
};

/// Configuration for quantized kernel operations.
pub const QuantConfig = struct {
    /// Threads per threadgroup for Metal kernel launches.
    threads_per_group: u32 = 256,
    /// Enable performance statistics collection.
    enable_stats: bool = true,
    /// Preferred quantization format for new operations.
    preferred_format: QuantFormat = .q4_0,
    /// Maximum batch size for batched operations.
    max_batch_size: u32 = 64,
    /// Enable kernel fusion optimizations (SwiGLU, RMSNorm).
    enable_fusion: bool = true,

    pub const QuantFormat = enum {
        q4_0,
        q4_1,
        q5_0,
        q5_1,
        q8_0,
    };

    /// Default configuration for LLM inference.
    pub fn forInference() QuantConfig {
        return .{
            .threads_per_group = 256,
            .enable_stats = false,
            .preferred_format = .q4_0,
            .enable_fusion = true,
        };
    }

    /// Configuration optimized for debugging/profiling.
    pub fn forProfiling() QuantConfig {
        return .{
            .threads_per_group = 128,
            .enable_stats = true,
            .preferred_format = .q8_0,
            .enable_fusion = false,
        };
    }
};

/// Q4_0 block size (elements per block)
pub const Q4_BLOCK_SIZE: u32 = 32;
/// Q4_0 bytes per block (16 4-bit pairs + 2 bytes scale)
pub const Q4_BLOCK_BYTES: u32 = 18;

/// Q8_0 block size (elements per block)
pub const Q8_BLOCK_SIZE: u32 = 32;
/// Q8_0 bytes per block (32 int8 + 2 bytes scale)
pub const Q8_BLOCK_BYTES: u32 = 34;

/// Metal Shading Language kernel for Q4_0 matrix-vector multiplication.
/// Each threadgroup processes one output row.
/// Uses SIMD group reduction for efficient summation.
const Q4_MATMUL_KERNEL_MSL =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void q4_matmul_kernel(
    \\    device const uchar* a_quant [[buffer(0)]],    // [M, K/32 * 18] Q4_0 quantized
    \\    device const float* x [[buffer(1)]],           // [K] input vector
    \\    device float* y [[buffer(2)]],                 // [M] output vector
    \\    constant int& M [[buffer(3)]],                 // output dimension
    \\    constant int& K [[buffer(4)]],                 // inner dimension
    \\    constant int& blocks_per_row [[buffer(5)]],    // K / 32
    \\    uint row [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint simd_lane [[thread_index_in_simdgroup]],
    \\    uint simd_group [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    if (row >= uint(M)) return;
    \\
    \\    const uint threads_per_group = 256;
    \\    const uint num_simd_groups = threads_per_group / 32;
    \\
    \\    // Each thread accumulates partial sum
    \\    float local_sum = 0.0f;
    \\
    \\    // Row start in quantized data
    \\    device const uchar* row_data = a_quant + row * blocks_per_row * 18;
    \\
    \\    // Process blocks assigned to this thread
    \\    for (uint b = tid; b < uint(blocks_per_row); b += threads_per_group) {
    \\        // Read scale (f16 stored as 2 bytes)
    \\        device const uchar* block_ptr = row_data + b * 18;
    \\        const half scale_h = *reinterpret_cast<device const half*>(block_ptr);
    \\        const float scale = float(scale_h);
    \\        device const uchar* quants = block_ptr + 2;
    \\
    \\        float block_sum = 0.0f;
    \\        const uint base = b * 32;
    \\
    \\        // Process 16 bytes (32 4-bit values)
    \\        for (uint i = 0; i < 16; i++) {
    \\            const uchar byte = quants[i];
    \\            const int lo = int(byte & 0x0F) - 8;
    \\            const int hi = int(byte >> 4) - 8;
    \\
    \\            block_sum += float(lo) * x[base + i];
    \\            block_sum += float(hi) * x[base + i + 16];
    \\        }
    \\
    \\        local_sum += block_sum * scale;
    \\    }
    \\
    \\    // SIMD group reduction
    \\    local_sum = simd_sum(local_sum);
    \\
    \\    // First thread in each SIMD group writes to threadgroup memory
    \\    threadgroup float simd_sums[8];  // Max 8 SIMD groups (256/32)
    \\    if (simd_lane == 0) {
    \\        simd_sums[simd_group] = local_sum;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // First SIMD group reduces all SIMD group sums
    \\    if (simd_group == 0) {
    \\        local_sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
    \\        local_sum = simd_sum(local_sum);
    \\
    \\        if (simd_lane == 0) {
    \\            y[row] = local_sum;
    \\        }
    \\    }
    \\}
;

/// Metal Shading Language kernel for Q8_0 matrix-vector multiplication.
const Q8_MATMUL_KERNEL_MSL =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void q8_matmul_kernel(
    \\    device const uchar* a_quant [[buffer(0)]],    // [M, K/32 * 34] Q8_0 quantized
    \\    device const float* x [[buffer(1)]],           // [K] input vector
    \\    device float* y [[buffer(2)]],                 // [M] output vector
    \\    constant int& M [[buffer(3)]],
    \\    constant int& K [[buffer(4)]],
    \\    constant int& blocks_per_row [[buffer(5)]],
    \\    uint row [[threadgroup_position_in_grid]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint simd_lane [[thread_index_in_simdgroup]],
    \\    uint simd_group [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    if (row >= uint(M)) return;
    \\
    \\    const uint threads_per_group = 256;
    \\    const uint num_simd_groups = threads_per_group / 32;
    \\
    \\    float local_sum = 0.0f;
    \\
    \\    device const uchar* row_data = a_quant + row * blocks_per_row * 34;
    \\
    \\    for (uint b = tid; b < uint(blocks_per_row); b += threads_per_group) {
    \\        device const uchar* block_ptr = row_data + b * 34;
    \\        const half scale_h = *reinterpret_cast<device const half*>(block_ptr);
    \\        const float scale = float(scale_h);
    \\        device const char* quants = reinterpret_cast<device const char*>(block_ptr + 2);
    \\
    \\        float block_sum = 0.0f;
    \\        const uint base = b * 32;
    \\
    \\        // Process 32 int8 values with loop unrolling
    \\        for (uint i = 0; i < 32; i += 4) {
    \\            block_sum += float(quants[i + 0]) * x[base + i + 0];
    \\            block_sum += float(quants[i + 1]) * x[base + i + 1];
    \\            block_sum += float(quants[i + 2]) * x[base + i + 2];
    \\            block_sum += float(quants[i + 3]) * x[base + i + 3];
    \\        }
    \\
    \\        local_sum += block_sum * scale;
    \\    }
    \\
    \\    // SIMD group reduction
    \\    local_sum = simd_sum(local_sum);
    \\
    \\    threadgroup float simd_sums[8];
    \\    if (simd_lane == 0) {
    \\        simd_sums[simd_group] = local_sum;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    if (simd_group == 0) {
    \\        local_sum = (simd_lane < num_simd_groups) ? simd_sums[simd_lane] : 0.0f;
    \\        local_sum = simd_sum(local_sum);
    \\
    \\        if (simd_lane == 0) {
    \\            y[row] = local_sum;
    \\        }
    \\    }
    \\}
;

/// Fused SwiGLU kernel for FFN: out = silu(gate) * up
const SWIGLU_KERNEL_MSL =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void swiglu_kernel(
    \\    device float* gate [[buffer(0)]],      // [N] gate projection (modified in-place)
    \\    device const float* up [[buffer(1)]],  // [N] up projection
    \\    constant int& n [[buffer(2)]],
    \\    uint i [[thread_position_in_grid]]
    \\) {
    \\    if (i < uint(n)) {
    \\        const float g = gate[i];
    \\        const float silu_g = g / (1.0f + exp(-g));  // SiLU(gate)
    \\        gate[i] = silu_g * up[i];  // SiLU(gate) * up
    \\    }
    \\}
;

/// Fused RMSNorm + scale kernel
const RMSNORM_SCALE_KERNEL_MSL =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void rmsnorm_scale_kernel(
    \\    device const float* x [[buffer(0)]],       // [N] input
    \\    device const float* weight [[buffer(1)]],  // [N] norm weights
    \\    device float* out [[buffer(2)]],           // [N] output
    \\    constant float& inv_rms [[buffer(3)]],     // 1/sqrt(mean(x^2) + eps)
    \\    constant int& n [[buffer(4)]],
    \\    uint i [[thread_position_in_grid]]
    \\) {
    \\    if (i < uint(n)) {
    \\        out[i] = x[i] * inv_rms * weight[i];
    \\    }
    \\}
;

/// Softmax kernel with numerical stability
const SOFTMAX_KERNEL_MSL =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void softmax_kernel(
    \\    device float* x [[buffer(0)]],
    \\    constant int& size [[buffer(1)]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint simd_lane [[thread_index_in_simdgroup]],
    \\    uint simd_group [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const uint threads_per_group = 256;
    \\
    \\    // Find max value
    \\    float local_max = -1e30f;
    \\    for (uint i = tid; i < uint(size); i += threads_per_group) {
    \\        local_max = max(local_max, x[i]);
    \\    }
    \\
    \\    // SIMD reduction for max
    \\    local_max = simd_max(local_max);
    \\
    \\    threadgroup float shared_maxes[8];
    \\    if (simd_lane == 0) {
    \\        shared_maxes[simd_group] = local_max;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    if (simd_group == 0) {
    \\        local_max = (simd_lane < 8) ? shared_maxes[simd_lane] : -1e30f;
    \\        local_max = simd_max(local_max);
    \\    }
    \\    threadgroup float shared_max;
    \\    if (tid == 0) shared_max = local_max;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Compute exp(x - max) and sum
    \\    float local_sum = 0.0f;
    \\    for (uint i = tid; i < uint(size); i += threads_per_group) {
    \\        x[i] = exp(x[i] - shared_max);
    \\        local_sum += x[i];
    \\    }
    \\
    \\    // SIMD reduction for sum
    \\    local_sum = simd_sum(local_sum);
    \\
    \\    threadgroup float shared_sums[8];
    \\    if (simd_lane == 0) {
    \\        shared_sums[simd_group] = local_sum;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    if (simd_group == 0) {
    \\        local_sum = (simd_lane < 8) ? shared_sums[simd_lane] : 0.0f;
    \\        local_sum = simd_sum(local_sum);
    \\    }
    \\    threadgroup float shared_sum;
    \\    if (tid == 0) shared_sum = local_sum;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Normalize
    \\    const float inv_sum = 1.0f / shared_sum;
    \\    for (uint i = tid; i < uint(size); i += threads_per_group) {
    \\        x[i] *= inv_sum;
    \\    }
    \\}
;

/// RMSNorm kernel
const RMSNORM_KERNEL_MSL =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void rmsnorm_kernel(
    \\    device float* x [[buffer(0)]],
    \\    device const float* weight [[buffer(1)]],
    \\    constant int& size [[buffer(2)]],
    \\    constant float& eps [[buffer(3)]],
    \\    uint tid [[thread_index_in_threadgroup]],
    \\    uint simd_lane [[thread_index_in_simdgroup]],
    \\    uint simd_group [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const uint threads_per_group = 256;
    \\
    \\    // Compute sum of squares
    \\    float local_ss = 0.0f;
    \\    for (uint i = tid; i < uint(size); i += threads_per_group) {
    \\        local_ss += x[i] * x[i];
    \\    }
    \\
    \\    // SIMD reduction
    \\    local_ss = simd_sum(local_ss);
    \\
    \\    threadgroup float shared_ss[8];
    \\    if (simd_lane == 0) {
    \\        shared_ss[simd_group] = local_ss;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    if (simd_group == 0) {
    \\        local_ss = (simd_lane < 8) ? shared_ss[simd_lane] : 0.0f;
    \\        local_ss = simd_sum(local_ss);
    \\    }
    \\
    \\    threadgroup float inv_rms;
    \\    if (tid == 0) {
    \\        inv_rms = rsqrt(local_ss / float(size) + eps);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    // Normalize and apply weight
    \\    for (uint i = tid; i < uint(size); i += threads_per_group) {
    \\        x[i] = x[i] * inv_rms * weight[i];
    \\    }
    \\}
;

/// SiLU activation kernel
const SILU_KERNEL_MSL =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\kernel void silu_kernel(
    \\    device float* x [[buffer(0)]],
    \\    constant int& n [[buffer(1)]],
    \\    uint i [[thread_position_in_grid]]
    \\) {
    \\    if (i < uint(n)) {
    \\        const float val = x[i];
    \\        x[i] = val / (1.0f + exp(-val));
    \\    }
    \\}
;

/// Quantized kernel module for GPU-accelerated quantized operations on Metal.
pub const QuantizedKernelModule = struct {
    allocator: std.mem.Allocator,
    /// Compiled Q4 kernel
    q4_kernel: ?*anyopaque,
    /// Compiled Q8 kernel
    q8_kernel: ?*anyopaque,
    /// SwiGLU kernel
    swiglu_kernel: ?*anyopaque,
    /// RMSNorm scale kernel
    rmsnorm_scale_kernel: ?*anyopaque,
    /// Softmax kernel
    softmax_kernel: ?*anyopaque,
    /// RMSNorm kernel
    rmsnorm_kernel: ?*anyopaque,
    /// SiLU kernel
    silu_kernel: ?*anyopaque,
    /// Whether Metal is available
    metal_available: bool,
    /// Statistics
    stats: KernelStats,

    pub const KernelStats = struct {
        q4_ops: u64 = 0,
        q8_ops: u64 = 0,
        swiglu_ops: u64 = 0,
        softmax_ops: u64 = 0,
        rmsnorm_ops: u64 = 0,
        silu_ops: u64 = 0,
        total_time_ns: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator) !QuantizedKernelModule {
        var self = QuantizedKernelModule{
            .allocator = allocator,
            .q4_kernel = null,
            .q8_kernel = null,
            .swiglu_kernel = null,
            .rmsnorm_scale_kernel = null,
            .softmax_kernel = null,
            .rmsnorm_kernel = null,
            .silu_kernel = null,
            .metal_available = false,
            .stats = .{},
        };

        // Check if Metal is available (macOS only)
        if (builtin.target.os.tag != .macos) {
            return self;
        }

        // Try to initialize Metal
        metal.init() catch {
            return self;
        };

        self.metal_available = true;

        // Compile Q4 kernel
        self.q4_kernel = compileKernel(allocator, Q4_MATMUL_KERNEL_MSL, "q4_matmul_kernel") catch |err| blk: {
            std.log.warn("Failed to compile Q4 matmul kernel: {}", .{err});
            break :blk null;
        };

        // Compile Q8 kernel
        self.q8_kernel = compileKernel(allocator, Q8_MATMUL_KERNEL_MSL, "q8_matmul_kernel") catch |err| blk: {
            std.log.warn("Failed to compile Q8 matmul kernel: {}", .{err});
            break :blk null;
        };

        // Compile SwiGLU kernel
        self.swiglu_kernel = compileKernel(allocator, SWIGLU_KERNEL_MSL, "swiglu_kernel") catch |err| blk: {
            std.log.warn("Failed to compile SwiGLU kernel: {}", .{err});
            break :blk null;
        };

        // Compile RMSNorm scale kernel
        self.rmsnorm_scale_kernel = compileKernel(allocator, RMSNORM_SCALE_KERNEL_MSL, "rmsnorm_scale_kernel") catch |err| blk: {
            std.log.warn("Failed to compile RMSNorm scale kernel: {}", .{err});
            break :blk null;
        };

        // Compile Softmax kernel
        self.softmax_kernel = compileKernel(allocator, SOFTMAX_KERNEL_MSL, "softmax_kernel") catch |err| blk: {
            std.log.warn("Failed to compile Softmax kernel: {}", .{err});
            break :blk null;
        };

        // Compile RMSNorm kernel
        self.rmsnorm_kernel = compileKernel(allocator, RMSNORM_KERNEL_MSL, "rmsnorm_kernel") catch |err| blk: {
            std.log.warn("Failed to compile RMSNorm kernel: {}", .{err});
            break :blk null;
        };

        // Compile SiLU kernel
        self.silu_kernel = compileKernel(allocator, SILU_KERNEL_MSL, "silu_kernel") catch |err| blk: {
            std.log.warn("Failed to compile SiLU kernel: {}", .{err});
            break :blk null;
        };

        return self;
    }

    pub fn deinit(self: *QuantizedKernelModule) void {
        if (self.q4_kernel) |k| metal.destroyKernel(self.allocator, k);
        if (self.q8_kernel) |k| metal.destroyKernel(self.allocator, k);
        if (self.swiglu_kernel) |k| metal.destroyKernel(self.allocator, k);
        if (self.rmsnorm_scale_kernel) |k| metal.destroyKernel(self.allocator, k);
        if (self.softmax_kernel) |k| metal.destroyKernel(self.allocator, k);
        if (self.rmsnorm_kernel) |k| metal.destroyKernel(self.allocator, k);
        if (self.silu_kernel) |k| metal.destroyKernel(self.allocator, k);
        self.* = undefined;
    }

    /// Check if quantized kernels are available.
    pub fn isAvailable(self: *const QuantizedKernelModule) bool {
        return self.metal_available and self.q4_kernel != null;
    }

    /// Q4_0 matrix-vector multiplication on GPU.
    pub fn q4Matmul(
        self: *QuantizedKernelModule,
        a_quant_buffer: *anyopaque,
        x_buffer: *anyopaque,
        y_buffer: *anyopaque,
        m: u32,
        k: u32,
    ) QuantKernelError!void {
        const kernel = self.q4_kernel orelse return QuantKernelError.NotInitialized;

        const blocks_per_row = k / Q4_BLOCK_SIZE;

        // Prepare constant buffers for M, K, and blocks_per_row
        var m_val: i32 = @intCast(m);
        var k_val: i32 = @intCast(k);
        var bpr_val: i32 = @intCast(blocks_per_row);

        const m_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, m_buffer);

        const k_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, k_buffer);

        const bpr_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, bpr_buffer);

        // Copy constants to device
        metal.memcpyHostToDevice(m_buffer, @ptrCast(&m_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        metal.memcpyHostToDevice(k_buffer, @ptrCast(&k_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        metal.memcpyHostToDevice(bpr_buffer, @ptrCast(&bpr_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;

        const args = [_]?*const anyopaque{
            a_quant_buffer,
            x_buffer,
            y_buffer,
            m_buffer,
            k_buffer,
            bpr_buffer,
        };

        const config = types.KernelConfig{
            .grid_size = .{ m, 1, 1 },
            .block_size = .{ 256, 1, 1 },
            .shared_memory_size = 0,
        };

        metal.launchKernel(self.allocator, kernel, config, &args) catch
            return QuantKernelError.KernelLaunchFailed;

        self.stats.q4_ops += 1;
    }

    /// Q8_0 matrix-vector multiplication on GPU.
    pub fn q8Matmul(
        self: *QuantizedKernelModule,
        a_quant_buffer: *anyopaque,
        x_buffer: *anyopaque,
        y_buffer: *anyopaque,
        m: u32,
        k: u32,
    ) QuantKernelError!void {
        const kernel = self.q8_kernel orelse return QuantKernelError.NotInitialized;

        const blocks_per_row = k / Q8_BLOCK_SIZE;

        var m_val: i32 = @intCast(m);
        var k_val: i32 = @intCast(k);
        var bpr_val: i32 = @intCast(blocks_per_row);

        const m_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, m_buffer);

        const k_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, k_buffer);

        const bpr_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, bpr_buffer);

        metal.memcpyHostToDevice(m_buffer, @ptrCast(&m_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        metal.memcpyHostToDevice(k_buffer, @ptrCast(&k_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        metal.memcpyHostToDevice(bpr_buffer, @ptrCast(&bpr_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;

        const args = [_]?*const anyopaque{
            a_quant_buffer,
            x_buffer,
            y_buffer,
            m_buffer,
            k_buffer,
            bpr_buffer,
        };

        const config = types.KernelConfig{
            .grid_size = .{ m, 1, 1 },
            .block_size = .{ 256, 1, 1 },
            .shared_memory_size = 0,
        };

        metal.launchKernel(self.allocator, kernel, config, &args) catch
            return QuantKernelError.KernelLaunchFailed;

        self.stats.q8_ops += 1;
    }

    /// Fused SwiGLU activation: gate = silu(gate) * up
    pub fn fusedSwiglu(
        self: *QuantizedKernelModule,
        gate_buffer: *anyopaque,
        up_buffer: *anyopaque,
        n: u32,
    ) QuantKernelError!void {
        const kernel = self.swiglu_kernel orelse return QuantKernelError.NotInitialized;

        var n_val: i32 = @intCast(n);

        const n_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, n_buffer);

        metal.memcpyHostToDevice(n_buffer, @ptrCast(&n_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;

        const args = [_]?*const anyopaque{
            gate_buffer,
            up_buffer,
            n_buffer,
        };

        const threads_per_group: u32 = 256;
        const grid_size = (n + threads_per_group - 1) / threads_per_group;

        const config = types.KernelConfig{
            .grid_size = .{ grid_size, 1, 1 },
            .block_size = .{ threads_per_group, 1, 1 },
            .shared_memory_size = 0,
        };

        metal.launchKernel(self.allocator, kernel, config, &args) catch
            return QuantKernelError.KernelLaunchFailed;

        self.stats.swiglu_ops += 1;
    }

    /// Softmax operation on GPU.
    pub fn softmax(
        self: *QuantizedKernelModule,
        x_buffer: *anyopaque,
        size: u32,
    ) QuantKernelError!void {
        const kernel = self.softmax_kernel orelse return QuantKernelError.NotInitialized;

        var size_val: i32 = @intCast(size);

        const size_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, size_buffer);

        metal.memcpyHostToDevice(size_buffer, @ptrCast(&size_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;

        const args = [_]?*const anyopaque{
            x_buffer,
            size_buffer,
        };

        const config = types.KernelConfig{
            .grid_size = .{ 1, 1, 1 },
            .block_size = .{ 256, 1, 1 },
            .shared_memory_size = 0,
        };

        metal.launchKernel(self.allocator, kernel, config, &args) catch
            return QuantKernelError.KernelLaunchFailed;

        self.stats.softmax_ops += 1;
    }

    /// RMSNorm operation on GPU.
    pub fn rmsnorm(
        self: *QuantizedKernelModule,
        x_buffer: *anyopaque,
        weight_buffer: *anyopaque,
        size: u32,
        eps: f32,
    ) QuantKernelError!void {
        const kernel = self.rmsnorm_kernel orelse return QuantKernelError.NotInitialized;

        var size_val: i32 = @intCast(size);
        var eps_val: f32 = eps;

        const size_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, size_buffer);

        const eps_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(f32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, eps_buffer);

        metal.memcpyHostToDevice(size_buffer, @ptrCast(&size_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        metal.memcpyHostToDevice(eps_buffer, @ptrCast(&eps_val), @sizeOf(f32)) catch
            return QuantKernelError.MemoryError;

        const args = [_]?*const anyopaque{
            x_buffer,
            weight_buffer,
            size_buffer,
            eps_buffer,
        };

        const config = types.KernelConfig{
            .grid_size = .{ 1, 1, 1 },
            .block_size = .{ 256, 1, 1 },
            .shared_memory_size = 0,
        };

        metal.launchKernel(self.allocator, kernel, config, &args) catch
            return QuantKernelError.KernelLaunchFailed;

        self.stats.rmsnorm_ops += 1;
    }

    /// SiLU activation on GPU.
    pub fn silu(
        self: *QuantizedKernelModule,
        x_buffer: *anyopaque,
        n: u32,
    ) QuantKernelError!void {
        const kernel = self.silu_kernel orelse return QuantKernelError.NotInitialized;

        var n_val: i32 = @intCast(n);

        const n_buffer = metal.allocateDeviceMemory(self.allocator, @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;
        defer metal.freeDeviceMemory(self.allocator, n_buffer);

        metal.memcpyHostToDevice(n_buffer, @ptrCast(&n_val), @sizeOf(i32)) catch
            return QuantKernelError.MemoryError;

        const args = [_]?*const anyopaque{
            x_buffer,
            n_buffer,
        };

        const threads_per_group: u32 = 256;
        const grid_size = (n + threads_per_group - 1) / threads_per_group;

        const config = types.KernelConfig{
            .grid_size = .{ grid_size, 1, 1 },
            .block_size = .{ threads_per_group, 1, 1 },
            .shared_memory_size = 0,
        };

        metal.launchKernel(self.allocator, kernel, config, &args) catch
            return QuantKernelError.KernelLaunchFailed;

        self.stats.silu_ops += 1;
    }

    /// Get kernel statistics.
    pub fn getStats(self: *const QuantizedKernelModule) KernelStats {
        return self.stats;
    }
};

/// Compile an MSL kernel from source.
fn compileKernel(allocator: std.mem.Allocator, source: []const u8, entry_point: []const u8) !*anyopaque {
    const kernel_source = types.KernelSource{
        .code = source,
        .entry_point = entry_point,
        .language = .msl,
    };
    return try metal.compileKernel(allocator, kernel_source);
}

/// Check if Metal quantized kernels are available.
pub fn isAvailable() bool {
    if (builtin.target.os.tag != .macos) return false;
    metal.init() catch return false;
    return true;
}

// ============================================================================
// Tests
// ============================================================================

test "quantized kernels module init" {
    var module = try QuantizedKernelModule.init(std.testing.allocator);
    defer module.deinit();

    // On non-macOS systems, module should still init (with unavailable kernels)
    if (builtin.target.os.tag != .macos) {
        try std.testing.expect(!module.isAvailable());
    }
}

test "kernel source availability" {
    // Verify kernel sources are valid strings
    try std.testing.expect(Q4_MATMUL_KERNEL_MSL.len > 100);
    try std.testing.expect(Q8_MATMUL_KERNEL_MSL.len > 100);
    try std.testing.expect(SWIGLU_KERNEL_MSL.len > 50);
    try std.testing.expect(SOFTMAX_KERNEL_MSL.len > 50);
    try std.testing.expect(RMSNORM_KERNEL_MSL.len > 50);
    try std.testing.expect(SILU_KERNEL_MSL.len > 50);
}

test "quantization constants" {
    try std.testing.expectEqual(@as(u32, 32), Q4_BLOCK_SIZE);
    try std.testing.expectEqual(@as(u32, 18), Q4_BLOCK_BYTES);
    try std.testing.expectEqual(@as(u32, 32), Q8_BLOCK_SIZE);
    try std.testing.expectEqual(@as(u32, 34), Q8_BLOCK_BYTES);
}
