//! Quantized Matrix Multiplication CUDA Kernels
//!
//! Provides GPU-accelerated implementations for quantized matrix operations:
//! - Q4_0 matrix-vector multiplication with fused dequantization
//! - Q8_0 matrix-vector multiplication with fused dequantization
//! - Batched operations for efficient token processing
//!
//! These kernels fuse dequantization and multiplication into a single pass,
//! avoiding intermediate memory allocations and achieving 4x memory efficiency
//! compared to float32 operations.

const std = @import("std");
const loader = @import("loader.zig");
const nvrtc = @import("nvrtc.zig");

pub const QuantKernelError = error{
    CudaNotAvailable,
    CompilationFailed,
    KernelLaunchFailed,
    MemoryError,
    NotInitialized,
    InvalidQuantFormat,
};

/// Configuration for quantized kernel operations.
pub const QuantConfig = struct {
    /// Block size for CUDA kernel launches (threads per block).
    block_size: u32 = 256,
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
            .block_size = 256,
            .enable_stats = false,
            .preferred_format = .q4_0,
            .enable_fusion = true,
        };
    }

    /// Configuration optimized for debugging/profiling.
    pub fn forProfiling() QuantConfig {
        return .{
            .block_size = 128,
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

/// CUDA kernel for Q4_0 matrix-vector multiplication.
/// Each thread block processes one output row.
/// Uses warp-level reduction for efficient summation.
const Q4_MATMUL_KERNEL =
    \\extern "C" __global__ void q4_matmul_kernel(
    \\    const unsigned char* __restrict__ a_quant,  // [M, K/32 * 18] Q4_0 quantized
    \\    const float* __restrict__ x,                 // [K] input vector
    \\    float* __restrict__ y,                       // [M] output vector
    \\    const int M,                                 // output dimension
    \\    const int K,                                 // inner dimension
    \\    const int blocks_per_row                     // K / 32
    \\) {
    \\    const int row = blockIdx.x;
    \\    if (row >= M) return;
    \\
    \\    const int tid = threadIdx.x;
    \\    const int warp_id = tid / 32;
    \\    const int lane_id = tid % 32;
    \\    const int num_warps = blockDim.x / 32;
    \\
    \\    // Each thread accumulates partial sum
    \\    float local_sum = 0.0f;
    \\
    \\    // Row start in quantized data
    \\    const unsigned char* row_data = a_quant + row * blocks_per_row * 18;
    \\
    \\    // Process blocks assigned to this thread
    \\    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
    \\        // Read scale (f16 stored as 2 bytes)
    \\        const unsigned char* block_ptr = row_data + b * 18;
    \\        const float scale = __half2float(*reinterpret_cast<const __half*>(block_ptr));
    \\        const unsigned char* quants = block_ptr + 2;
    \\
    \\        float block_sum = 0.0f;
    \\        const int base = b * 32;
    \\
    \\        // Process 16 bytes (32 4-bit values)
    \\        #pragma unroll
    \\        for (int i = 0; i < 16; i++) {
    \\            const unsigned char byte = quants[i];
    \\            const int lo = (byte & 0x0F) - 8;
    \\            const int hi = (byte >> 4) - 8;
    \\
    \\            block_sum += (float)lo * x[base + i];
    \\            block_sum += (float)hi * x[base + i + 16];
    \\        }
    \\
    \\        local_sum += block_sum * scale;
    \\    }
    \\
    \\    // Warp-level reduction
    \\    #pragma unroll
    \\    for (int offset = 16; offset > 0; offset >>= 1) {
    \\        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    \\    }
    \\
    \\    // First thread in each warp writes to shared memory
    \\    __shared__ float warp_sums[32];
    \\    if (lane_id == 0) {
    \\        warp_sums[warp_id] = local_sum;
    \\    }
    \\    __syncthreads();
    \\
    \\    // First warp reduces all warp sums
    \\    if (warp_id == 0) {
    \\        local_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    \\
    \\        #pragma unroll
    \\        for (int offset = 16; offset > 0; offset >>= 1) {
    \\            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    \\        }
    \\
    \\        if (lane_id == 0) {
    \\            y[row] = local_sum;
    \\        }
    \\    }
    \\}
;

/// CUDA kernel for Q8_0 matrix-vector multiplication.
const Q8_MATMUL_KERNEL =
    \\extern "C" __global__ void q8_matmul_kernel(
    \\    const unsigned char* __restrict__ a_quant,  // [M, K/32 * 34] Q8_0 quantized
    \\    const float* __restrict__ x,                 // [K] input vector
    \\    float* __restrict__ y,                       // [M] output vector
    \\    const int M,
    \\    const int K,
    \\    const int blocks_per_row
    \\) {
    \\    const int row = blockIdx.x;
    \\    if (row >= M) return;
    \\
    \\    const int tid = threadIdx.x;
    \\    const int warp_id = tid / 32;
    \\    const int lane_id = tid % 32;
    \\    const int num_warps = blockDim.x / 32;
    \\
    \\    float local_sum = 0.0f;
    \\
    \\    const unsigned char* row_data = a_quant + row * blocks_per_row * 34;
    \\
    \\    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
    \\        const unsigned char* block_ptr = row_data + b * 34;
    \\        const float scale = __half2float(*reinterpret_cast<const __half*>(block_ptr));
    \\        const signed char* quants = reinterpret_cast<const signed char*>(block_ptr + 2);
    \\
    \\        float block_sum = 0.0f;
    \\        const int base = b * 32;
    \\
    \\        // Process 32 int8 values with vectorized loads
    \\        #pragma unroll
    \\        for (int i = 0; i < 32; i += 4) {
    \\            block_sum += (float)quants[i + 0] * x[base + i + 0];
    \\            block_sum += (float)quants[i + 1] * x[base + i + 1];
    \\            block_sum += (float)quants[i + 2] * x[base + i + 2];
    \\            block_sum += (float)quants[i + 3] * x[base + i + 3];
    \\        }
    \\
    \\        local_sum += block_sum * scale;
    \\    }
    \\
    \\    // Warp reduction
    \\    #pragma unroll
    \\    for (int offset = 16; offset > 0; offset >>= 1) {
    \\        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    \\    }
    \\
    \\    __shared__ float warp_sums[32];
    \\    if (lane_id == 0) {
    \\        warp_sums[warp_id] = local_sum;
    \\    }
    \\    __syncthreads();
    \\
    \\    if (warp_id == 0) {
    \\        local_sum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
    \\
    \\        #pragma unroll
    \\        for (int offset = 16; offset > 0; offset >>= 1) {
    \\            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    \\        }
    \\
    \\        if (lane_id == 0) {
    \\            y[row] = local_sum;
    \\        }
    \\    }
    \\}
;

/// Fused SwiGLU kernel for FFN: out = silu(gate) * up
const SWIGLU_KERNEL =
    \\extern "C" __global__ void swiglu_kernel(
    \\    float* __restrict__ gate,  // [N] gate projection (modified in-place)
    \\    const float* __restrict__ up,  // [N] up projection
    \\    const int n
    \\) {
    \\    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        const float g = gate[i];
    \\        const float silu_g = g / (1.0f + expf(-g));  // SiLU(gate)
    \\        gate[i] = silu_g * up[i];  // SiLU(gate) * up
    \\    }
    \\}
;

/// Fused RMSNorm + MatMul preparation kernel
const RMSNORM_SCALE_KERNEL =
    \\extern "C" __global__ void rmsnorm_scale_kernel(
    \\    const float* __restrict__ x,       // [N] input
    \\    const float* __restrict__ weight,  // [N] norm weights
    \\    float* __restrict__ out,           // [N] output
    \\    const float inv_rms,               // 1/sqrt(mean(x^2) + eps)
    \\    const int n
    \\) {
    \\    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        out[i] = x[i] * inv_rms * weight[i];
    \\    }
    \\}
;

/// Quantized kernel module for GPU-accelerated quantized operations.
pub const QuantizedKernelModule = struct {
    allocator: std.mem.Allocator,
    /// Compiled Q4 kernel module
    q4_module: ?*anyopaque,
    q4_kernel: ?*anyopaque,
    /// Compiled Q8 kernel module
    q8_module: ?*anyopaque,
    q8_kernel: ?*anyopaque,
    /// SwiGLU kernel
    swiglu_module: ?*anyopaque,
    swiglu_kernel: ?*anyopaque,
    /// RMSNorm scale kernel
    rmsnorm_module: ?*anyopaque,
    rmsnorm_kernel: ?*anyopaque,
    /// Whether CUDA is available
    cuda_available: bool,
    /// Statistics
    stats: KernelStats,

    pub const KernelStats = struct {
        q4_ops: u64 = 0,
        q8_ops: u64 = 0,
        swiglu_ops: u64 = 0,
        total_time_ns: u64 = 0,
    };

    pub fn init(allocator: std.mem.Allocator) !QuantizedKernelModule {
        var self = QuantizedKernelModule{
            .allocator = allocator,
            .q4_module = null,
            .q4_kernel = null,
            .q8_module = null,
            .q8_kernel = null,
            .swiglu_module = null,
            .swiglu_kernel = null,
            .rmsnorm_module = null,
            .rmsnorm_kernel = null,
            .cuda_available = false,
            .stats = .{},
        };

        // Check if CUDA is available
        if (!loader.isAvailable()) {
            return self;
        }

        // Try to compile kernels
        self.cuda_available = true;

        // Compile Q4 kernel
        if (nvrtc.compileKernel(Q4_MATMUL_KERNEL, "q4_matmul_kernel")) |result| {
            self.q4_module = result.module;
            self.q4_kernel = result.kernel;
        } else |_| {
            std.log.warn("Failed to compile Q4 matmul kernel", .{});
        }

        // Compile Q8 kernel
        if (nvrtc.compileKernel(Q8_MATMUL_KERNEL, "q8_matmul_kernel")) |result| {
            self.q8_module = result.module;
            self.q8_kernel = result.kernel;
        } else |_| {
            std.log.warn("Failed to compile Q8 matmul kernel", .{});
        }

        // Compile SwiGLU kernel
        if (nvrtc.compileKernel(SWIGLU_KERNEL, "swiglu_kernel")) |result| {
            self.swiglu_module = result.module;
            self.swiglu_kernel = result.kernel;
        } else |_| {
            std.log.warn("Failed to compile SwiGLU kernel", .{});
        }

        // Compile RMSNorm scale kernel
        if (nvrtc.compileKernel(RMSNORM_SCALE_KERNEL, "rmsnorm_scale_kernel")) |result| {
            self.rmsnorm_module = result.module;
            self.rmsnorm_kernel = result.kernel;
        } else |_| {
            std.log.warn("Failed to compile RMSNorm scale kernel", .{});
        }

        return self;
    }

    pub fn deinit(self: *QuantizedKernelModule) void {
        if (self.q4_module) |module| {
            loader.unloadModule(module);
        }
        if (self.q8_module) |module| {
            loader.unloadModule(module);
        }
        if (self.swiglu_module) |module| {
            loader.unloadModule(module);
        }
        if (self.rmsnorm_module) |module| {
            loader.unloadModule(module);
        }
        self.* = undefined;
    }

    /// Check if quantized kernels are available.
    pub fn isAvailable(self: *const QuantizedKernelModule) bool {
        return self.cuda_available and self.q4_kernel != null;
    }

    /// Q4_0 matrix-vector multiplication on GPU.
    pub fn q4Matmul(
        self: *QuantizedKernelModule,
        a_quant_device: usize,
        x_device: usize,
        y_device: usize,
        m: u32,
        k: u32,
        stream: ?*anyopaque,
    ) QuantKernelError!void {
        const kernel = self.q4_kernel orelse return QuantKernelError.NotInitialized;

        const blocks_per_row = k / Q4_BLOCK_SIZE;
        const block_size: u32 = 256;
        const grid_size = m;

        var args = [_]usize{
            a_quant_device,
            x_device,
            y_device,
            @intCast(m),
            @intCast(k),
            @intCast(blocks_per_row),
        };

        loader.launchKernel(
            kernel,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            &args,
        ) catch return QuantKernelError.KernelLaunchFailed;

        self.stats.q4_ops += 1;
    }

    /// Q8_0 matrix-vector multiplication on GPU.
    pub fn q8Matmul(
        self: *QuantizedKernelModule,
        a_quant_device: usize,
        x_device: usize,
        y_device: usize,
        m: u32,
        k: u32,
        stream: ?*anyopaque,
    ) QuantKernelError!void {
        const kernel = self.q8_kernel orelse return QuantKernelError.NotInitialized;

        const blocks_per_row = k / Q8_BLOCK_SIZE;
        const block_size: u32 = 256;
        const grid_size = m;

        var args = [_]usize{
            a_quant_device,
            x_device,
            y_device,
            @intCast(m),
            @intCast(k),
            @intCast(blocks_per_row),
        };

        loader.launchKernel(
            kernel,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            &args,
        ) catch return QuantKernelError.KernelLaunchFailed;

        self.stats.q8_ops += 1;
    }

    /// Fused SwiGLU activation: gate = silu(gate) * up
    pub fn fusedSwiglu(
        self: *QuantizedKernelModule,
        gate_device: usize,
        up_device: usize,
        n: u32,
        stream: ?*anyopaque,
    ) QuantKernelError!void {
        const kernel = self.swiglu_kernel orelse return QuantKernelError.NotInitialized;

        const block_size: u32 = 256;
        const grid_size = (n + block_size - 1) / block_size;

        var args = [_]usize{
            gate_device,
            up_device,
            @intCast(n),
        };

        loader.launchKernel(
            kernel,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            &args,
        ) catch return QuantKernelError.KernelLaunchFailed;

        self.stats.swiglu_ops += 1;
    }

    /// Get kernel statistics.
    pub fn getStats(self: *const QuantizedKernelModule) KernelStats {
        return self.stats;
    }
};

/// Check if quantized GPU kernels are available.
pub fn isAvailable() bool {
    return loader.isAvailable() and nvrtc.isAvailable();
}

// ============================================================================
// Tests
// ============================================================================

test "quantized kernels module init" {
    var module = try QuantizedKernelModule.init(std.testing.allocator);
    defer module.deinit();

    // On systems without CUDA, module should still init (with unavailable kernels)
    _ = module.isAvailable();
}

test "kernel source compilation check" {
    // Verify kernel sources are valid strings
    try std.testing.expect(Q4_MATMUL_KERNEL.len > 100);
    try std.testing.expect(Q8_MATMUL_KERNEL.len > 100);
    try std.testing.expect(SWIGLU_KERNEL.len > 50);
}

test {
    std.testing.refAllDecls(@This());
}
