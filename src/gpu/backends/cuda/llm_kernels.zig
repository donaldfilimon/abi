//! CUDA Kernels for LLM Operations
//!
//! Provides GPU-accelerated implementations of common LLM operations:
//! - Softmax (row-wise and in-place)
//! - RMSNorm (Root Mean Square Normalization)
//! - SiLU (Swish activation)
//! - Element-wise operations (add, mul, scale)
//!
//! Uses NVRTC for runtime PTX compilation with kernel caching.

const std = @import("std");
const loader = @import("loader.zig");
const nvrtc = @import("nvrtc.zig");

pub const LlmKernelError = error{
    CudaNotAvailable,
    CompilationFailed,
    KernelLaunchFailed,
    MemoryError,
    NotInitialized,
};

/// CUDA kernel sources for LLM operations.
const SOFTMAX_KERNEL =
    \\extern "C" __global__ void softmax_kernel(
    \\    float* x,
    \\    const int size
    \\) {
    \\    // Block-level softmax for numerical stability
    \\    __shared__ float shared_max;
    \\    __shared__ float shared_sum;
    \\
    \\    const int tid = threadIdx.x;
    \\    const int block_size = blockDim.x;
    \\
    \\    // Find max value
    \\    float local_max = -1e30f;
    \\    for (int i = tid; i < size; i += block_size) {
    \\        local_max = fmaxf(local_max, x[i]);
    \\    }
    \\
    \\    // Reduce max across threads
    \\    __shared__ float shared_maxes[256];
    \\    shared_maxes[tid] = local_max;
    \\    __syncthreads();
    \\
    \\    for (int s = block_size / 2; s > 0; s >>= 1) {
    \\        if (tid < s && tid + s < block_size) {
    \\            shared_maxes[tid] = fmaxf(shared_maxes[tid], shared_maxes[tid + s]);
    \\        }
    \\        __syncthreads();
    \\    }
    \\
    \\    if (tid == 0) shared_max = shared_maxes[0];
    \\    __syncthreads();
    \\
    \\    // Compute exp(x - max) and sum
    \\    float local_sum = 0.0f;
    \\    for (int i = tid; i < size; i += block_size) {
    \\        x[i] = expf(x[i] - shared_max);
    \\        local_sum += x[i];
    \\    }
    \\
    \\    // Reduce sum across threads
    \\    shared_maxes[tid] = local_sum;
    \\    __syncthreads();
    \\
    \\    for (int s = block_size / 2; s > 0; s >>= 1) {
    \\        if (tid < s && tid + s < block_size) {
    \\            shared_maxes[tid] += shared_maxes[tid + s];
    \\        }
    \\        __syncthreads();
    \\    }
    \\
    \\    if (tid == 0) shared_sum = shared_maxes[0];
    \\    __syncthreads();
    \\
    \\    // Normalize
    \\    const float inv_sum = 1.0f / shared_sum;
    \\    for (int i = tid; i < size; i += block_size) {
    \\        x[i] *= inv_sum;
    \\    }
    \\}
;

const RMSNORM_KERNEL =
    \\extern "C" __global__ void rmsnorm_kernel(
    \\    float* x,
    \\    const float* weight,
    \\    const int size,
    \\    const float eps
    \\) {
    \\    const int tid = threadIdx.x;
    \\    const int block_size = blockDim.x;
    \\
    \\    // Compute sum of squares
    \\    __shared__ float shared_ss[256];
    \\    float local_ss = 0.0f;
    \\
    \\    for (int i = tid; i < size; i += block_size) {
    \\        local_ss += x[i] * x[i];
    \\    }
    \\
    \\    shared_ss[tid] = local_ss;
    \\    __syncthreads();
    \\
    \\    // Reduce
    \\    for (int s = block_size / 2; s > 0; s >>= 1) {
    \\        if (tid < s && tid + s < block_size) {
    \\            shared_ss[tid] += shared_ss[tid + s];
    \\        }
    \\        __syncthreads();
    \\    }
    \\
    \\    __shared__ float inv_rms;
    \\    if (tid == 0) {
    \\        inv_rms = rsqrtf(shared_ss[0] / size + eps);
    \\    }
    \\    __syncthreads();
    \\
    \\    // Normalize and apply weight
    \\    for (int i = tid; i < size; i += block_size) {
    \\        x[i] = x[i] * inv_rms * weight[i];
    \\    }
    \\}
;

const SILU_KERNEL =
    \\extern "C" __global__ void silu_kernel(
    \\    float* x,
    \\    const int n
    \\) {
    \\    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        const float val = x[i];
    \\        x[i] = val / (1.0f + expf(-val));
    \\    }
    \\}
;

const ELEMENTWISE_MUL_KERNEL =
    \\extern "C" __global__ void elementwise_mul_kernel(
    \\    float* a,
    \\    const float* b,
    \\    const int n
    \\) {
    \\    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        a[i] *= b[i];
    \\    }
    \\}
;

const ELEMENTWISE_ADD_KERNEL =
    \\extern "C" __global__ void elementwise_add_kernel(
    \\    float* a,
    \\    const float* b,
    \\    const int n
    \\) {
    \\    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        a[i] += b[i];
    \\    }
    \\}
;

const SCALE_KERNEL =
    \\extern "C" __global__ void scale_kernel(
    \\    float* x,
    \\    const float scale,
    \\    const int n
    \\) {
    \\    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        x[i] *= scale;
    \\    }
    \\}
;

const GELU_KERNEL =
    \\extern "C" __global__ void gelu_kernel(
    \\    float* x,
    \\    const int n
    \\) {
    \\    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    \\    if (i < n) {
    \\        // GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    \\        const float val = x[i];
    \\        const float sqrt_2_over_pi = 0.7978845608f;
    \\        const float val3 = val * val * val;
    \\        const float inner = sqrt_2_over_pi * (val + 0.044715f * val3);
    \\        x[i] = 0.5f * val * (1.0f + tanhf(inner));
    \\    }
    \\}
;

/// Fused Attention Kernel: Computes Q*K^T, softmax, and @V in a single pass.
/// This avoids storing the full N×N attention matrix in global memory.
///
/// Each thread block processes one query row against all K/V.
/// Uses shared memory for local softmax computation.
const FUSED_ATTENTION_KERNEL =
    \\extern "C" __global__ void fused_attention_kernel(
    \\    const float* __restrict__ Q,      // [seq_len, head_dim]
    \\    const float* __restrict__ K,      // [kv_len, head_dim]
    \\    const float* __restrict__ V,      // [kv_len, head_dim]
    \\    float* __restrict__ output,       // [seq_len, head_dim]
    \\    const int seq_len,
    \\    const int kv_len,
    \\    const int head_dim,
    \\    const float scale,
    \\    const int causal                  // 0 = no mask, 1 = causal mask
    \\) {
    \\    // Each block handles one query position
    \\    const int q_idx = blockIdx.x;
    \\    if (q_idx >= seq_len) return;
    \\
    \\    const int tid = threadIdx.x;
    \\    const int block_size = blockDim.x;
    \\
    \\    // Shared memory for attention scores and reduction
    \\    extern __shared__ float shared[];
    \\    float* scores = shared;                           // [kv_len]
    \\    float* shared_max = shared + kv_len;              // [block_size]
    \\    float* shared_sum = shared_max + block_size;      // [block_size]
    \\
    \\    // Pointer to this query row
    \\    const float* q_row = Q + q_idx * head_dim;
    \\
    \\    // Step 1: Compute Q @ K^T for this query row
    \\    // Each thread computes dot products for a subset of K rows
    \\    float local_max = -1e30f;
    \\
    \\    for (int k_idx = tid; k_idx < kv_len; k_idx += block_size) {
    \\        // Apply causal mask: mask future positions
    \\        if (causal && k_idx > q_idx) {
    \\            scores[k_idx] = -1e30f;
    \\        } else {
    \\            // Compute dot product Q[q_idx] @ K[k_idx]
    \\            float dot = 0.0f;
    \\            const float* k_row = K + k_idx * head_dim;
    \\            for (int d = 0; d < head_dim; d++) {
    \\                dot += q_row[d] * k_row[d];
    \\            }
    \\            scores[k_idx] = dot * scale;
    \\        }
    \\        local_max = fmaxf(local_max, scores[k_idx]);
    \\    }
    \\    __syncthreads();
    \\
    \\    // Step 2: Reduce to find global max
    \\    shared_max[tid] = local_max;
    \\    __syncthreads();
    \\
    \\    for (int s = block_size / 2; s > 0; s >>= 1) {
    \\        if (tid < s && tid + s < block_size) {
    \\            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
    \\        }
    \\        __syncthreads();
    \\    }
    \\    float max_val = shared_max[0];
    \\
    \\    // Step 3: Compute exp(score - max) and sum
    \\    float local_sum = 0.0f;
    \\    for (int k_idx = tid; k_idx < kv_len; k_idx += block_size) {
    \\        float exp_val = expf(scores[k_idx] - max_val);
    \\        scores[k_idx] = exp_val;
    \\        local_sum += exp_val;
    \\    }
    \\    __syncthreads();
    \\
    \\    // Step 4: Reduce sum
    \\    shared_sum[tid] = local_sum;
    \\    __syncthreads();
    \\
    \\    for (int s = block_size / 2; s > 0; s >>= 1) {
    \\        if (tid < s && tid + s < block_size) {
    \\            shared_sum[tid] += shared_sum[tid + s];
    \\        }
    \\        __syncthreads();
    \\    }
    \\    float sum_val = shared_sum[0];
    \\    float inv_sum = 1.0f / sum_val;
    \\
    \\    // Step 5: Normalize and compute output = softmax(scores) @ V
    \\    // Each thread computes a subset of output dimensions
    \\    float* out_row = output + q_idx * head_dim;
    \\
    \\    for (int d = tid; d < head_dim; d += block_size) {
    \\        float acc = 0.0f;
    \\        for (int k_idx = 0; k_idx < kv_len; k_idx++) {
    \\            float attn = scores[k_idx] * inv_sum;
    \\            acc += attn * V[k_idx * head_dim + d];
    \\        }
    \\        out_row[d] = acc;
    \\    }
    \\}
;

/// Flash-style Fused Attention with tiling for memory efficiency.
/// Processes K,V in blocks to reduce shared memory usage.
const FUSED_ATTENTION_TILED_KERNEL =
    \\extern "C" __global__ void fused_attention_tiled_kernel(
    \\    const float* __restrict__ Q,      // [seq_len, head_dim]
    \\    const float* __restrict__ K,      // [kv_len, head_dim]
    \\    const float* __restrict__ V,      // [kv_len, head_dim]
    \\    float* __restrict__ output,       // [seq_len, head_dim]
    \\    const int seq_len,
    \\    const int kv_len,
    \\    const int head_dim,
    \\    const float scale,
    \\    const int causal,
    \\    const int block_kv                // Tile size for K,V
    \\) {
    \\    const int q_idx = blockIdx.x;
    \\    if (q_idx >= seq_len) return;
    \\
    \\    const int tid = threadIdx.x;
    \\    const int block_size = blockDim.x;
    \\
    \\    // Shared memory layout
    \\    extern __shared__ float shared[];
    \\    float* tile_scores = shared;                       // [block_kv]
    \\    float* tile_k = tile_scores + block_kv;            // [block_kv * head_dim]
    \\    float* tile_v = tile_k + block_kv * head_dim;      // [block_kv * head_dim]
    \\    float* reduce_buf = tile_v + block_kv * head_dim;  // [block_size * 2]
    \\
    \\    // Query row pointer
    \\    const float* q_row = Q + q_idx * head_dim;
    \\
    \\    // Running statistics for online softmax
    \\    float running_max = -1e30f;
    \\    float running_sum = 0.0f;
    \\
    \\    // Output accumulator (per thread, for subset of dimensions)
    \\    float acc[32];  // Assume head_dim <= 32 * block_size
    \\    for (int i = 0; i < 32; i++) acc[i] = 0.0f;
    \\
    \\    // Process K,V in tiles
    \\    for (int kv_start = 0; kv_start < kv_len; kv_start += block_kv) {
    \\        int kv_end = min(kv_start + block_kv, kv_len);
    \\        int tile_len = kv_end - kv_start;
    \\
    \\        // Skip if all positions are masked (causal)
    \\        if (causal && kv_start > q_idx) break;
    \\
    \\        // Load K tile to shared memory
    \\        for (int i = tid; i < tile_len * head_dim; i += block_size) {
    \\            int k_idx = i / head_dim;
    \\            int d = i % head_dim;
    \\            tile_k[k_idx * head_dim + d] = K[(kv_start + k_idx) * head_dim + d];
    \\        }
    \\        // Load V tile to shared memory
    \\        for (int i = tid; i < tile_len * head_dim; i += block_size) {
    \\            int v_idx = i / head_dim;
    \\            int d = i % head_dim;
    \\            tile_v[v_idx * head_dim + d] = V[(kv_start + v_idx) * head_dim + d];
    \\        }
    \\        __syncthreads();
    \\
    \\        // Compute scores for this tile
    \\        float tile_max = -1e30f;
    \\        for (int k = tid; k < tile_len; k += block_size) {
    \\            int global_k = kv_start + k;
    \\            if (causal && global_k > q_idx) {
    \\                tile_scores[k] = -1e30f;
    \\            } else {
    \\                float dot = 0.0f;
    \\                for (int d = 0; d < head_dim; d++) {
    \\                    dot += q_row[d] * tile_k[k * head_dim + d];
    \\                }
    \\                tile_scores[k] = dot * scale;
    \\            }
    \\            tile_max = fmaxf(tile_max, tile_scores[k]);
    \\        }
    \\        __syncthreads();
    \\
    \\        // Reduce tile max
    \\        reduce_buf[tid] = tile_max;
    \\        __syncthreads();
    \\        for (int s = block_size / 2; s > 0; s >>= 1) {
    \\            if (tid < s) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
    \\            __syncthreads();
    \\        }
    \\        tile_max = reduce_buf[0];
    \\
    \\        // Online softmax: rescale previous accumulator if max changed
    \\        float new_max = fmaxf(running_max, tile_max);
    \\        float rescale = expf(running_max - new_max);
    \\        running_sum *= rescale;
    \\        for (int i = 0; i < 32; i++) acc[i] *= rescale;
    \\
    \\        // Compute exp and sum for this tile
    \\        float tile_sum = 0.0f;
    \\        for (int k = tid; k < tile_len; k += block_size) {
    \\            float exp_val = expf(tile_scores[k] - new_max);
    \\            tile_scores[k] = exp_val;
    \\            tile_sum += exp_val;
    \\        }
    \\        __syncthreads();
    \\
    \\        // Reduce tile sum
    \\        reduce_buf[tid] = tile_sum;
    \\        __syncthreads();
    \\        for (int s = block_size / 2; s > 0; s >>= 1) {
    \\            if (tid < s) reduce_buf[tid] += reduce_buf[tid + s];
    \\            __syncthreads();
    \\        }
    \\        running_sum += reduce_buf[0];
    \\        running_max = new_max;
    \\
    \\        // Accumulate: output += scores @ V_tile
    \\        for (int d = tid; d < head_dim; d += block_size) {
    \\            float sum = 0.0f;
    \\            for (int k = 0; k < tile_len; k++) {
    \\                sum += tile_scores[k] * tile_v[k * head_dim + d];
    \\            }
    \\            acc[d / block_size] += sum;
    \\        }
    \\        __syncthreads();
    \\    }
    \\
    \\    // Final normalization and write output
    \\    float inv_sum = 1.0f / running_sum;
    \\    float* out_row = output + q_idx * head_dim;
    \\    for (int d = tid; d < head_dim; d += block_size) {
    \\        out_row[d] = acc[d / block_size] * inv_sum;
    \\    }
    \\}
;

/// Compiled CUDA module containing LLM kernels.
pub const LlmKernelModule = struct {
    allocator: std.mem.Allocator,
    module: ?*anyopaque,
    softmax_fn: ?*anyopaque,
    rmsnorm_fn: ?*anyopaque,
    silu_fn: ?*anyopaque,
    gelu_fn: ?*anyopaque,
    elementwise_mul_fn: ?*anyopaque,
    elementwise_add_fn: ?*anyopaque,
    scale_fn: ?*anyopaque,
    fused_attention_fn: ?*anyopaque,
    fused_attention_tiled_fn: ?*anyopaque,
    cuda_fns: *const loader.CudaFunctions,

    /// Initialize and compile all LLM kernels.
    pub fn init(allocator: std.mem.Allocator) !LlmKernelModule {
        if (!loader.isAvailable()) {
            return LlmKernelError.CudaNotAvailable;
        }

        const cuda_fns = loader.getFunctions() orelse return LlmKernelError.CudaNotAvailable;

        // Try to initialize NVRTC
        nvrtc.init() catch {
            return LlmKernelError.CompilationFailed;
        };

        // Compile kernels
        const combined_source = SOFTMAX_KERNEL ++ "\n" ++ RMSNORM_KERNEL ++ "\n" ++
            SILU_KERNEL ++ "\n" ++ GELU_KERNEL ++ "\n" ++ ELEMENTWISE_MUL_KERNEL ++ "\n" ++
            ELEMENTWISE_ADD_KERNEL ++ "\n" ++ SCALE_KERNEL ++ "\n" ++
            FUSED_ATTENTION_KERNEL ++ "\n" ++ FUSED_ATTENTION_TILED_KERNEL;

        const compile_result = nvrtc.compileToPTX(
            allocator,
            combined_source,
            "llm_kernels",
            .{},
        ) catch {
            return LlmKernelError.CompilationFailed;
        };
        defer allocator.free(compile_result.ptx);
        defer if (compile_result.log.len > 0) allocator.free(compile_result.log);

        // Load PTX module
        const module_load_fn = cuda_fns.kernel.cuModuleLoadData orelse
            return LlmKernelError.CudaNotAvailable;
        const get_fn = cuda_fns.kernel.cuModuleGetFunction orelse
            return LlmKernelError.CudaNotAvailable;

        var module: ?*anyopaque = null;
        if (module_load_fn(&module, compile_result.ptx.ptr) != .success) {
            return LlmKernelError.CompilationFailed;
        }

        // Get function handles
        var softmax_fn: ?*anyopaque = null;
        var rmsnorm_fn: ?*anyopaque = null;
        var silu_fn: ?*anyopaque = null;
        var gelu_fn: ?*anyopaque = null;
        var elementwise_mul_fn: ?*anyopaque = null;
        var elementwise_add_fn: ?*anyopaque = null;
        var scale_fn: ?*anyopaque = null;
        var fused_attention_fn: ?*anyopaque = null;
        var fused_attention_tiled_fn: ?*anyopaque = null;

        if (get_fn(&softmax_fn, module, "softmax_kernel") != .success) {
            std.log.warn("Failed to get softmax_kernel function", .{});
        }
        if (get_fn(&rmsnorm_fn, module, "rmsnorm_kernel") != .success) {
            std.log.warn("Failed to get rmsnorm_kernel function", .{});
        }
        if (get_fn(&silu_fn, module, "silu_kernel") != .success) {
            std.log.warn("Failed to get silu_kernel function", .{});
        }
        if (get_fn(&gelu_fn, module, "gelu_kernel") != .success) {
            std.log.warn("Failed to get gelu_kernel function", .{});
        }
        if (get_fn(&elementwise_mul_fn, module, "elementwise_mul_kernel") != .success) {
            std.log.warn("Failed to get elementwise_mul_kernel function", .{});
        }
        if (get_fn(&elementwise_add_fn, module, "elementwise_add_kernel") != .success) {
            std.log.warn("Failed to get elementwise_add_kernel function", .{});
        }
        if (get_fn(&scale_fn, module, "scale_kernel") != .success) {
            std.log.warn("Failed to get scale_kernel function", .{});
        }
        if (get_fn(&fused_attention_fn, module, "fused_attention_kernel") != .success) {
            std.log.warn("Failed to get fused_attention_kernel function", .{});
        }
        if (get_fn(&fused_attention_tiled_fn, module, "fused_attention_tiled_kernel") != .success) {
            std.log.warn("Failed to get fused_attention_tiled_kernel function", .{});
        }

        return .{
            .allocator = allocator,
            .module = module,
            .softmax_fn = softmax_fn,
            .rmsnorm_fn = rmsnorm_fn,
            .silu_fn = silu_fn,
            .gelu_fn = gelu_fn,
            .elementwise_mul_fn = elementwise_mul_fn,
            .elementwise_add_fn = elementwise_add_fn,
            .scale_fn = scale_fn,
            .fused_attention_fn = fused_attention_fn,
            .fused_attention_tiled_fn = fused_attention_tiled_fn,
            .cuda_fns = cuda_fns,
        };
    }

    pub fn deinit(self: *LlmKernelModule) void {
        if (self.module) |mod| {
            const unload_fn = self.cuda_fns.kernel.cuModuleUnload orelse return;
            _ = unload_fn(mod);
        }
        self.* = undefined;
    }

    /// Launch softmax kernel on GPU data.
    pub fn softmax(self: *LlmKernelModule, device_ptr: u64, size: u32, stream: ?*anyopaque) !void {
        const launch_fn = self.cuda_fns.kernel.cuLaunchKernel orelse
            return LlmKernelError.CudaNotAvailable;
        const fn_ptr = self.softmax_fn orelse return LlmKernelError.NotInitialized;

        const block_size: u32 = 256;
        var args = [_]?*anyopaque{
            @ptrCast(&device_ptr),
            @ptrCast(&size),
        };

        if (launch_fn(
            fn_ptr,
            1, // gridDimX - single block for now
            1,
            1, // gridDim
            block_size,
            1,
            1, // blockDim
            0, // sharedMemBytes
            stream,
            &args,
            null,
        ) != .success) {
            return LlmKernelError.KernelLaunchFailed;
        }
    }

    /// Launch RMSNorm kernel on GPU data.
    pub fn rmsnorm(
        self: *LlmKernelModule,
        x_ptr: u64,
        weight_ptr: u64,
        size: u32,
        eps: f32,
        stream: ?*anyopaque,
    ) !void {
        const launch_fn = self.cuda_fns.kernel.cuLaunchKernel orelse
            return LlmKernelError.CudaNotAvailable;
        const fn_ptr = self.rmsnorm_fn orelse return LlmKernelError.NotInitialized;

        const block_size: u32 = 256;
        var args = [_]?*anyopaque{
            @ptrCast(&x_ptr),
            @ptrCast(&weight_ptr),
            @ptrCast(&size),
            @ptrCast(&eps),
        };

        if (launch_fn(
            fn_ptr,
            1,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            &args,
            null,
        ) != .success) {
            return LlmKernelError.KernelLaunchFailed;
        }
    }

    /// Launch SiLU activation kernel on GPU data.
    pub fn silu(self: *LlmKernelModule, device_ptr: u64, n: u32, stream: ?*anyopaque) !void {
        const launch_fn = self.cuda_fns.kernel.cuLaunchKernel orelse
            return LlmKernelError.CudaNotAvailable;
        const fn_ptr = self.silu_fn orelse return LlmKernelError.NotInitialized;

        const block_size: u32 = 256;
        const grid_size: u32 = (n + block_size - 1) / block_size;

        var args = [_]?*anyopaque{
            @ptrCast(&device_ptr),
            @ptrCast(&n),
        };

        if (launch_fn(
            fn_ptr,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            &args,
            null,
        ) != .success) {
            return LlmKernelError.KernelLaunchFailed;
        }
    }

    /// Launch GELU activation kernel on GPU data.
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(self: *LlmKernelModule, device_ptr: u64, n: u32, stream: ?*anyopaque) !void {
        const launch_fn = self.cuda_fns.kernel.cuLaunchKernel orelse
            return LlmKernelError.CudaNotAvailable;
        const fn_ptr = self.gelu_fn orelse return LlmKernelError.NotInitialized;

        const block_size: u32 = 256;
        const grid_size: u32 = (n + block_size - 1) / block_size;

        var args = [_]?*anyopaque{
            @ptrCast(&device_ptr),
            @ptrCast(&n),
        };

        if (launch_fn(
            fn_ptr,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            &args,
            null,
        ) != .success) {
            return LlmKernelError.KernelLaunchFailed;
        }
    }

    /// Launch element-wise multiply kernel.
    pub fn elementwiseMul(
        self: *LlmKernelModule,
        a_ptr: u64,
        b_ptr: u64,
        n: u32,
        stream: ?*anyopaque,
    ) !void {
        const launch_fn = self.cuda_fns.kernel.cuLaunchKernel orelse
            return LlmKernelError.CudaNotAvailable;
        const fn_ptr = self.elementwise_mul_fn orelse return LlmKernelError.NotInitialized;

        const block_size: u32 = 256;
        const grid_size: u32 = (n + block_size - 1) / block_size;

        var args = [_]?*anyopaque{
            @ptrCast(&a_ptr),
            @ptrCast(&b_ptr),
            @ptrCast(&n),
        };

        if (launch_fn(
            fn_ptr,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            &args,
            null,
        ) != .success) {
            return LlmKernelError.KernelLaunchFailed;
        }
    }

    /// Launch element-wise add kernel.
    pub fn elementwiseAdd(
        self: *LlmKernelModule,
        a_ptr: u64,
        b_ptr: u64,
        n: u32,
        stream: ?*anyopaque,
    ) !void {
        const launch_fn = self.cuda_fns.kernel.cuLaunchKernel orelse
            return LlmKernelError.CudaNotAvailable;
        const fn_ptr = self.elementwise_add_fn orelse return LlmKernelError.NotInitialized;

        const block_size: u32 = 256;
        const grid_size: u32 = (n + block_size - 1) / block_size;

        var args = [_]?*anyopaque{
            @ptrCast(&a_ptr),
            @ptrCast(&b_ptr),
            @ptrCast(&n),
        };

        if (launch_fn(
            fn_ptr,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            &args,
            null,
        ) != .success) {
            return LlmKernelError.KernelLaunchFailed;
        }
    }

    /// Launch scale kernel.
    pub fn scale(
        self: *LlmKernelModule,
        x_ptr: u64,
        scale_val: f32,
        n: u32,
        stream: ?*anyopaque,
    ) !void {
        const launch_fn = self.cuda_fns.kernel.cuLaunchKernel orelse
            return LlmKernelError.CudaNotAvailable;
        const fn_ptr = self.scale_fn orelse return LlmKernelError.NotInitialized;

        const block_size: u32 = 256;
        const grid_size: u32 = (n + block_size - 1) / block_size;

        var args = [_]?*anyopaque{
            @ptrCast(&x_ptr),
            @ptrCast(&scale_val),
            @ptrCast(&n),
        };

        if (launch_fn(
            fn_ptr,
            grid_size,
            1,
            1,
            block_size,
            1,
            1,
            0,
            stream,
            &args,
            null,
        ) != .success) {
            return LlmKernelError.KernelLaunchFailed;
        }
    }

    /// Launch fused attention kernel.
    /// Computes Q*K^T, softmax, and @V in a single GPU pass.
    ///
    /// Parameters:
    /// - q_ptr: Device pointer to Q matrix [seq_len, head_dim]
    /// - k_ptr: Device pointer to K matrix [kv_len, head_dim]
    /// - v_ptr: Device pointer to V matrix [kv_len, head_dim]
    /// - output_ptr: Device pointer to output [seq_len, head_dim]
    /// - seq_len: Number of query positions
    /// - kv_len: Number of key/value positions
    /// - head_dim: Dimension per head
    /// - scale_val: Scaling factor (typically 1/sqrt(head_dim))
    /// - causal: Whether to apply causal masking (1 = yes, 0 = no)
    pub fn fusedAttention(
        self: *LlmKernelModule,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        output_ptr: u64,
        seq_len: u32,
        kv_len: u32,
        head_dim: u32,
        scale_val: f32,
        causal: bool,
        stream: ?*anyopaque,
    ) !void {
        const launch_fn = self.cuda_fns.kernel.cuLaunchKernel orelse
            return LlmKernelError.CudaNotAvailable;
        const fn_ptr = self.fused_attention_fn orelse return LlmKernelError.NotInitialized;

        const block_size: u32 = 256;
        const causal_int: i32 = if (causal) 1 else 0;

        // Shared memory: scores[kv_len] + shared_max[block_size] + shared_sum[block_size]
        const shared_mem_bytes: u32 = (@as(u32, kv_len) + 2 * block_size) * @sizeOf(f32);

        var args = [_]?*anyopaque{
            @ptrCast(&q_ptr),
            @ptrCast(&k_ptr),
            @ptrCast(&v_ptr),
            @ptrCast(&output_ptr),
            @ptrCast(&seq_len),
            @ptrCast(&kv_len),
            @ptrCast(&head_dim),
            @ptrCast(&scale_val),
            @ptrCast(&causal_int),
        };

        if (launch_fn(
            fn_ptr,
            seq_len, // One block per query position
            1,
            1,
            block_size,
            1,
            1,
            shared_mem_bytes,
            stream,
            &args,
            null,
        ) != .success) {
            return LlmKernelError.KernelLaunchFailed;
        }
    }

    /// Launch tiled fused attention kernel (Flash Attention style).
    /// More memory-efficient for long sequences by processing K,V in tiles.
    ///
    /// Parameters:
    /// - q_ptr: Device pointer to Q matrix [seq_len, head_dim]
    /// - k_ptr: Device pointer to K matrix [kv_len, head_dim]
    /// - v_ptr: Device pointer to V matrix [kv_len, head_dim]
    /// - output_ptr: Device pointer to output [seq_len, head_dim]
    /// - seq_len: Number of query positions
    /// - kv_len: Number of key/value positions
    /// - head_dim: Dimension per head
    /// - scale_val: Scaling factor (typically 1/sqrt(head_dim))
    /// - causal: Whether to apply causal masking
    /// - block_kv: Tile size for K,V processing (default 64)
    pub fn fusedAttentionTiled(
        self: *LlmKernelModule,
        q_ptr: u64,
        k_ptr: u64,
        v_ptr: u64,
        output_ptr: u64,
        seq_len: u32,
        kv_len: u32,
        head_dim: u32,
        scale_val: f32,
        causal: bool,
        block_kv: u32,
        stream: ?*anyopaque,
    ) !void {
        const launch_fn = self.cuda_fns.kernel.cuLaunchKernel orelse
            return LlmKernelError.CudaNotAvailable;
        const fn_ptr = self.fused_attention_tiled_fn orelse return LlmKernelError.NotInitialized;

        const block_size: u32 = 256;
        const causal_int: i32 = if (causal) 1 else 0;

        // Shared memory: tile_scores[block_kv] + tile_k[block_kv * head_dim] +
        //                tile_v[block_kv * head_dim] + reduce_buf[block_size * 2]
        const shared_mem_bytes: u32 = (block_kv + 2 * block_kv * head_dim + 2 * block_size) * @sizeOf(f32);

        var args = [_]?*anyopaque{
            @ptrCast(&q_ptr),
            @ptrCast(&k_ptr),
            @ptrCast(&v_ptr),
            @ptrCast(&output_ptr),
            @ptrCast(&seq_len),
            @ptrCast(&kv_len),
            @ptrCast(&head_dim),
            @ptrCast(&scale_val),
            @ptrCast(&causal_int),
            @ptrCast(&block_kv),
        };

        if (launch_fn(
            fn_ptr,
            seq_len, // One block per query position
            1,
            1,
            block_size,
            1,
            1,
            shared_mem_bytes,
            stream,
            &args,
            null,
        ) != .success) {
            return LlmKernelError.KernelLaunchFailed;
        }
    }
};

/// Check if LLM kernels are available (CUDA + NVRTC).
pub fn isAvailable() bool {
    if (!loader.isAvailable()) return false;
    // Try to initialize NVRTC
    nvrtc.init() catch return false;
    return true;
}

test "llm kernels availability check" {
    // Just check that availability check doesn't crash
    _ = isAvailable();
}
