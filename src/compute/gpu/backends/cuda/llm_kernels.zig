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

/// Compiled CUDA module containing LLM kernels.
pub const LlmKernelModule = struct {
    allocator: std.mem.Allocator,
    module: ?*anyopaque,
    softmax_fn: ?*anyopaque,
    rmsnorm_fn: ?*anyopaque,
    silu_fn: ?*anyopaque,
    elementwise_mul_fn: ?*anyopaque,
    elementwise_add_fn: ?*anyopaque,
    scale_fn: ?*anyopaque,
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
            SILU_KERNEL ++ "\n" ++ ELEMENTWISE_MUL_KERNEL ++ "\n" ++
            ELEMENTWISE_ADD_KERNEL ++ "\n" ++ SCALE_KERNEL;

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
        var elementwise_mul_fn: ?*anyopaque = null;
        var elementwise_add_fn: ?*anyopaque = null;
        var scale_fn: ?*anyopaque = null;

        if (get_fn(&softmax_fn, module, "softmax_kernel") != .success) {
            std.log.warn("Failed to get softmax_kernel function", .{});
        }
        if (get_fn(&rmsnorm_fn, module, "rmsnorm_kernel") != .success) {
            std.log.warn("Failed to get rmsnorm_kernel function", .{});
        }
        if (get_fn(&silu_fn, module, "silu_kernel") != .success) {
            std.log.warn("Failed to get silu_kernel function", .{});
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

        return .{
            .allocator = allocator,
            .module = module,
            .softmax_fn = softmax_fn,
            .rmsnorm_fn = rmsnorm_fn,
            .silu_fn = silu_fn,
            .elementwise_mul_fn = elementwise_mul_fn,
            .elementwise_add_fn = elementwise_add_fn,
            .scale_fn = scale_fn,
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
