//! GPU-accelerated backward operations for LLM training.
//!
//! Provides cuBLAS-accelerated implementations of backward pass operations
//! for matrix multiplication and attention. Falls back to CPU implementations
//! when GPU is unavailable.
//!
//! GPU access is provided through the centralized ai_ops interface, which
//! handles compile-time gating and provides stub implementations when GPU
//! is disabled.

const std = @import("std");
const build_options = @import("build_options");
const matmul_backward = @import("matmul_backward.zig");
const attention_backward = @import("attention_backward.zig");

// Centralized GPU interface - handles compile-time gating and stubs
const ai_ops = @import("../../../gpu/ai_ops.zig");

// Re-export GPU modules from ai_ops (stubs provided when GPU disabled)
const cublas = ai_ops.cublas;
const cuda_memory = ai_ops.memory;

/// GPU backward context for training operations.
pub const GpuBackwardContext = struct {
    allocator: std.mem.Allocator,
    cublas_ctx: ?cublas.CublasContext,
    gpu_available: bool,
    /// Statistics
    gpu_ops: u64 = 0,
    cpu_fallbacks: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) GpuBackwardContext {
        var ctx: ?cublas.CublasContext = null;
        var gpu_available = false;

        if (build_options.enable_gpu and cublas.isAvailable()) {
            // Initialize CUDA memory subsystem
            cuda_memory.init(allocator) catch {
                return .{
                    .allocator = allocator,
                    .cublas_ctx = null,
                    .gpu_available = false,
                };
            };

            ctx = cublas.CublasContext.init() catch null;
            gpu_available = ctx != null;
            if (gpu_available) {
                std.log.info("GPU backward ops initialized with cuBLAS", .{});
            }
        }

        return .{
            .allocator = allocator,
            .cublas_ctx = ctx,
            .gpu_available = gpu_available,
        };
    }

    pub fn deinit(self: *GpuBackwardContext) void {
        if (self.cublas_ctx) |*ctx| {
            ctx.deinit();
        }
        self.* = undefined;
    }

    /// GPU-accelerated matrix multiplication backward.
    /// For forward: C = A @ B
    /// Computes: dA = dC @ B^T, dB = A^T @ dC
    pub fn matmulBackward(
        self: *GpuBackwardContext,
        dC: []const f32,
        A: []const f32,
        B: []const f32,
        dA: []f32,
        dB: []f32,
        m: u32,
        k: u32,
        n: u32,
    ) void {
        if (self.gpu_available and self.cublas_ctx != null) {
            self.gpuMatmulBackward(dC, A, B, dA, dB, m, k, n) catch {
                // Fallback to CPU
                self.cpu_fallbacks += 1;
                matmul_backward.matmulBackward(dC, A, B, dA, dB, m, k, n);
                return;
            };
            self.gpu_ops += 1;
        } else {
            self.cpu_fallbacks += 1;
            matmul_backward.matmulBackward(dC, A, B, dA, dB, m, k, n);
        }
    }

    /// GPU-accelerated batched matrix multiplication backward.
    pub fn batchedMatmulBackward(
        self: *GpuBackwardContext,
        dC: []const f32,
        A: []const f32,
        B: []const f32,
        dA: []f32,
        dB: []f32,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    ) void {
        if (self.gpu_available and self.cublas_ctx != null) {
            self.gpuBatchedMatmulBackward(dC, A, B, dA, dB, batch, m, k, n) catch {
                self.cpu_fallbacks += 1;
                matmul_backward.batchedMatmulBackward(dC, A, B, dA, dB, batch, m, k, n);
                return;
            };
            self.gpu_ops += 1;
        } else {
            self.cpu_fallbacks += 1;
            matmul_backward.batchedMatmulBackward(dC, A, B, dA, dB, batch, m, k, n);
        }
    }

    /// GPU implementation of matmul backward using cuBLAS.
    fn gpuMatmulBackward(
        self: *GpuBackwardContext,
        dC: []const f32,
        A: []const f32,
        B: []const f32,
        dA: []f32,
        dB: []f32,
        m: u32,
        k: u32,
        n: u32,
    ) !void {
        var ctx = self.cublas_ctx orelse return error.NotAvailable;

        const dC_size = @as(usize, m) * n * @sizeOf(f32);
        const A_size = @as(usize, m) * k * @sizeOf(f32);
        const B_size = @as(usize, k) * n * @sizeOf(f32);

        // Allocate device memory
        var dC_dev = try cuda_memory.DeviceMemory.init(self.allocator, dC_size);
        defer dC_dev.deinit();
        var A_dev = try cuda_memory.DeviceMemory.init(self.allocator, A_size);
        defer A_dev.deinit();
        var B_dev = try cuda_memory.DeviceMemory.init(self.allocator, B_size);
        defer B_dev.deinit();
        var dA_dev = try cuda_memory.DeviceMemory.init(self.allocator, A_size);
        defer dA_dev.deinit();
        var dB_dev = try cuda_memory.DeviceMemory.init(self.allocator, B_size);
        defer dB_dev.deinit();

        // Copy inputs to device
        try cuda_memory.memcpyHostToDevice(dC_dev.ptr.?, @ptrCast(dC.ptr), dC_size);
        try cuda_memory.memcpyHostToDevice(A_dev.ptr.?, @ptrCast(A.ptr), A_size);
        try cuda_memory.memcpyHostToDevice(B_dev.ptr.?, @ptrCast(B.ptr), B_size);

        // Copy existing dA, dB to device (for accumulation)
        try cuda_memory.memcpyHostToDevice(dA_dev.ptr.?, @ptrCast(dA.ptr), A_size);
        try cuda_memory.memcpyHostToDevice(dB_dev.ptr.?, @ptrCast(dB.ptr), B_size);

        // dA = dC @ B^T (accumulate with beta=1)
        // cuBLAS: C = alpha*op(A)*op(B) + beta*C
        // We want dA += dC @ B^T
        // In row-major: dA[M,K] = dC[M,N] @ B^T[N,K]
        // cuBLAS col-major trick: dA^T = B @ dC^T
        // sgemm(N, T, K, M, N, 1, B, N, dC, N, 1, dA, K)
        try ctx.sgemm(
            .no_trans,
            .trans,
            @intCast(k),
            @intCast(m),
            @intCast(n),
            1.0,
            @ptrCast(B_dev.ptr.?),
            @intCast(n),
            @ptrCast(dC_dev.ptr.?),
            @intCast(n),
            1.0, // beta=1 for accumulation
            @ptrCast(dA_dev.ptr.?),
            @intCast(k),
        );

        // dB = A^T @ dC (accumulate with beta=1)
        // We want dB += A^T @ dC
        // In row-major: dB[K,N] = A^T[K,M] @ dC[M,N]
        // cuBLAS col-major trick: dB^T = dC^T @ A
        // sgemm(N, T, N, K, M, 1, dC, N, A, K, 1, dB, N)
        try ctx.sgemm(
            .no_trans,
            .trans,
            @intCast(n),
            @intCast(k),
            @intCast(m),
            1.0,
            @ptrCast(dC_dev.ptr.?),
            @intCast(n),
            @ptrCast(A_dev.ptr.?),
            @intCast(k),
            1.0, // beta=1 for accumulation
            @ptrCast(dB_dev.ptr.?),
            @intCast(n),
        );

        // Copy results back
        try cuda_memory.memcpyDeviceToHost(@ptrCast(dA.ptr), dA_dev.ptr.?, A_size);
        try cuda_memory.memcpyDeviceToHost(@ptrCast(dB.ptr), dB_dev.ptr.?, B_size);
    }

    /// GPU implementation of batched matmul backward.
    fn gpuBatchedMatmulBackward(
        self: *GpuBackwardContext,
        dC: []const f32,
        A: []const f32,
        B: []const f32,
        dA: []f32,
        dB: []f32,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    ) !void {
        var ctx = self.cublas_ctx orelse return error.NotAvailable;

        const batch_usize = @as(usize, batch);
        if (batch_usize == 0) return;

        const a_stride_elems: i64 = @as(i64, m) * k;
        const b_stride_elems: i64 = @as(i64, k) * n;
        const c_stride_elems: i64 = @as(i64, m) * n;

        const a_size = batch_usize * @as(usize, @intCast(a_stride_elems)) * @sizeOf(f32);
        const b_size = batch_usize * @as(usize, @intCast(b_stride_elems)) * @sizeOf(f32);
        const c_size = batch_usize * @as(usize, @intCast(c_stride_elems)) * @sizeOf(f32);

        // Allocate device memory
        var dC_dev = try cuda_memory.DeviceMemory.init(self.allocator, c_size);
        defer dC_dev.deinit();
        var A_dev = try cuda_memory.DeviceMemory.init(self.allocator, a_size);
        defer A_dev.deinit();
        var B_dev = try cuda_memory.DeviceMemory.init(self.allocator, b_size);
        defer B_dev.deinit();
        var dA_dev = try cuda_memory.DeviceMemory.init(self.allocator, a_size);
        defer dA_dev.deinit();
        var dB_dev = try cuda_memory.DeviceMemory.init(self.allocator, b_size);
        defer dB_dev.deinit();

        // Copy inputs to device
        try cuda_memory.memcpyHostToDevice(dC_dev.ptr.?, @ptrCast(dC.ptr), c_size);
        try cuda_memory.memcpyHostToDevice(A_dev.ptr.?, @ptrCast(A.ptr), a_size);
        try cuda_memory.memcpyHostToDevice(B_dev.ptr.?, @ptrCast(B.ptr), b_size);

        // Copy existing dA, dB to device (for accumulation)
        try cuda_memory.memcpyHostToDevice(dA_dev.ptr.?, @ptrCast(dA.ptr), a_size);
        try cuda_memory.memcpyHostToDevice(dB_dev.ptr.?, @ptrCast(dB.ptr), b_size);

        // dA = dC @ B^T (accumulate with beta=1)
        try ctx.sgemmStridedBatched(
            .no_trans,
            .trans,
            @intCast(k),
            @intCast(m),
            @intCast(n),
            1.0,
            @ptrCast(B_dev.ptr.?),
            @intCast(n),
            b_stride_elems,
            @ptrCast(dC_dev.ptr.?),
            @intCast(n),
            c_stride_elems,
            1.0,
            @ptrCast(dA_dev.ptr.?),
            @intCast(k),
            a_stride_elems,
            @intCast(batch),
        );

        // dB = A^T @ dC (accumulate with beta=1)
        try ctx.sgemmStridedBatched(
            .no_trans,
            .trans,
            @intCast(n),
            @intCast(k),
            @intCast(m),
            1.0,
            @ptrCast(dC_dev.ptr.?),
            @intCast(n),
            c_stride_elems,
            @ptrCast(A_dev.ptr.?),
            @intCast(k),
            a_stride_elems,
            1.0,
            @ptrCast(dB_dev.ptr.?),
            @intCast(n),
            b_stride_elems,
            @intCast(batch),
        );

        // Copy results back
        try cuda_memory.memcpyDeviceToHost(@ptrCast(dA.ptr), dA_dev.ptr.?, a_size);
        try cuda_memory.memcpyDeviceToHost(@ptrCast(dB.ptr), dB_dev.ptr.?, b_size);
    }

    /// Get GPU utilization statistics.
    pub fn stats(self: *const GpuBackwardContext) struct {
        gpu_ops: u64,
        cpu_fallbacks: u64,
        gpu_utilization: f64,
    } {
        const total = self.gpu_ops + self.cpu_fallbacks;
        const util = if (total > 0)
            @as(f64, @floatFromInt(self.gpu_ops)) / @as(f64, @floatFromInt(total))
        else
            0.0;
        return .{
            .gpu_ops = self.gpu_ops,
            .cpu_fallbacks = self.cpu_fallbacks,
            .gpu_utilization = util,
        };
    }
};

/// Create a GPU backward context for training.
pub fn createContext(allocator: std.mem.Allocator) GpuBackwardContext {
    return GpuBackwardContext.init(allocator);
}

test "gpu backward context init" {
    const allocator = std.testing.allocator;

    var ctx = GpuBackwardContext.init(allocator);
    defer ctx.deinit();

    // Test matmul backward (will use CPU fallback if no GPU)
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var dC = [_]f32{ 1, 0, 0, 1 };
    var dA = [_]f32{ 0, 0, 0, 0 };
    var dB = [_]f32{ 0, 0, 0, 0 };

    ctx.matmulBackward(&dC, &A, &B, &dA, &dB, 2, 2, 2);

    // Verify results match CPU version
    try std.testing.expectApproxEqAbs(@as(f32, 5), dA[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7), dA[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6), dA[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8), dA[3], 0.001);
}

test "gpu backward stats" {
    const allocator = std.testing.allocator;

    var ctx = GpuBackwardContext.init(allocator);
    defer ctx.deinit();

    // Run a few operations
    const A = [_]f32{ 1, 2, 3, 4 };
    const B = [_]f32{ 5, 6, 7, 8 };
    var dC = [_]f32{ 1, 1, 1, 1 };
    var dA = [_]f32{ 0, 0, 0, 0 };
    var dB = [_]f32{ 0, 0, 0, 0 };

    ctx.matmulBackward(&dC, &A, &B, &dA, &dB, 2, 2, 2);

    const s = ctx.stats();
    // Either used GPU or fell back to CPU
    try std.testing.expect(s.gpu_ops + s.cpu_fallbacks >= 1);
}

test {
    std.testing.refAllDecls(@This());
}
