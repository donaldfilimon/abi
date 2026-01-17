//! GPU-accelerated LLM operations.
//!
//! Provides unified GPU inference path for LLM operations with automatic
//! fallback to CPU when GPU is unavailable.

const std = @import("std");
const build_options = @import("build_options");
const matmul = @import("matmul.zig");
const attention = @import("attention.zig");
const rmsnorm = @import("rmsnorm.zig");
const activations = @import("activations.zig");

// GPU backend detection
const backend_mod = if (build_options.enable_gpu)
    @import("../../../../compute/gpu/backend.zig")
else
    struct {
        pub fn summary() Summary {
            return .{
                .module_enabled = false,
                .enabled_backend_count = 0,
                .available_backend_count = 0,
                .device_count = 0,
                .emulated_devices = 0,
            };
        }

        pub const Summary = struct {
            module_enabled: bool,
            enabled_backend_count: usize,
            available_backend_count: usize,
            device_count: usize,
            emulated_devices: usize,
        };
    };

/// GPU operation context for LLM inference.
pub const GpuOpsContext = struct {
    allocator: std.mem.Allocator,
    gpu_available: bool,
    device_id: u32,
    /// Scratch buffer for GPU operations
    scratch_buffer: ?[]f32,
    scratch_size: usize,

    pub fn init(allocator: std.mem.Allocator) GpuOpsContext {
        const gpu_available = build_options.enable_gpu and checkGpuAvailability();
        return .{
            .allocator = allocator,
            .gpu_available = gpu_available,
            .device_id = 0,
            .scratch_buffer = null,
            .scratch_size = 0,
        };
    }

    pub fn deinit(self: *GpuOpsContext) void {
        if (self.scratch_buffer) |buf| {
            self.allocator.free(buf);
        }
        self.* = undefined;
    }

    /// Ensure scratch buffer is at least the given size.
    pub fn ensureScratchBuffer(self: *GpuOpsContext, size: usize) ![]f32 {
        if (self.scratch_buffer == null or self.scratch_size < size) {
            if (self.scratch_buffer) |buf| {
                self.allocator.free(buf);
            }
            self.scratch_buffer = try self.allocator.alloc(f32, size);
            self.scratch_size = size;
        }
        return self.scratch_buffer.?;
    }

    /// Check if GPU operations are available.
    pub fn isGpuAvailable(self: *const GpuOpsContext) bool {
        return self.gpu_available;
    }

    /// Matrix multiplication with GPU acceleration.
    pub fn matrixMultiply(
        self: *GpuOpsContext,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: u32,
        k: u32,
        n: u32,
    ) void {
        if (self.gpu_available) {
            // Try GPU path
            self.gpuMatmul(a, b, c, m, k, n) catch {
                // Fallback to CPU
                matmul.matrixMultiply(a, b, c, m, k, n);
            };
        } else {
            // CPU path
            matmul.matrixMultiply(a, b, c, m, k, n);
        }
    }

    /// Batched matrix multiplication with GPU acceleration.
    pub fn batchedMatmul(
        self: *GpuOpsContext,
        a: []const f32,
        b: []const f32,
        c: []f32,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    ) void {
        if (self.gpu_available) {
            self.gpuBatchedMatmul(a, b, c, batch, m, k, n) catch {
                // Fallback: iterate batches on CPU
                const a_stride = @as(usize, m) * k;
                const b_stride = @as(usize, k) * n;
                const c_stride = @as(usize, m) * n;

                for (0..batch) |i| {
                    const a_offset = i * a_stride;
                    const b_offset = i * b_stride;
                    const c_offset = i * c_stride;
                    matmul.matrixMultiply(
                        a[a_offset..][0..a_stride],
                        b[b_offset..][0..b_stride],
                        c[c_offset..][0..c_stride],
                        m,
                        k,
                        n,
                    );
                }
            };
        } else {
            // CPU batched matmul
            const a_stride = @as(usize, m) * k;
            const b_stride = @as(usize, k) * n;
            const c_stride = @as(usize, m) * n;

            for (0..batch) |i| {
                const a_offset = i * a_stride;
                const b_offset = i * b_stride;
                const c_offset = i * c_stride;
                matmul.matrixMultiply(
                    a[a_offset..][0..a_stride],
                    b[b_offset..][0..b_stride],
                    c[c_offset..][0..c_stride],
                    m,
                    k,
                    n,
                );
            }
        }
    }

    /// Multi-head attention with GPU acceleration.
    pub fn multiHeadAttention(
        self: *GpuOpsContext,
        q: []const f32,
        k_cache: []const f32,
        v_cache: []const f32,
        output: []f32,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        kv_len: u32,
    ) !void {
        if (self.gpu_available) {
            self.gpuAttention(q, k_cache, v_cache, output, seq_len, n_heads, head_dim, kv_len) catch {
                // Fallback to CPU attention
                try attention.multiHeadAttention(
                    self.allocator,
                    q,
                    k_cache,
                    v_cache,
                    output,
                    seq_len,
                    n_heads,
                    head_dim,
                    kv_len,
                );
            };
        } else {
            try attention.multiHeadAttention(
                self.allocator,
                q,
                k_cache,
                v_cache,
                output,
                seq_len,
                n_heads,
                head_dim,
                kv_len,
            );
        }
    }

    /// RMS normalization with GPU acceleration.
    pub fn rmsNorm(
        self: *GpuOpsContext,
        x: []f32,
        weight: []const f32,
        eps: f32,
    ) void {
        if (self.gpu_available) {
            self.gpuRmsNorm(x, weight, eps) catch {
                rmsnorm.rmsNormInPlace(x, weight, eps);
            };
        } else {
            rmsnorm.rmsNormInPlace(x, weight, eps);
        }
    }

    /// Softmax with GPU acceleration.
    pub fn softmax(self: *GpuOpsContext, x: []f32) void {
        if (self.gpu_available) {
            self.gpuSoftmax(x) catch {
                activations.softmaxInPlace(x);
            };
        } else {
            activations.softmaxInPlace(x);
        }
    }

    /// SiLU activation with GPU acceleration.
    pub fn silu(self: *GpuOpsContext, x: []f32) void {
        if (self.gpu_available) {
            self.gpuSilu(x) catch {
                activations.siluInPlace(x);
            };
        } else {
            activations.siluInPlace(x);
        }
    }

    /// Element-wise multiply with GPU acceleration.
    pub fn elementwiseMul(self: *GpuOpsContext, a: []f32, b: []const f32) void {
        if (self.gpu_available) {
            self.gpuElementwiseMul(a, b) catch {
                for (a, b) |*av, bv| {
                    av.* *= bv;
                }
            };
        } else {
            for (a, b) |*av, bv| {
                av.* *= bv;
            }
        }
    }

    /// Vector add with GPU acceleration.
    pub fn vectorAdd(self: *GpuOpsContext, a: []f32, b: []const f32) void {
        if (self.gpu_available) {
            self.gpuVectorAdd(a, b) catch {
                for (a, b) |*av, bv| {
                    av.* += bv;
                }
            };
        } else {
            for (a, b) |*av, bv| {
                av.* += bv;
            }
        }
    }

    // GPU implementation stubs - to be connected to actual GPU backends
    fn gpuMatmul(
        self: *GpuOpsContext,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: u32,
        k: u32,
        n: u32,
    ) !void {
        _ = self;
        // Placeholder: Would dispatch to GPU backend
        // For now, use SIMD-optimized CPU path
        matmul.matrixMultiply(a, b, c, m, k, n);
    }

    fn gpuBatchedMatmul(
        self: *GpuOpsContext,
        a: []const f32,
        b: []const f32,
        c: []f32,
        batch: u32,
        m: u32,
        k: u32,
        n: u32,
    ) !void {
        _ = self;
        _ = batch;
        // Placeholder: Would dispatch to GPU backend
        matmul.matrixMultiply(a, b, c, m, k, n);
    }

    fn gpuAttention(
        self: *GpuOpsContext,
        q: []const f32,
        k_cache: []const f32,
        v_cache: []const f32,
        output: []f32,
        seq_len: u32,
        n_heads: u32,
        head_dim: u32,
        kv_len: u32,
    ) !void {
        // Placeholder: Would dispatch to GPU backend
        try attention.multiHeadAttention(
            self.allocator,
            q,
            k_cache,
            v_cache,
            output,
            seq_len,
            n_heads,
            head_dim,
            kv_len,
        );
    }

    fn gpuRmsNorm(
        self: *GpuOpsContext,
        x: []f32,
        weight: []const f32,
        eps: f32,
    ) !void {
        _ = self;
        rmsnorm.rmsNormInPlace(x, weight, eps);
    }

    fn gpuSoftmax(self: *GpuOpsContext, x: []f32) !void {
        _ = self;
        activations.softmaxInPlace(x);
    }

    fn gpuSilu(self: *GpuOpsContext, x: []f32) !void {
        _ = self;
        activations.siluInPlace(x);
    }

    fn gpuElementwiseMul(self: *GpuOpsContext, a: []f32, b: []const f32) !void {
        _ = self;
        for (a, b) |*av, bv| {
            av.* *= bv;
        }
    }

    fn gpuVectorAdd(self: *GpuOpsContext, a: []f32, b: []const f32) !void {
        _ = self;
        for (a, b) |*av, bv| {
            av.* += bv;
        }
    }
};

/// Check if GPU is available at runtime.
/// Uses the GPU backend detection infrastructure to probe for real devices.
fn checkGpuAvailability() bool {
    if (!build_options.enable_gpu) return false;

    // Use the backend summary to check for available devices
    const gpu_summary = backend_mod.summary();

    // GPU is available if:
    // 1. The module is enabled
    // 2. At least one backend is available
    // 3. There are real (non-emulated) devices available
    if (!gpu_summary.module_enabled) return false;
    if (gpu_summary.available_backend_count == 0) return false;
    if (gpu_summary.device_count == 0) return false;

    // Prefer real hardware over emulated devices
    // Return true if there are any non-emulated devices
    return gpu_summary.device_count > gpu_summary.emulated_devices;
}

/// GPU execution statistics.
pub const GpuStats = struct {
    /// Total GPU operations executed
    total_ops: u64 = 0,
    /// Total GPU time (nanoseconds)
    total_time_ns: u64 = 0,
    /// Operations that fell back to CPU
    fallback_ops: u64 = 0,
    /// Peak GPU memory used (bytes)
    peak_memory_bytes: u64 = 0,

    pub fn addOp(self: *GpuStats, time_ns: u64, used_gpu: bool) void {
        self.total_ops += 1;
        self.total_time_ns += time_ns;
        if (!used_gpu) {
            self.fallback_ops += 1;
        }
    }

    pub fn gpuUtilization(self: GpuStats) f64 {
        if (self.total_ops == 0) return 0;
        return 1.0 - (@as(f64, @floatFromInt(self.fallback_ops)) / @as(f64, @floatFromInt(self.total_ops)));
    }
};

/// Create a GPU operations context for LLM inference.
pub fn createContext(allocator: std.mem.Allocator) GpuOpsContext {
    return GpuOpsContext.init(allocator);
}

test "gpu ops context init" {
    const allocator = std.testing.allocator;

    var ctx = GpuOpsContext.init(allocator);
    defer ctx.deinit();

    // Should be able to run matmul (falls back to CPU)
    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };

    ctx.matrixMultiply(&a, &b, &c, 2, 2, 2);
}

test "gpu stats tracking" {
    var stats = GpuStats{};

    stats.addOp(1000, true);
    stats.addOp(2000, false);
    stats.addOp(3000, true);

    try std.testing.expectEqual(@as(u64, 3), stats.total_ops);
    try std.testing.expectEqual(@as(u64, 1), stats.fallback_ops);
    try std.testing.expectApproxEqAbs(@as(f64, 0.666), stats.gpuUtilization(), 0.01);
}
