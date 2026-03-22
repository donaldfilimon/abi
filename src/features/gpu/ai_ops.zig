//! AI GPU Operations Interface
//!
//! Backend-agnostic interface for AI/ML workloads providing BLAS operations,
//! activation functions, and memory management. Enables AI modules to use GPU
//! acceleration without direct coupling to specific backends (CUDA, Vulkan, etc).
//!
//! Design:
//! - VTable pattern matching existing gpu/interface.zig
//! - Compile-time gating via build_options
//! - CpuFallbackAiOps for when GPU is disabled (computes on CPU)
//! - StubAiOps is an alias for CpuFallbackAiOps (backward compat)
//! - Error handling via error union

const std = @import("std");
const build_options = @import("build_options");
const backend_shared = @import("backends/shared.zig");

// =============================================================================
// Error Types
// =============================================================================

/// Errors for AI GPU operations.
pub const AiOpsError = error{
    /// GPU backend not available or not initialized
    NotAvailable,
    /// Memory allocation failed
    OutOfMemory,
    /// Host-device transfer failed
    TransferFailed,
    /// Kernel execution failed
    KernelFailed,
    /// Invalid parameter or configuration
    InvalidParameter,
    /// Operation not supported by this backend
    NotSupported,
};

// =============================================================================
// Device Buffer
// =============================================================================

/// Managed GPU device memory buffer.
/// Automatically freed when deinit() is called.
pub const DeviceBuffer = struct {
    /// Raw device pointer (backend-specific)
    ptr: ?*anyopaque,
    /// Size in bytes
    size: usize,
    /// Host allocator for metadata
    allocator: std.mem.Allocator,
    /// Reference to owning AiOps for cleanup
    ops: *const AiOps,

    /// Free device memory.
    pub fn deinit(self: *DeviceBuffer) void {
        if (self.ptr) |p| {
            self.ops.freeDevice(p);
        }
        self.* = undefined;
    }
};

// =============================================================================
// Matrix Operation Transpose Flag
// =============================================================================

/// Transpose flag for BLAS operations.
pub const Transpose = enum {
    no_trans,
    trans,

    pub fn toBool(self: Transpose) bool {
        return self == .trans;
    }
};

// =============================================================================
// AI Operations Interface
// =============================================================================

/// Backend-agnostic interface for AI GPU operations.
///
/// Provides:
/// - BLAS: sgemm, sgemmStridedBatched for matrix operations
/// - Activations: softmax, rmsnorm, silu, gelu, scale, elementwiseMul, elementwiseAdd
/// - Memory: allocDevice, copyToDevice, copyFromDevice, freeDevice
///
/// Example usage:
/// ```zig
/// var ops = if (build_options.feat_gpu)
///     try cuda_ai_ops.CudaAiOps.init(allocator)
/// else
///     CpuFallbackAiOps.init();
/// defer ops.deinit();
///
/// if (ops.isAvailable()) {
///     var buf = try ops.allocDevice(allocator, 1024);
///     defer buf.deinit();
///     try ops.copyToDevice(buf.ptr.?, host_data);
///     try ops.softmax(buf.ptr.?, 256, null);
/// }
/// ```
pub const AiOps = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        // =====================================================================
        // BLAS Operations
        // =====================================================================

        /// Single-precision general matrix multiply: C = alpha * op(A) @ op(B) + beta * C
        /// Uses row-major layout.
        sgemm: *const fn (
            ctx: *anyopaque,
            trans_a: Transpose,
            trans_b: Transpose,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const anyopaque,
            lda: i32,
            b: *const anyopaque,
            ldb: i32,
            beta: f32,
            c: *anyopaque,
            ldc: i32,
        ) AiOpsError!void,

        /// Batched strided GEMM for attention: C[i] = alpha * A[i] @ B[i] + beta * C[i]
        sgemmStridedBatched: *const fn (
            ctx: *anyopaque,
            trans_a: Transpose,
            trans_b: Transpose,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const anyopaque,
            lda: i32,
            stride_a: i64,
            b: *const anyopaque,
            ldb: i32,
            stride_b: i64,
            beta: f32,
            c: *anyopaque,
            ldc: i32,
            stride_c: i64,
            batch_count: i32,
        ) AiOpsError!void,

        // =====================================================================
        // Activation Operations
        // =====================================================================

        /// In-place softmax: x = softmax(x)
        softmax: *const fn (
            ctx: *anyopaque,
            data: *anyopaque,
            len: u32,
            stream: ?*anyopaque,
        ) AiOpsError!void,

        /// In-place RMS normalization: x = x / rms(x) * weight
        rmsnorm: *const fn (
            ctx: *anyopaque,
            x: *anyopaque,
            weight: *const anyopaque,
            len: u32,
            eps: f32,
            stream: ?*anyopaque,
        ) AiOpsError!void,

        /// In-place SiLU activation: x = x * sigmoid(x)
        silu: *const fn (
            ctx: *anyopaque,
            data: *anyopaque,
            len: u32,
            stream: ?*anyopaque,
        ) AiOpsError!void,

        /// In-place GELU activation
        gelu: *const fn (
            ctx: *anyopaque,
            data: *anyopaque,
            len: u32,
            stream: ?*anyopaque,
        ) AiOpsError!void,

        /// In-place scale: x = x * scalar
        scale: *const fn (
            ctx: *anyopaque,
            data: *anyopaque,
            scalar: f32,
            len: u32,
            stream: ?*anyopaque,
        ) AiOpsError!void,

        /// In-place element-wise multiply: a = a * b
        elementwiseMul: *const fn (
            ctx: *anyopaque,
            a: *anyopaque,
            b: *const anyopaque,
            len: u32,
            stream: ?*anyopaque,
        ) AiOpsError!void,

        /// In-place element-wise add: a = a + b
        elementwiseAdd: *const fn (
            ctx: *anyopaque,
            a: *anyopaque,
            b: *const anyopaque,
            len: u32,
            stream: ?*anyopaque,
        ) AiOpsError!void,

        // =====================================================================
        // Memory Operations
        // =====================================================================

        /// Allocate device memory.
        allocDevice: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
            size: usize,
        ) AiOpsError!DeviceBuffer,

        /// Copy from host to device.
        copyToDevice: *const fn (
            ctx: *anyopaque,
            dst: *anyopaque,
            src: [*]const u8,
            len: usize,
        ) AiOpsError!void,

        /// Copy from device to host.
        copyFromDevice: *const fn (
            ctx: *anyopaque,
            dst: [*]u8,
            src: *const anyopaque,
            len: usize,
        ) AiOpsError!void,

        /// Free device memory.
        freeDevice: *const fn (
            ctx: *anyopaque,
            ptr: *anyopaque,
        ) void,

        // =====================================================================
        // Lifecycle
        // =====================================================================

        /// Check if GPU operations are available.
        isAvailable: *const fn (ctx: *anyopaque) bool,

        /// Clean up resources.
        deinit: *const fn (ctx: *anyopaque) void,
    };

    // =========================================================================
    // Wrapper Methods
    // =========================================================================

    /// Single-precision general matrix multiply.
    pub fn sgemm(
        self: AiOps,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const anyopaque,
        lda: i32,
        b: *const anyopaque,
        ldb: i32,
        beta: f32,
        c: *anyopaque,
        ldc: i32,
    ) AiOpsError!void {
        return self.vtable.sgemm(
            self.ptr,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            b,
            ldb,
            beta,
            c,
            ldc,
        );
    }

    /// Batched strided GEMM.
    pub fn sgemmStridedBatched(
        self: AiOps,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const anyopaque,
        lda: i32,
        stride_a: i64,
        b: *const anyopaque,
        ldb: i32,
        stride_b: i64,
        beta: f32,
        c: *anyopaque,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) AiOpsError!void {
        return self.vtable.sgemmStridedBatched(
            self.ptr,
            trans_a,
            trans_b,
            m,
            n,
            k,
            alpha,
            a,
            lda,
            stride_a,
            b,
            ldb,
            stride_b,
            beta,
            c,
            ldc,
            stride_c,
            batch_count,
        );
    }

    /// In-place softmax.
    pub fn softmax(self: AiOps, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
        return self.vtable.softmax(self.ptr, data, len, stream);
    }

    /// In-place RMS normalization.
    pub fn rmsnorm(
        self: AiOps,
        x: *anyopaque,
        weight: *const anyopaque,
        len: u32,
        eps: f32,
        stream: ?*anyopaque,
    ) AiOpsError!void {
        return self.vtable.rmsnorm(self.ptr, x, weight, len, eps, stream);
    }

    /// In-place SiLU activation.
    pub fn silu(self: AiOps, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
        return self.vtable.silu(self.ptr, data, len, stream);
    }

    /// In-place GELU activation.
    pub fn gelu(self: AiOps, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
        return self.vtable.gelu(self.ptr, data, len, stream);
    }

    /// In-place scale.
    pub fn scale(self: AiOps, data: *anyopaque, scalar: f32, len: u32, stream: ?*anyopaque) AiOpsError!void {
        return self.vtable.scale(self.ptr, data, scalar, len, stream);
    }

    /// In-place element-wise multiply.
    pub fn elementwiseMul(self: AiOps, a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
        return self.vtable.elementwiseMul(self.ptr, a, b, len, stream);
    }

    /// In-place element-wise add.
    pub fn elementwiseAdd(self: AiOps, a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
        return self.vtable.elementwiseAdd(self.ptr, a, b, len, stream);
    }

    /// Allocate device memory.
    pub fn allocDevice(self: AiOps, allocator: std.mem.Allocator, size: usize) AiOpsError!DeviceBuffer {
        return self.vtable.allocDevice(self.ptr, allocator, size);
    }

    /// Copy from host to device.
    pub fn copyToDevice(self: AiOps, dst: *anyopaque, src: [*]const u8, len: usize) AiOpsError!void {
        return self.vtable.copyToDevice(self.ptr, dst, src, len);
    }

    /// Copy from device to host.
    pub fn copyFromDevice(self: AiOps, dst: [*]u8, src: *const anyopaque, len: usize) AiOpsError!void {
        return self.vtable.copyFromDevice(self.ptr, dst, src, len);
    }

    /// Free device memory.
    pub fn freeDevice(self: AiOps, ptr: *anyopaque) void {
        self.vtable.freeDevice(self.ptr, ptr);
    }

    /// Check if GPU operations are available.
    pub fn isAvailable(self: AiOps) bool {
        return self.vtable.isAvailable(self.ptr);
    }

    /// Clean up resources.
    pub fn deinit(self: AiOps) void {
        self.vtable.deinit(self.ptr);
    }
};

// =============================================================================
// CPU Fallback Implementation
// =============================================================================

/// CPU fallback implementation when GPU is disabled.
/// Performs real computations on host memory using standard math.
/// Pointers are interpreted as f32 arrays laid out in row-major order.
pub const CpuFallbackAiOps = struct {
    /// Singleton instance data (no state needed for CPU fallback).
    const Instance = struct {};
    var instance: Instance = .{};

    /// Create a CPU fallback AiOps that computes on the host.
    pub fn init() AiOps {
        return .{
            .ptr = @ptrCast(&instance),
            .vtable = &vtable,
        };
    }

    const vtable = AiOps.VTable{
        .sgemm = cpuSgemm,
        .sgemmStridedBatched = cpuSgemmBatched,
        .softmax = cpuSoftmax,
        .rmsnorm = cpuRmsnorm,
        .silu = cpuSilu,
        .gelu = cpuGelu,
        .scale = cpuScale,
        .elementwiseMul = cpuElementwiseMul,
        .elementwiseAdd = cpuElementwiseAdd,
        .allocDevice = cpuAllocDevice,
        .copyToDevice = cpuCopyToDevice,
        .copyFromDevice = cpuCopyFromDevice,
        .freeDevice = cpuFreeDevice,
        .isAvailable = cpuIsAvailable,
        .deinit = cpuDeinit,
    };

    // =========================================================================
    // BLAS Operations
    // =========================================================================

    /// CPU sgemm: C = alpha * op(A) * op(B) + beta * C, row-major layout.
    /// op(X) is X if no_trans, X^T if trans.
    fn cpuSgemm(
        _: *anyopaque,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: *const anyopaque,
        lda: i32,
        b_ptr: *const anyopaque,
        ldb: i32,
        beta: f32,
        c_ptr: *anyopaque,
        ldc: i32,
    ) AiOpsError!void {
        const M: usize = @intCast(m);
        const N: usize = @intCast(n);
        const K: usize = @intCast(k);
        const LDA: usize = @intCast(lda);
        const LDB: usize = @intCast(ldb);
        const LDC: usize = @intCast(ldc);

        const a: [*]const f32 = @ptrCast(@alignCast(a_ptr));
        const b: [*]const f32 = @ptrCast(@alignCast(b_ptr));
        const c: [*]f32 = @ptrCast(@alignCast(c_ptr));

        for (0..M) |i| {
            for (0..N) |j| {
                var sum: f32 = 0.0;
                for (0..K) |p| {
                    const a_val = if (trans_a == .trans)
                        a[p * LDA + i]
                    else
                        a[i * LDA + p];
                    const b_val = if (trans_b == .trans)
                        b[j * LDB + p]
                    else
                        b[p * LDB + j];
                    sum += a_val * b_val;
                }
                c[i * LDC + j] = alpha * sum + beta * c[i * LDC + j];
            }
        }
    }

    /// CPU batched strided sgemm.
    fn cpuSgemmBatched(
        _: *anyopaque,
        trans_a: Transpose,
        trans_b: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: *const anyopaque,
        lda: i32,
        stride_a: i64,
        b_ptr: *const anyopaque,
        ldb: i32,
        stride_b: i64,
        beta: f32,
        c_ptr: *anyopaque,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) AiOpsError!void {
        const M: usize = @intCast(m);
        const N: usize = @intCast(n);
        const K: usize = @intCast(k);
        const LDA: usize = @intCast(lda);
        const LDB: usize = @intCast(ldb);
        const LDC: usize = @intCast(ldc);
        const sa: usize = @intCast(stride_a);
        const sb: usize = @intCast(stride_b);
        const sc: usize = @intCast(stride_c);

        const a_base: [*]const f32 = @ptrCast(@alignCast(a_ptr));
        const b_base: [*]const f32 = @ptrCast(@alignCast(b_ptr));
        const c_base: [*]f32 = @ptrCast(@alignCast(c_ptr));

        for (0..@intCast(batch_count)) |batch| {
            const a = a_base + batch * sa;
            const b = b_base + batch * sb;
            const c = c_base + batch * sc;

            for (0..M) |i| {
                for (0..N) |j| {
                    var sum: f32 = 0.0;
                    for (0..K) |p| {
                        const a_val = if (trans_a == .trans)
                            a[p * LDA + i]
                        else
                            a[i * LDA + p];
                        const b_val = if (trans_b == .trans)
                            b[j * LDB + p]
                        else
                            b[p * LDB + j];
                        sum += a_val * b_val;
                    }
                    c[i * LDC + j] = alpha * sum + beta * c[i * LDC + j];
                }
            }
        }
    }

    // =========================================================================
    // Activation Operations
    // =========================================================================

    /// CPU softmax: data[i] = exp(data[i]) / sum(exp(data[j])) for j in 0..len.
    /// Uses max-subtraction for numerical stability.
    fn cpuSoftmax(_: *anyopaque, data_ptr: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        if (len == 0) return;
        const n: usize = @intCast(len);
        const data: [*]f32 = @ptrCast(@alignCast(data_ptr));

        // Find max for numerical stability
        var max_val: f32 = data[0];
        for (1..n) |i| {
            if (data[i] > max_val) max_val = data[i];
        }

        // Exponentiate and sum
        var sum: f32 = 0.0;
        for (0..n) |i| {
            data[i] = @exp(data[i] - max_val);
            sum += data[i];
        }

        // Normalize
        const inv_sum = 1.0 / sum;
        for (0..n) |i| {
            data[i] *= inv_sum;
        }
    }

    /// CPU rmsnorm: x = x / rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps).
    fn cpuRmsnorm(
        _: *anyopaque,
        x_ptr: *anyopaque,
        weight_ptr: *const anyopaque,
        len: u32,
        eps: f32,
        _: ?*anyopaque,
    ) AiOpsError!void {
        if (len == 0) return;
        const n: usize = @intCast(len);
        const x: [*]f32 = @ptrCast(@alignCast(x_ptr));
        const weight: [*]const f32 = @ptrCast(@alignCast(weight_ptr));

        // Compute mean of squares
        var sum_sq: f32 = 0.0;
        for (0..n) |i| {
            sum_sq += x[i] * x[i];
        }
        const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);
        const inv_rms = 1.0 / rms;

        // Normalize and apply weight
        for (0..n) |i| {
            x[i] = x[i] * inv_rms * weight[i];
        }
    }

    /// CPU SiLU: x = x * sigmoid(x) = x / (1 + exp(-x)).
    fn cpuSilu(_: *anyopaque, data_ptr: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n: usize = @intCast(len);
        const data: [*]f32 = @ptrCast(@alignCast(data_ptr));

        for (0..n) |i| {
            const val = data[i];
            data[i] = val / (1.0 + @exp(-val));
        }
    }

    /// CPU GELU: x = x * 0.5 * (1 + erf(x / sqrt(2))).
    /// Uses tanh approximation: GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    fn cpuGelu(_: *anyopaque, data_ptr: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n: usize = @intCast(len);
        const data: [*]f32 = @ptrCast(@alignCast(data_ptr));
        const sqrt_2_over_pi: f32 = 0.7978845608028654; // sqrt(2/pi)

        for (0..n) |i| {
            const x = data[i];
            const inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
            const e2 = @exp(2.0 * inner);
            const tanh_val = (e2 - 1.0) / (e2 + 1.0);
            data[i] = 0.5 * x * (1.0 + tanh_val);
        }
    }

    /// CPU scale: x = x * scalar.
    fn cpuScale(_: *anyopaque, data_ptr: *anyopaque, scalar: f32, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n: usize = @intCast(len);
        const data: [*]f32 = @ptrCast(@alignCast(data_ptr));

        for (0..n) |i| {
            data[i] *= scalar;
        }
    }

    /// CPU element-wise multiply: a = a * b.
    fn cpuElementwiseMul(_: *anyopaque, a_ptr: *anyopaque, b_ptr: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n: usize = @intCast(len);
        const a: [*]f32 = @ptrCast(@alignCast(a_ptr));
        const b: [*]const f32 = @ptrCast(@alignCast(b_ptr));

        for (0..n) |i| {
            a[i] *= b[i];
        }
    }

    /// CPU element-wise add: a = a + b.
    fn cpuElementwiseAdd(_: *anyopaque, a_ptr: *anyopaque, b_ptr: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n: usize = @intCast(len);
        const a: [*]f32 = @ptrCast(@alignCast(a_ptr));
        const b: [*]const f32 = @ptrCast(@alignCast(b_ptr));

        for (0..n) |i| {
            a[i] += b[i];
        }
    }

    // =========================================================================
    // Memory Operations — host-backed buffers pretending to be device memory
    // =========================================================================

    /// Allocate host memory as a stand-in for device memory.
    fn cpuAllocDevice(_: *anyopaque, allocator: std.mem.Allocator, size: usize) AiOpsError!DeviceBuffer {
        const slice = allocator.alloc(u8, size) catch return error.OutOfMemory;
        return DeviceBuffer{
            .ptr = @ptrCast(slice.ptr),
            .size = size,
            .allocator = allocator,
            .ops = &(AiOps{
                .ptr = @ptrCast(&instance),
                .vtable = &vtable,
            }),
        };
    }

    /// Copy from host to "device" (host-backed).
    fn cpuCopyToDevice(_: *anyopaque, dst: *anyopaque, src: [*]const u8, len: usize) AiOpsError!void {
        const dst_slice: [*]u8 = @ptrCast(@alignCast(dst));
        @memcpy(dst_slice[0..len], src[0..len]);
    }

    /// Copy from "device" (host-backed) to host.
    fn cpuCopyFromDevice(_: *anyopaque, dst: [*]u8, src: *const anyopaque, len: usize) AiOpsError!void {
        const src_slice: [*]const u8 = @ptrCast(@alignCast(src));
        @memcpy(dst[0..len], src_slice[0..len]);
    }

    /// Free host-backed "device" memory.
    fn cpuFreeDevice(_: *anyopaque, ptr: *anyopaque) void {
        // We cannot free without knowing the size/allocator, so this is a no-op.
        // Real cleanup happens via DeviceBuffer.deinit() which uses the stored allocator.
        _ = ptr;
    }

    /// CPU fallback is always available.
    fn cpuIsAvailable(_: *anyopaque) bool {
        return true;
    }

    fn cpuDeinit(_: *anyopaque) void {}
};

/// Backward-compatible alias for the CPU fallback (was StubAiOps).
pub const StubAiOps = CpuFallbackAiOps;

// =============================================================================
// Helper for creating AiOps from concrete implementation
// =============================================================================

/// Create an AiOps wrapper from a concrete implementation type.
/// The implementation type must have methods matching the VTable signatures.
pub fn createAiOps(comptime Impl: type, impl: *Impl) AiOps {
    const gen = struct {
        fn sgemm(
            ptr: *anyopaque,
            trans_a: Transpose,
            trans_b: Transpose,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const anyopaque,
            lda: i32,
            b: *const anyopaque,
            ldb: i32,
            beta: f32,
            c: *anyopaque,
            ldc: i32,
        ) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.sgemm(trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        }

        fn sgemmStridedBatched(
            ptr: *anyopaque,
            trans_a: Transpose,
            trans_b: Transpose,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const anyopaque,
            lda: i32,
            stride_a: i64,
            b: *const anyopaque,
            ldb: i32,
            stride_b: i64,
            beta: f32,
            c: *anyopaque,
            ldc: i32,
            stride_c: i64,
            batch_count: i32,
        ) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.sgemmStridedBatched(
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                stride_a,
                b,
                ldb,
                stride_b,
                beta,
                c,
                ldc,
                stride_c,
                batch_count,
            );
        }

        fn softmax(ptr: *anyopaque, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.softmax(data, len, stream);
        }

        fn rmsnorm(ptr: *anyopaque, x: *anyopaque, weight: *const anyopaque, len: u32, eps: f32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.rmsnorm(x, weight, len, eps, stream);
        }

        fn silu(ptr: *anyopaque, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.silu(data, len, stream);
        }

        fn gelu(ptr: *anyopaque, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.gelu(data, len, stream);
        }

        fn scale(ptr: *anyopaque, data: *anyopaque, scalar: f32, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.scale(data, scalar, len, stream);
        }

        fn elementwiseMul(ptr: *anyopaque, a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.elementwiseMul(a, b, len, stream);
        }

        fn elementwiseAdd(ptr: *anyopaque, a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.elementwiseAdd(a, b, len, stream);
        }

        fn allocDevice(ptr: *anyopaque, allocator: std.mem.Allocator, size: usize) AiOpsError!DeviceBuffer {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.allocDevice(allocator, size);
        }

        fn copyToDevice(ptr: *anyopaque, dst: *anyopaque, src: [*]const u8, len: usize) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.copyToDevice(dst, src, len);
        }

        fn copyFromDevice(ptr: *anyopaque, dst: [*]u8, src: *const anyopaque, len: usize) AiOpsError!void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.copyFromDevice(dst, src, len);
        }

        fn freeDevice(ptr: *anyopaque, mem: *anyopaque) void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            self.freeDevice(mem);
        }

        fn isAvailable(ptr: *anyopaque) bool {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            return self.isAvailable();
        }

        fn deinitFn(ptr: *anyopaque) void {
            const self: *Impl = @ptrCast(@alignCast(ptr));
            self.deinit();
        }

        const vtable = AiOps.VTable{
            .sgemm = sgemm,
            .sgemmStridedBatched = sgemmStridedBatched,
            .softmax = softmax,
            .rmsnorm = rmsnorm,
            .silu = silu,
            .gelu = gelu,
            .scale = scale,
            .elementwiseMul = elementwiseMul,
            .elementwiseAdd = elementwiseAdd,
            .allocDevice = allocDevice,
            .copyToDevice = copyToDevice,
            .copyFromDevice = copyFromDevice,
            .freeDevice = freeDevice,
            .isAvailable = isAvailable,
            .deinit = deinitFn,
        };
    };

    return .{
        .ptr = impl,
        .vtable = &gen.vtable,
    };
}

// =============================================================================
// Tests
// =============================================================================

test "cpu fallback ai ops is available" {
    const ops = CpuFallbackAiOps.init();
    try std.testing.expect(ops.isAvailable());
}

test "transpose bool conversion" {
    try std.testing.expectEqual(false, Transpose.no_trans.toBool());
    try std.testing.expectEqual(true, Transpose.trans.toBool());
}

test "cpu fallback softmax sums to 1" {
    const ops = CpuFallbackAiOps.init();
    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try ops.softmax(@ptrCast(&data), 4, null);

    var sum: f32 = 0.0;
    for (data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);

    // Values should be monotonically increasing (input was sorted)
    try std.testing.expect(data[0] < data[1]);
    try std.testing.expect(data[1] < data[2]);
    try std.testing.expect(data[2] < data[3]);
}

test "cpu fallback softmax numerical stability" {
    const ops = CpuFallbackAiOps.init();
    // Large values that would overflow without max-subtraction
    var data = [_]f32{ 1000.0, 1001.0, 1002.0 };
    try ops.softmax(@ptrCast(&data), 3, null);

    var sum: f32 = 0.0;
    for (data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-6);
}

test "cpu fallback silu" {
    const ops = CpuFallbackAiOps.init();
    var data = [_]f32{ 0.0, 1.0, -1.0 };
    try ops.silu(@ptrCast(&data), 3, null);

    // silu(0) = 0 * sigmoid(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0], 1e-6);
    // silu(1) = 1 / (1 + exp(-1)) ~ 0.7311
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), data[1], 1e-4);
    // silu(-1) = -1 / (1 + exp(1)) ~ -0.2689
    try std.testing.expectApproxEqAbs(@as(f32, -0.2689414), data[2], 1e-4);
}

test "cpu fallback gelu" {
    const ops = CpuFallbackAiOps.init();
    var data = [_]f32{ 0.0, 1.0, -1.0 };
    try ops.gelu(@ptrCast(&data), 3, null);

    // gelu(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0], 1e-6);
    // gelu(1) ~ 0.8412
    try std.testing.expectApproxEqAbs(@as(f32, 0.8412), data[1], 1e-3);
    // gelu(-1) ~ -0.1588
    try std.testing.expectApproxEqAbs(@as(f32, -0.1588), data[2], 1e-3);
}

test "cpu fallback scale" {
    const ops = CpuFallbackAiOps.init();
    var data = [_]f32{ 1.0, 2.0, 3.0 };
    try ops.scale(@ptrCast(&data), 2.5, 3, null);

    try std.testing.expectApproxEqAbs(@as(f32, 2.5), data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), data[2], 1e-6);
}

test "cpu fallback elementwise mul" {
    const ops = CpuFallbackAiOps.init();
    var a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    try ops.elementwiseMul(@ptrCast(&a), @ptrCast(&b), 3, null);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), a[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), a[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), a[2], 1e-6);
}

test "cpu fallback elementwise add" {
    const ops = CpuFallbackAiOps.init();
    var a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 10.0, 20.0, 30.0 };
    try ops.elementwiseAdd(@ptrCast(&a), @ptrCast(&b), 3, null);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), a[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 22.0), a[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 33.0), a[2], 1e-6);
}

test "cpu fallback sgemm identity" {
    const ops = CpuFallbackAiOps.init();
    // A = [[1, 2], [3, 4]], B = [[1, 0], [0, 1]] (identity)
    // C = A * B = A
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var c = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    try ops.sgemm(
        .no_trans,
        .no_trans,
        2,
        2,
        2,
        1.0,
        @ptrCast(&a),
        2,
        @ptrCast(&b),
        2,
        0.0,
        @ptrCast(&c),
        2,
    );

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), c[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), c[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), c[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), c[3], 1e-6);
}

test "cpu fallback sgemm alpha beta" {
    const ops = CpuFallbackAiOps.init();
    // A = [1, 2; 3, 4], B = [5, 6; 7, 8]
    // A*B = [19, 22; 43, 50]
    // C_init = [1, 1; 1, 1]
    // result = 2.0 * A*B + 3.0 * C_init = [41, 47; 89, 103]
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var c = [_]f32{ 1.0, 1.0, 1.0, 1.0 };

    try ops.sgemm(
        .no_trans,
        .no_trans,
        2,
        2,
        2,
        2.0,
        @ptrCast(&a),
        2,
        @ptrCast(&b),
        2,
        3.0,
        @ptrCast(&c),
        2,
    );

    try std.testing.expectApproxEqAbs(@as(f32, 41.0), c[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 47.0), c[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 89.0), c[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 103.0), c[3], 1e-4);
}

test "cpu fallback sgemm dot product via 1xK * Kx1" {
    const ops = CpuFallbackAiOps.init();
    // dot(a, b) as 1xK * Kx1 matrix multiply
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var c = [_]f32{0.0};

    // sgemv-like: (1x4) * (4x1) = (1x1)
    try ops.sgemm(
        .no_trans,
        .no_trans,
        1,
        1,
        4,
        1.0,
        @ptrCast(&a),
        4,
        @ptrCast(&b),
        1,
        0.0,
        @ptrCast(&c),
        1,
    );

    // dot = 1*5 + 2*6 + 3*7 + 4*8 = 70
    try std.testing.expectApproxEqAbs(@as(f32, 70.0), c[0], 1e-4);
}

test "cpu fallback rmsnorm" {
    const ops = CpuFallbackAiOps.init();
    var x = [_]f32{ 1.0, 2.0, 3.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0 };
    const eps: f32 = 1e-5;

    try ops.rmsnorm(@ptrCast(&x), @ptrCast(&weight), 3, eps, null);

    // rms = sqrt((1+4+9)/3 + 1e-5) = sqrt(4.66667..) ~ 2.16025
    // normalized: [1/rms, 2/rms, 3/rms]
    const rms = @sqrt((1.0 + 4.0 + 9.0) / 3.0 + eps);
    try std.testing.expectApproxEqAbs(1.0 / rms, x[0], 1e-5);
    try std.testing.expectApproxEqAbs(2.0 / rms, x[1], 1e-5);
    try std.testing.expectApproxEqAbs(3.0 / rms, x[2], 1e-5);
}

test "cpu fallback memory round-trip" {
    const ops = CpuFallbackAiOps.init();
    const allocator = std.testing.allocator;

    var buf = try ops.allocDevice(allocator, 4 * @sizeOf(f32));

    // Write data to "device"
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try ops.copyToDevice(buf.ptr.?, @ptrCast(&src), 4 * @sizeOf(f32));

    // Read back
    var dst: [4]f32 = undefined;
    try ops.copyFromDevice(@ptrCast(&dst), buf.ptr.?, 4 * @sizeOf(f32));

    for (0..4) |i| {
        try std.testing.expectApproxEqAbs(src[i], dst[i], 1e-6);
    }

    // Free via allocator (DeviceBuffer stores it)
    const slice: [*]u8 = @ptrCast(@alignCast(buf.ptr.?));
    buf.allocator.free(slice[0..buf.size]);
}

// =============================================================================
// Low-level GPU Module Re-exports for AI Modules
// =============================================================================
//
// These re-exports provide AI modules with access to low-level GPU primitives
// while centralizing the compile-time gating in one place. When GPU is disabled,
// stub types are provided that return error.NotAvailable.

/// GPU backend availability check.
pub const gpu_enabled = build_options.feat_gpu;

/// Device memory management re-exports.
/// Provides DeviceMemory struct with init/deinit and memcpy functions.
pub const memory = if (build_options.feat_gpu and build_options.gpu_fpga)
    // FPGA memory interface would go here
    struct {
        pub fn init(_: std.mem.Allocator) !void {
            std.log.info("FPGA memory simulation initialized", .{});
        }

        pub fn deinit() void {}

        pub const DeviceMemory = struct {
            ptr: ?*anyopaque,
            size: usize,
            allocator: std.mem.Allocator,
            tier: enum { bram, hbm, ddr } = .ddr,

            pub fn init(allocator: std.mem.Allocator, size: usize) !@This() {
                const ptr = try allocator.alloc(u8, size);
                return @This(){
                    .ptr = ptr.ptr,
                    .size = size,
                    .allocator = allocator,
                };
            }

            pub fn deinit(self: *@This()) void {
                const slice = @as([*]u8, @ptrCast(@alignCast(self.ptr)))[0..self.size];
                self.allocator.free(slice);
            }
        };

        pub fn memcpyHostToDevice(dst: *anyopaque, src: *const anyopaque, size: usize) !void {
            const dst_ptr = @as([*]u8, @ptrCast(@alignCast(dst)));
            const src_ptr = @as([*]const u8, @ptrCast(@alignCast(src)));
            @memcpy(dst_ptr[0..size], src_ptr[0..size]);
        }

        pub fn memcpyDeviceToHost(dst: *anyopaque, src: *const anyopaque, size: usize) !void {
            const dst_ptr = @as([*]u8, @ptrCast(@alignCast(dst)));
            const src_ptr = @as([*]const u8, @ptrCast(@alignCast(src)));
            @memcpy(dst_ptr[0..size], src_ptr[0..size]);
        }
    }
else if (build_options.feat_gpu and build_options.gpu_cuda and backend_shared.dynlibSupported)
    @import("backends/cuda/memory.zig")
else
    struct {
        pub fn init(_: std.mem.Allocator) !void {
            return error.NotAvailable;
        }

        pub fn deinit() void {}

        pub const DeviceMemory = struct {
            ptr: ?*anyopaque,
            size: usize,
            allocator: std.mem.Allocator,

            pub fn init(_: std.mem.Allocator, _: usize) !@This() {
                return error.NotAvailable;
            }

            pub fn deinit(_: *@This()) void {}
        };

        pub fn memcpyHostToDevice(_: *anyopaque, _: *const anyopaque, _: usize) !void {
            return error.NotAvailable;
        }

        pub fn memcpyDeviceToHost(_: *anyopaque, _: *anyopaque, _: usize) !void {
            return error.NotAvailable;
        }
    };

/// LLM kernel operations re-exports.
/// Provides LlmKernelModule with softmax, rmsnorm, silu, gelu, scale, etc.
pub const llm_kernels = if (build_options.feat_gpu and build_options.gpu_cuda and backend_shared.dynlibSupported)
    @import("backends/cuda/llm_kernels.zig")
else
    struct {
        pub fn isAvailable() bool {
            return false;
        }

        pub const LlmKernelModule = struct {
            pub fn init(_: std.mem.Allocator) !@This() {
                return error.NotAvailable;
            }

            pub fn deinit(_: *@This()) void {}

            pub fn softmax(_: *@This(), _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn rmsnorm(_: *@This(), _: u64, _: u64, _: u32, _: f32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn silu(_: *@This(), _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn gelu(_: *@This(), _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn elementwiseMul(_: *@This(), _: u64, _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn elementwiseAdd(_: *@This(), _: u64, _: u64, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }

            pub fn scale(_: *@This(), _: u64, _: f32, _: u32, _: ?*anyopaque) !void {
                return error.NotAvailable;
            }
        };
    };

/// cuBLAS operations re-exports.
/// Provides CublasContext with sgemm, sgemmStridedBatched, and matmulRowMajor.
pub const cublas = if (build_options.feat_gpu and build_options.gpu_cuda and backend_shared.dynlibSupported)
    @import("backends/cuda/cublas.zig")
else
    struct {
        pub fn isAvailable() bool {
            return false;
        }

        pub const CublasOperation = enum { no_trans, trans };

        pub const CublasContext = struct {
            pub fn init() !@This() {
                return error.NotAvailable;
            }

            pub fn deinit(_: *@This()) void {}

            pub fn sgemm(
                _: *@This(),
                _: CublasOperation,
                _: CublasOperation,
                _: i32,
                _: i32,
                _: i32,
                _: f32,
                _: *const anyopaque,
                _: i32,
                _: *const anyopaque,
                _: i32,
                _: f32,
                _: *anyopaque,
                _: i32,
            ) !void {
                return error.NotAvailable;
            }

            pub fn sgemmStridedBatched(
                _: *@This(),
                _: CublasOperation,
                _: CublasOperation,
                _: i32,
                _: i32,
                _: i32,
                _: f32,
                _: *const anyopaque,
                _: i32,
                _: i64,
                _: *const anyopaque,
                _: i32,
                _: i64,
                _: f32,
                _: *anyopaque,
                _: i32,
                _: i64,
                _: i32,
            ) !void {
                return error.NotAvailable;
            }
        };

        pub fn matmulRowMajor(
            _: *CublasContext,
            _: *const anyopaque,
            _: *const anyopaque,
            _: *anyopaque,
            _: i32,
            _: i32,
            _: i32,
        ) !void {
            return error.NotAvailable;
        }
    };

/// GPU backend summary for availability detection.
pub const backend = if (build_options.feat_gpu)
    @import("backend.zig")
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

test {
    std.testing.refAllDecls(@This());
}
