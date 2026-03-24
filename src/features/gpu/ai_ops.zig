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
//!
//! Sub-modules:
//! - `ai_ops/cpu_fallback.zig` — CPU fallback implementation
//! - `ai_ops/adapters.zig` — Generic adapter helper for wrapping concrete impls
//! - `ai_ops/reexports.zig` — Low-level GPU module re-exports for AI modules

const std = @import("std");
const build_options = @import("build_options");

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
// Re-exports from sub-modules
// =============================================================================

/// CPU fallback implementation when GPU is disabled.
pub const CpuFallbackAiOps = @import("ai_ops/cpu_fallback.zig").CpuFallbackAiOps;

/// Backward-compatible alias for the CPU fallback (was StubAiOps).
pub const StubAiOps = @import("ai_ops/cpu_fallback.zig").StubAiOps;

/// Create an AiOps wrapper from a concrete implementation type.
pub const createAiOps = @import("ai_ops/adapters.zig").createAiOps;

// =============================================================================
// Low-level GPU Module Re-exports for AI Modules
// =============================================================================

const reexports = @import("ai_ops/reexports.zig");

pub const gpu_enabled = reexports.gpu_enabled;
pub const memory = reexports.memory;
pub const llm_kernels = reexports.llm_kernels;
pub const cublas = reexports.cublas;
pub const backend = reexports.backend;

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
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var c = [_]f32{0.0};

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

    try std.testing.expectApproxEqAbs(@as(f32, 70.0), c[0], 1e-4);
}

test "cpu fallback rmsnorm" {
    const ops = CpuFallbackAiOps.init();
    var x = [_]f32{ 1.0, 2.0, 3.0 };
    const weight = [_]f32{ 1.0, 1.0, 1.0 };
    const eps: f32 = 1e-5;

    try ops.rmsnorm(@ptrCast(&x), @ptrCast(&weight), 3, eps, null);

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

test {
    _ = @import("ai_ops/cpu_fallback.zig");
    _ = @import("ai_ops/adapters.zig");
    _ = @import("ai_ops/reexports.zig");
    std.testing.refAllDecls(@This());
}
