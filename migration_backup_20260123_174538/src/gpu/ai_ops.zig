//! AI GPU Operations Interface
//!
//! Backend-agnostic interface for AI/ML workloads providing BLAS operations,
//! activation functions, and memory management. Enables AI modules to use GPU
//! acceleration without direct coupling to specific backends (CUDA, Vulkan, etc).
//!
//! Design:
//! - VTable pattern matching existing gpu/interface.zig
//! - Compile-time gating via build_options
//! - StubAiOps for when GPU is disabled
//! - Error handling via error union

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
/// var ops = if (build_options.enable_gpu)
///     try cuda_ai_ops.CudaAiOps.init(allocator)
/// else
///     StubAiOps.init();
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
// Stub Implementation
// =============================================================================

/// Stub implementation when GPU is disabled.
/// All operations return error.NotAvailable or false.
pub const StubAiOps = struct {
    /// Singleton instance data (no state needed for stub).
    const Instance = struct {};
    var instance: Instance = .{};

    /// Create a stub AiOps that returns NotAvailable for all operations.
    pub fn init() AiOps {
        return .{
            .ptr = @ptrCast(&instance),
            .vtable = &vtable,
        };
    }

    const vtable = AiOps.VTable{
        .sgemm = stubSgemm,
        .sgemmStridedBatched = stubSgemmBatched,
        .softmax = stubSoftmax,
        .rmsnorm = stubRmsnorm,
        .silu = stubSilu,
        .gelu = stubGelu,
        .scale = stubScale,
        .elementwiseMul = stubElementwiseMul,
        .elementwiseAdd = stubElementwiseAdd,
        .allocDevice = stubAllocDevice,
        .copyToDevice = stubCopyToDevice,
        .copyFromDevice = stubCopyFromDevice,
        .freeDevice = stubFreeDevice,
        .isAvailable = stubIsAvailable,
        .deinit = stubDeinit,
    };

    fn stubSgemm(
        _: *anyopaque,
        _: Transpose,
        _: Transpose,
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
    ) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubSgemmBatched(
        _: *anyopaque,
        _: Transpose,
        _: Transpose,
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
    ) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubSoftmax(_: *anyopaque, _: *anyopaque, _: u32, _: ?*anyopaque) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubRmsnorm(_: *anyopaque, _: *anyopaque, _: *const anyopaque, _: u32, _: f32, _: ?*anyopaque) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubSilu(_: *anyopaque, _: *anyopaque, _: u32, _: ?*anyopaque) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubGelu(_: *anyopaque, _: *anyopaque, _: u32, _: ?*anyopaque) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubScale(_: *anyopaque, _: *anyopaque, _: f32, _: u32, _: ?*anyopaque) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubElementwiseMul(_: *anyopaque, _: *anyopaque, _: *const anyopaque, _: u32, _: ?*anyopaque) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubElementwiseAdd(_: *anyopaque, _: *anyopaque, _: *const anyopaque, _: u32, _: ?*anyopaque) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubAllocDevice(_: *anyopaque, _: std.mem.Allocator, _: usize) AiOpsError!DeviceBuffer {
        return error.NotAvailable;
    }

    fn stubCopyToDevice(_: *anyopaque, _: *anyopaque, _: [*]const u8, _: usize) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubCopyFromDevice(_: *anyopaque, _: [*]u8, _: *const anyopaque, _: usize) AiOpsError!void {
        return error.NotAvailable;
    }

    fn stubFreeDevice(_: *anyopaque, _: *anyopaque) void {}

    fn stubIsAvailable(_: *anyopaque) bool {
        return false;
    }

    fn stubDeinit(_: *anyopaque) void {}
};

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

test "stub ai ops returns not available" {
    const ops = StubAiOps.init();

    // Verify isAvailable returns false
    try std.testing.expectEqual(false, ops.isAvailable());

    // Verify operations return NotAvailable
    try std.testing.expectError(error.NotAvailable, ops.softmax(undefined, 0, null));
    try std.testing.expectError(error.NotAvailable, ops.silu(undefined, 0, null));
}

test "transpose bool conversion" {
    try std.testing.expectEqual(false, Transpose.no_trans.toBool());
    try std.testing.expectEqual(true, Transpose.trans.toBool());
}

// =============================================================================
// Low-level GPU Module Re-exports for AI Modules
// =============================================================================
//
// These re-exports provide AI modules with access to low-level GPU primitives
// while centralizing the compile-time gating in one place. When GPU is disabled,
// stub types are provided that return error.NotAvailable.

/// GPU backend availability check.
pub const gpu_enabled = build_options.enable_gpu;

/// Device memory management re-exports.
/// Provides DeviceMemory struct with init/deinit and memcpy functions.
pub const memory = if (build_options.enable_gpu)
    @import("backends/cuda/memory.zig")
else
    struct {
        pub fn init() !void {
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
pub const llm_kernels = if (build_options.enable_gpu)
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
pub const cublas = if (build_options.enable_gpu)
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
pub const backend = if (build_options.enable_gpu)
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
