//! cuBLAS Library Bindings
//!
//! Provides GPU-accelerated BLAS operations for LLM training:
//! - SGEMM (single-precision matrix multiplication)
//! - SGEMM batched
//! - SGEMM strided batched

const std = @import("std");
const builtin = @import("builtin");
const cuda_loader = @import("loader.zig");
const shared = @import("../shared.zig");

/// cuBLAS status codes
pub const CublasStatus = enum(i32) {
    success = 0,
    not_initialized = 1,
    alloc_failed = 3,
    invalid_value = 7,
    arch_mismatch = 8,
    mapping_error = 11,
    execution_failed = 13,
    internal_error = 14,
    not_supported = 15,
    license_error = 16,
    _,
};

/// cuBLAS operation type
pub const CublasOperation = enum(i32) {
    no_trans = 0, // N
    trans = 1, // T
    conj_trans = 2, // C
};

/// cuBLAS compute type (for mixed precision)
pub const CublasComputeType = enum(i32) {
    compute_16f = 64,
    compute_16f_pedantic = 65,
    compute_32f = 68,
    compute_32f_pedantic = 69,
    compute_32f_fast_16f = 74,
    compute_32f_fast_tf32 = 77,
    compute_64f = 70,
    compute_64f_pedantic = 71,
};

/// cuBLAS GEMM algorithm
pub const CublasGemmAlgo = enum(i32) {
    default = -1,
    algo_0 = 0,
    algo_1 = 1,
    algo_2 = 2,
    algo_3 = 3,
    algo_4 = 4,
    algo_5 = 5,
    algo_6 = 6,
    algo_7 = 7,
};

// cuBLAS function types
pub const CublasCreateFn = *const fn (*?*anyopaque) callconv(.c) CublasStatus;
pub const CublasDestroyFn = *const fn (?*anyopaque) callconv(.c) CublasStatus;
pub const CublasSetStreamFn = *const fn (?*anyopaque, ?*anyopaque) callconv(.c) CublasStatus;
pub const CublasGetVersionFn = *const fn (?*anyopaque, *i32) callconv(.c) CublasStatus;

pub const CublasSgemmFn = *const fn (
    handle: ?*anyopaque,
    transa: CublasOperation,
    transb: CublasOperation,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: *const anyopaque,
    lda: i32,
    b: *const anyopaque,
    ldb: i32,
    beta: *const f32,
    c: *anyopaque,
    ldc: i32,
) callconv(.c) CublasStatus;

pub const CublasSgemmBatchedFn = *const fn (
    handle: ?*anyopaque,
    transa: CublasOperation,
    transb: CublasOperation,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a_array: *const *const anyopaque,
    lda: i32,
    b_array: *const *const anyopaque,
    ldb: i32,
    beta: *const f32,
    c_array: *const *anyopaque,
    ldc: i32,
    batch_count: i32,
) callconv(.c) CublasStatus;

pub const CublasSgemmStridedBatchedFn = *const fn (
    handle: ?*anyopaque,
    transa: CublasOperation,
    transb: CublasOperation,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: *const anyopaque,
    lda: i32,
    stride_a: i64,
    b: *const anyopaque,
    ldb: i32,
    stride_b: i64,
    beta: *const f32,
    c: *anyopaque,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) callconv(.c) CublasStatus;

/// cuBLAS function set
pub const CublasFunctions = struct {
    cublasCreate: ?CublasCreateFn = null,
    cublasDestroy: ?CublasDestroyFn = null,
    cublasSetStream: ?CublasSetStreamFn = null,
    cublasGetVersion: ?CublasGetVersionFn = null,
    cublasSgemm: ?CublasSgemmFn = null,
    cublasSgemmBatched: ?CublasSgemmBatchedFn = null,
    cublasSgemmStridedBatched: ?CublasSgemmStridedBatchedFn = null,
};

var cublas_lib: ?std.DynLib = null;
var cublas_functions: CublasFunctions = .{};
var cublas_load_attempted: bool = false;

/// Load cuBLAS library
pub fn load() !*const CublasFunctions {
    if (!shared.canUseDynLib()) {
        cublas_load_attempted = true;
        return error.PlatformNotSupported;
    }

    if (cublas_load_attempted) {
        if (cublas_lib != null) return &cublas_functions;
        return error.LibraryNotFound;
    }
    cublas_load_attempted = true;

    if (!shared.canUseDynLib()) {
        return error.PlatformNotSupported;
    }

    if (!shared.dynlibSupported) {
        return error.PlatformNotSupported;
    }

    // Try platform-specific library names
    const lib_names: []const []const u8 = switch (builtin.os.tag) {
        .windows => &.{
            "cublas64_12.dll",
            "cublas64_11.dll",
            "cublas64_10.dll",
        },
        .linux => &.{
            "libcublas.so.12",
            "libcublas.so.11",
            "libcublas.so.10",
            "libcublas.so",
        },
        else => return error.PlatformNotSupported,
    };

    for (lib_names) |name| {
        const lib = std.DynLib.open(name) catch continue;
        cublas_lib = lib;
        break;
    }

    if (cublas_lib == null) return error.LibraryNotFound;

    if (cublas_lib) |*lib| {
        cublas_functions.cublasCreate = lib.lookup(CublasCreateFn, "cublasCreate_v2");
        cublas_functions.cublasDestroy = lib.lookup(CublasDestroyFn, "cublasDestroy_v2");
        cublas_functions.cublasSetStream = lib.lookup(CublasSetStreamFn, "cublasSetStream_v2");
        cublas_functions.cublasGetVersion = lib.lookup(CublasGetVersionFn, "cublasGetVersion_v2");
        cublas_functions.cublasSgemm = lib.lookup(CublasSgemmFn, "cublasSgemm_v2");
        cublas_functions.cublasSgemmBatched = lib.lookup(CublasSgemmBatchedFn, "cublasSgemmBatched");
        cublas_functions.cublasSgemmStridedBatched = lib.lookup(CublasSgemmStridedBatchedFn, "cublasSgemmStridedBatched");
    } else {
        return error.LibraryNotFound;
    }

    return &cublas_functions;
}

/// Unload cuBLAS library
pub fn unload() void {
    if (!shared.canUseDynLib()) {
        cublas_functions = .{};
        cublas_load_attempted = false;
        return;
    }

    if (cublas_lib) |*lib| {
        lib.close();
        cublas_lib = null;
    }
    cublas_functions = .{};
    cublas_load_attempted = false;
}

/// Check if cuBLAS is available
pub fn isAvailable() bool {
    if (!shared.canUseDynLib()) return false;
    if (!cublas_load_attempted) {
        _ = load() catch return false;
    }
    return cublas_lib != null and cublas_functions.cublasCreate != null;
}

/// Get loaded functions
pub fn getFunctions() ?*const CublasFunctions {
    if (cublas_lib == null) return null;
    return &cublas_functions;
}

/// Convert cuBLAS status to error
pub fn checkStatus(status: CublasStatus) !void {
    return switch (status) {
        .success => {},
        .not_initialized => error.NotInitialized,
        .alloc_failed => error.OutOfMemory,
        .invalid_value => error.InvalidValue,
        .execution_failed => error.ExecutionFailed,
        .internal_error => error.InternalError,
        else => error.CublasError,
    };
}

pub const CublasError = error{
    NotInitialized,
    OutOfMemory,
    InvalidValue,
    ExecutionFailed,
    InternalError,
    CublasError,
    LibraryNotFound,
    HandleNotCreated,
};

/// High-level cuBLAS context
pub const CublasContext = struct {
    handle: ?*anyopaque,
    functions: *const CublasFunctions,

    pub fn init() CublasError!CublasContext {
        const funcs = load() catch return error.LibraryNotFound;
        const create_fn = funcs.cublasCreate orelse return error.LibraryNotFound;

        var handle: ?*anyopaque = null;
        const status = create_fn(&handle);
        try checkStatus(status);

        if (handle == null) return error.HandleNotCreated;

        return .{
            .handle = handle,
            .functions = funcs,
        };
    }

    pub fn deinit(self: *CublasContext) void {
        if (self.functions.cublasDestroy) |destroy| {
            _ = destroy(self.handle);
        }
        self.handle = null;
    }

    /// Set CUDA stream for operations
    pub fn setStream(self: *CublasContext, stream: ?*anyopaque) !void {
        const set_stream = self.functions.cublasSetStream orelse return error.LibraryNotFound;
        try checkStatus(set_stream(self.handle, stream));
    }

    /// Get cuBLAS version
    pub fn getVersion(self: *CublasContext) !i32 {
        const get_version = self.functions.cublasGetVersion orelse return error.LibraryNotFound;
        var version: i32 = 0;
        try checkStatus(get_version(self.handle, &version));
        return version;
    }

    /// Single-precision matrix multiplication: C = alpha * op(A) * op(B) + beta * C
    /// A: [m, k] if no_trans, [k, m] if trans
    /// B: [k, n] if no_trans, [n, k] if trans
    /// C: [m, n]
    pub fn sgemm(
        self: *CublasContext,
        trans_a: CublasOperation,
        trans_b: CublasOperation,
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
    ) !void {
        const sgemm_fn = self.functions.cublasSgemm orelse return error.LibraryNotFound;
        try checkStatus(sgemm_fn(
            self.handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            &alpha,
            a,
            lda,
            b,
            ldb,
            &beta,
            c,
            ldc,
        ));
    }

    /// Batched SGEMM with strided memory layout
    pub fn sgemmStridedBatched(
        self: *CublasContext,
        trans_a: CublasOperation,
        trans_b: CublasOperation,
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
    ) !void {
        const fn_ptr = self.functions.cublasSgemmStridedBatched orelse return error.LibraryNotFound;
        try checkStatus(fn_ptr(
            self.handle,
            trans_a,
            trans_b,
            m,
            n,
            k,
            &alpha,
            a,
            lda,
            stride_a,
            b,
            ldb,
            stride_b,
            &beta,
            c,
            ldc,
            stride_c,
            batch_count,
        ));
    }
};

/// Helper for matrix multiplication with row-major layout
/// (cuBLAS uses column-major, so we transpose the operation)
pub fn matmulRowMajor(
    ctx: *CublasContext,
    a: *const anyopaque, // [m, k]
    b: *const anyopaque, // [k, n]
    c: *anyopaque, // [m, n]
    m: i32,
    k: i32,
    n: i32,
) !void {
    // For row-major: C = A @ B
    // In column-major: C^T = B^T @ A^T
    // So we call: sgemm(N, N, n, m, k, B, A, C)
    try ctx.sgemm(
        .no_trans,
        .no_trans,
        n, // rows of B^T = cols of B
        m, // cols of A^T = rows of A
        k, // inner dim
        1.0, // alpha
        b,
        n, // ldb
        a,
        k, // lda
        0.0, // beta
        c,
        n, // ldc
    );
}

/// Helper for backward matmul: dA = dC @ B^T
pub fn matmulBackwardA(
    ctx: *CublasContext,
    dc: *const anyopaque, // [m, n]
    b: *const anyopaque, // [k, n]
    da: *anyopaque, // [m, k]
    m: i32,
    k: i32,
    n: i32,
) !void {
    // dA = dC @ B^T
    // In col-major: dA^T = B @ dC^T
    try ctx.sgemm(
        .no_trans,
        .trans,
        k, // rows of B
        m, // cols of dC^T
        n, // inner dim
        1.0,
        b,
        k,
        dc,
        n,
        1.0, // accumulate
        da,
        k,
    );
}

/// Helper for backward matmul: dB = A^T @ dC
pub fn matmulBackwardB(
    ctx: *CublasContext,
    a: *const anyopaque, // [m, k]
    dc: *const anyopaque, // [m, n]
    db: *anyopaque, // [k, n]
    m: i32,
    k: i32,
    n: i32,
) !void {
    // dB = A^T @ dC
    // In col-major: dB^T = dC^T @ A
    try ctx.sgemm(
        .trans,
        .no_trans,
        n, // rows of dC^T
        k, // cols of A
        m, // inner dim
        1.0,
        dc,
        n,
        a,
        k,
        1.0, // accumulate
        db,
        n,
    );
}

test "cublas availability check" {
    // Just check that availability check doesn't crash
    const available = isAvailable();
    _ = available;
}

test {
    std.testing.refAllDecls(@This());
}
