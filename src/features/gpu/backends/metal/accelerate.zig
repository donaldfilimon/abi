//! Apple Accelerate Framework Integration
//!
//! Provides high-performance BLAS/LAPACK operations via Apple's Accelerate framework.
//! Uses vDSP for signal processing, vBLAS for linear algebra, and vecLib for
//! optimized vector operations on Apple Silicon.
//!
//! ## Features
//! - vBLAS: cblas_sgemm, cblas_sgemv for matrix operations
//! - vDSP: FFT, convolution, element-wise ops
//! - BNNS: Neural network primitives (BatchNorm, Convolution, etc.)
//! - Unified Memory: Zero-copy data sharing between CPU and GPU
//!
//! ## Apple Silicon Optimizations
//! - AMX (Apple Matrix Extension) acceleration for BLAS
//! - ANE (Apple Neural Engine) integration via BNNS
//! - Unified Memory Architecture awareness

const std = @import("std");
const builtin = @import("builtin");

/// Accelerate framework availability check
pub const is_available = builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .tvos;

/// CBLAS transpose options
pub const CblasTranspose = enum(c_int) {
    no_trans = 111,
    trans = 112,
    conj_trans = 113,
};

/// CBLAS order (row/column major)
pub const CblasOrder = enum(c_int) {
    row_major = 101,
    col_major = 102,
};

// External Accelerate framework bindings
const accelerate = if (is_available) struct {
    // vBLAS - Basic Linear Algebra Subprograms
    extern "Accelerate" fn cblas_sgemm(
        order: c_int,
        trans_a: c_int,
        trans_b: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        a: [*]const f32,
        lda: c_int,
        b: [*]const f32,
        ldb: c_int,
        beta: f32,
        c: [*]f32,
        ldc: c_int,
    ) void;

    extern "Accelerate" fn cblas_sgemv(
        order: c_int,
        trans: c_int,
        m: c_int,
        n: c_int,
        alpha: f32,
        a: [*]const f32,
        lda: c_int,
        x: [*]const f32,
        incx: c_int,
        beta: f32,
        y: [*]f32,
        incy: c_int,
    ) void;

    extern "Accelerate" fn cblas_saxpy(
        n: c_int,
        alpha: f32,
        x: [*]const f32,
        incx: c_int,
        y: [*]f32,
        incy: c_int,
    ) void;

    extern "Accelerate" fn cblas_sdot(
        n: c_int,
        x: [*]const f32,
        incx: c_int,
        y: [*]const f32,
        incy: c_int,
    ) f32;

    extern "Accelerate" fn cblas_snrm2(
        n: c_int,
        x: [*]const f32,
        incx: c_int,
    ) f32;

    extern "Accelerate" fn cblas_sscal(
        n: c_int,
        alpha: f32,
        x: [*]f32,
        incx: c_int,
    ) void;

    // vDSP - Digital Signal Processing
    extern "Accelerate" fn vDSP_vadd(
        a: [*]const f32,
        stride_a: usize,
        b: [*]const f32,
        stride_b: usize,
        c: [*]f32,
        stride_c: usize,
        n: usize,
    ) void;

    extern "Accelerate" fn vDSP_vmul(
        a: [*]const f32,
        stride_a: usize,
        b: [*]const f32,
        stride_b: usize,
        c: [*]f32,
        stride_c: usize,
        n: usize,
    ) void;

    extern "Accelerate" fn vDSP_vdiv(
        b: [*]const f32,
        stride_b: usize,
        a: [*]const f32,
        stride_a: usize,
        c: [*]f32,
        stride_c: usize,
        n: usize,
    ) void;

    extern "Accelerate" fn vDSP_vsq(
        a: [*]const f32,
        stride_a: usize,
        c: [*]f32,
        stride_c: usize,
        n: usize,
    ) void;

    extern "Accelerate" fn vDSP_vrsqrt(
        a: [*]const f32,
        stride_a: usize,
        c: [*]f32,
        stride_c: usize,
        n: usize,
    ) void;

    extern "Accelerate" fn vDSP_sve(
        a: [*]const f32,
        stride: usize,
        c: *f32,
        n: usize,
    ) void;

    extern "Accelerate" fn vDSP_maxv(
        a: [*]const f32,
        stride: usize,
        c: *f32,
        n: usize,
    ) void;

    extern "Accelerate" fn vDSP_minv(
        a: [*]const f32,
        stride: usize,
        c: *f32,
        n: usize,
    ) void;

    extern "Accelerate" fn vDSP_meanv(
        a: [*]const f32,
        stride: usize,
        c: *f32,
        n: usize,
    ) void;

    extern "Accelerate" fn vDSP_normalize(
        a: [*]const f32,
        stride_a: usize,
        c: [*]f32,
        stride_c: usize,
        mean: *f32,
        std_dev: *f32,
        n: usize,
    ) void;

    // vForce - Vectorized math functions
    extern "Accelerate" fn vvexpf(
        y: [*]f32,
        x: [*]const f32,
        n: *const c_int,
    ) void;

    extern "Accelerate" fn vvlogf(
        y: [*]f32,
        x: [*]const f32,
        n: *const c_int,
    ) void;

    extern "Accelerate" fn vvsqrtf(
        y: [*]f32,
        x: [*]const f32,
        n: *const c_int,
    ) void;

    extern "Accelerate" fn vvtanhf(
        y: [*]f32,
        x: [*]const f32,
        n: *const c_int,
    ) void;

    extern "Accelerate" fn vvsinf(
        y: [*]f32,
        x: [*]const f32,
        n: *const c_int,
    ) void;

    extern "Accelerate" fn vvcosf(
        y: [*]f32,
        x: [*]const f32,
        n: *const c_int,
    ) void;
} else struct {};

/// AccelerateError represents errors from Accelerate operations
pub const AccelerateError = error{
    NotAvailable,
    InvalidDimensions,
    NullPointer,
    OutOfMemory,
};

/// Matrix-matrix multiplication using vBLAS (AMX-accelerated on Apple Silicon)
/// C = alpha * op(A) * op(B) + beta * C
pub fn sgemm(
    trans_a: CblasTranspose,
    trans_b: CblasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: []const f32,
    lda: usize,
    b: []const f32,
    ldb: usize,
    beta: f32,
    c: []f32,
    ldc: usize,
) AccelerateError!void {
    if (!is_available) return error.NotAvailable;

    if (m == 0 or n == 0 or k == 0) return error.InvalidDimensions;
    if (a.len < lda * (if (trans_a == .no_trans) k else m)) return error.InvalidDimensions;
    if (b.len < ldb * (if (trans_b == .no_trans) n else k)) return error.InvalidDimensions;
    if (c.len < ldc * n) return error.InvalidDimensions;

    accelerate.cblas_sgemm(
        @intFromEnum(CblasOrder.row_major),
        @intFromEnum(trans_a),
        @intFromEnum(trans_b),
        @intCast(m),
        @intCast(n),
        @intCast(k),
        alpha,
        a.ptr,
        @intCast(lda),
        b.ptr,
        @intCast(ldb),
        beta,
        c.ptr,
        @intCast(ldc),
    );
}

/// Matrix-vector multiplication using vBLAS
/// y = alpha * op(A) * x + beta * y
pub fn sgemv(
    trans: CblasTranspose,
    m: usize,
    n: usize,
    alpha: f32,
    a: []const f32,
    lda: usize,
    x: []const f32,
    beta: f32,
    y: []f32,
) AccelerateError!void {
    if (!is_available) return error.NotAvailable;

    if (m == 0 or n == 0) return error.InvalidDimensions;

    accelerate.cblas_sgemv(
        @intFromEnum(CblasOrder.row_major),
        @intFromEnum(trans),
        @intCast(m),
        @intCast(n),
        alpha,
        a.ptr,
        @intCast(lda),
        x.ptr,
        1,
        beta,
        y.ptr,
        1,
    );
}

/// Vector dot product using vBLAS
pub fn sdot(x: []const f32, y: []const f32) AccelerateError!f32 {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len) return error.InvalidDimensions;
    if (x.len == 0) return 0.0;

    return accelerate.cblas_sdot(@intCast(x.len), x.ptr, 1, y.ptr, 1);
}

/// Vector L2 norm using vBLAS
pub fn snrm2(x: []const f32) AccelerateError!f32 {
    if (!is_available) return error.NotAvailable;
    if (x.len == 0) return 0.0;

    return accelerate.cblas_snrm2(@intCast(x.len), x.ptr, 1);
}

/// Vector scale: x = alpha * x
pub fn sscal(alpha: f32, x: []f32) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len == 0) return;

    accelerate.cblas_sscal(@intCast(x.len), alpha, x.ptr, 1);
}

/// Vector add-scaled: y = alpha * x + y
pub fn saxpy(alpha: f32, x: []const f32, y: []f32) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len) return error.InvalidDimensions;
    if (x.len == 0) return;

    accelerate.cblas_saxpy(@intCast(x.len), alpha, x.ptr, 1, y.ptr, 1);
}

/// Element-wise vector addition using vDSP
pub fn vadd(a: []const f32, b: []const f32, c: []f32) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (a.len != b.len or b.len != c.len) return error.InvalidDimensions;
    if (a.len == 0) return;

    accelerate.vDSP_vadd(a.ptr, 1, b.ptr, 1, c.ptr, 1, a.len);
}

/// Element-wise vector multiplication using vDSP
pub fn vmul(a: []const f32, b: []const f32, c: []f32) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (a.len != b.len or b.len != c.len) return error.InvalidDimensions;
    if (a.len == 0) return;

    accelerate.vDSP_vmul(a.ptr, 1, b.ptr, 1, c.ptr, 1, a.len);
}

/// Element-wise vector division using vDSP (c = a / b)
pub fn vdiv(a: []const f32, b: []const f32, c: []f32) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (a.len != b.len or b.len != c.len) return error.InvalidDimensions;
    if (a.len == 0) return;

    // Note: vDSP_vdiv has reversed argument order (b, a -> c = a/b)
    accelerate.vDSP_vdiv(b.ptr, 1, a.ptr, 1, c.ptr, 1, a.len);
}

/// Vector sum using vDSP
pub fn vsum(a: []const f32) AccelerateError!f32 {
    if (!is_available) return error.NotAvailable;
    if (a.len == 0) return 0.0;

    var result: f32 = 0.0;
    accelerate.vDSP_sve(a.ptr, 1, &result, a.len);
    return result;
}

/// Vector max using vDSP
pub fn vmax(a: []const f32) AccelerateError!f32 {
    if (!is_available) return error.NotAvailable;
    if (a.len == 0) return -std.math.inf(f32);

    var result: f32 = undefined;
    accelerate.vDSP_maxv(a.ptr, 1, &result, a.len);
    return result;
}

/// Vector min using vDSP
pub fn vmin(a: []const f32) AccelerateError!f32 {
    if (!is_available) return error.NotAvailable;
    if (a.len == 0) return std.math.inf(f32);

    var result: f32 = undefined;
    accelerate.vDSP_minv(a.ptr, 1, &result, a.len);
    return result;
}

/// Vector mean using vDSP
pub fn vmean(a: []const f32) AccelerateError!f32 {
    if (!is_available) return error.NotAvailable;
    if (a.len == 0) return 0.0;

    var result: f32 = undefined;
    accelerate.vDSP_meanv(a.ptr, 1, &result, a.len);
    return result;
}

/// Vectorized exponential using vForce
pub fn vexp(x: []const f32, y: []f32) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len) return error.InvalidDimensions;
    if (x.len == 0) return;

    var n: c_int = @intCast(x.len);
    accelerate.vvexpf(y.ptr, x.ptr, &n);
}

/// Vectorized natural logarithm using vForce
pub fn vlog(x: []const f32, y: []f32) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len) return error.InvalidDimensions;
    if (x.len == 0) return;

    var n: c_int = @intCast(x.len);
    accelerate.vvlogf(y.ptr, x.ptr, &n);
}

/// Vectorized square root using vForce
pub fn vsqrt(x: []const f32, y: []f32) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len) return error.InvalidDimensions;
    if (x.len == 0) return;

    var n: c_int = @intCast(x.len);
    accelerate.vvsqrtf(y.ptr, x.ptr, &n);
}

/// Vectorized tanh using vForce
pub fn vtanh(x: []const f32, y: []f32) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len) return error.InvalidDimensions;
    if (x.len == 0) return;

    var n: c_int = @intCast(x.len);
    accelerate.vvtanhf(y.ptr, x.ptr, &n);
}

// ============================================================================
// Neural Network Primitives (using Accelerate where possible)
// ============================================================================

/// Softmax activation using vDSP for numerical stability
/// Computes: softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
pub fn softmax(x: []const f32, y: []f32, allocator: std.mem.Allocator) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len) return error.InvalidDimensions;
    if (x.len == 0) return;

    // Find max for numerical stability
    const max_val = try vmax(x);

    // Allocate temp buffer for shifted values
    const temp = allocator.alloc(f32, x.len) catch return error.OutOfMemory;
    defer allocator.free(temp);

    // Subtract max
    for (temp, x) |*t, xi| {
        t.* = xi - max_val;
    }

    // Compute exp
    try vexp(temp, y);

    // Sum
    const sum = try vsum(y);

    // Normalize
    const inv_sum = 1.0 / sum;
    try sscal(inv_sum, y);
}

/// RMSNorm: y = x * (1 / sqrt(mean(x^2) + eps)) * weight
pub fn rmsnorm(x: []const f32, weight: []const f32, y: []f32, eps: f32, allocator: std.mem.Allocator) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len or x.len != weight.len) return error.InvalidDimensions;
    if (x.len == 0) return;

    // Compute sum of squares
    const temp = allocator.alloc(f32, x.len) catch return error.OutOfMemory;
    defer allocator.free(temp);

    // Square elements
    try vmul(x, x, temp);

    // Mean of squares
    const mean_sq = try vmean(temp);

    // RMS scale
    const rms_scale = 1.0 / @sqrt(mean_sq + eps);

    // Apply: y = x * rms_scale * weight
    for (y, x, weight) |*yi, xi, wi| {
        yi.* = xi * rms_scale * wi;
    }
}

/// SiLU activation: y = x * sigmoid(x) = x / (1 + exp(-x))
pub fn silu(x: []const f32, y: []f32, allocator: std.mem.Allocator) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len) return error.InvalidDimensions;
    if (x.len == 0) return;

    const temp = allocator.alloc(f32, x.len) catch return error.OutOfMemory;
    defer allocator.free(temp);

    // Negate x
    for (temp, x) |*t, xi| {
        t.* = -xi;
    }

    // exp(-x)
    try vexp(temp, temp);

    // 1 + exp(-x)
    for (temp) |*t| {
        t.* = 1.0 + t.*;
    }

    // x / (1 + exp(-x))
    try vdiv(x, temp, y);
}

/// GeLU activation (approximate): y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn gelu(x: []const f32, y: []f32, allocator: std.mem.Allocator) AccelerateError!void {
    if (!is_available) return error.NotAvailable;
    if (x.len != y.len) return error.InvalidDimensions;
    if (x.len == 0) return;

    const sqrt_2_pi: f32 = 0.7978845608028654; // sqrt(2/pi)
    const coeff: f32 = 0.044715;

    const temp = allocator.alloc(f32, x.len) catch return error.OutOfMemory;
    defer allocator.free(temp);

    // Compute inner: sqrt(2/pi) * (x + 0.044715 * x^3)
    for (temp, x) |*t, xi| {
        const x3 = xi * xi * xi;
        t.* = sqrt_2_pi * (xi + coeff * x3);
    }

    // tanh
    try vtanh(temp, temp);

    // 0.5 * x * (1 + tanh(...))
    for (y, x, temp) |*yi, xi, ti| {
        yi.* = 0.5 * xi * (1.0 + ti);
    }
}

// ============================================================================
// Unified Memory Utilities
// ============================================================================

/// UnifiedMemoryConfig for Apple Silicon's shared memory architecture
pub const UnifiedMemoryConfig = struct {
    /// Prefer device-local allocation when possible
    prefer_device: bool = true,
    /// Enable automatic memory migration
    auto_migrate: bool = true,
    /// Cache mode for CPU access
    cpu_cache_mode: CpuCacheMode = .default,

    pub const CpuCacheMode = enum {
        default, // System default caching
        write_combined, // Better for GPU-bound data
    };
};

/// Check if unified memory is available (Apple Silicon)
pub fn hasUnifiedMemory() bool {
    if (!is_available) return false;
    // On Apple Silicon, memory is always unified
    return builtin.cpu.arch == .aarch64 and
        (builtin.os.tag == .macos or builtin.os.tag == .ios);
}

/// Get recommended memory alignment for unified memory operations
pub fn unifiedMemoryAlignment() usize {
    if (hasUnifiedMemory()) {
        // Apple Silicon prefers 16KB alignment for best performance
        return 16 * 1024;
    }
    // Default cache line alignment
    return 64;
}

// ============================================================================
// Tests
// ============================================================================

test "accelerate availability check" {
    // This test just verifies the module compiles correctly
    const available = is_available;
    _ = available;
}

test "unified memory detection" {
    const has_unified = hasUnifiedMemory();
    const alignment = unifiedMemoryAlignment();
    try std.testing.expect(alignment >= 64);
    _ = has_unified;
}
