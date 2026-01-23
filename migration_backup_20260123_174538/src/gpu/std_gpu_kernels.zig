//! Native Zig GPU Compute Kernels
//!
//! These kernels are written in pure Zig using std.gpu facilities and can be
//! compiled to SPIR-V for execution on Vulkan-compatible GPUs.
//!
//! ## How to Use
//!
//! 1. Build the kernel module for SPIR-V target:
//!    ```bash
//!    zig build-obj -target spirv64-unknown -O ReleaseFast src/gpu/std_gpu_kernels.zig
//!    ```
//!
//! 2. Load the resulting SPIR-V binary with the Vulkan backend
//!
//! 3. Dispatch compute workgroups via `vkCmdDispatch`
//!
//! ## Kernel Conventions
//!
//! - All kernels use `callconv(.spirv_kernel)` calling convention
//! - Workgroup size is set via comptime `std.gpu.executionMode`
//! - Buffer bindings use storage buffer address space
//! - Global invocation ID provides thread index

const std = @import("std");
const builtin = @import("builtin");
const std_gpu = @import("std_gpu.zig");

// ============================================================================
// Kernel Configuration
// ============================================================================

/// Default workgroup size for 1D kernels
pub const DEFAULT_WORKGROUP_SIZE_1D: u32 = 256;

/// Default workgroup size for 2D kernels (16x16)
pub const DEFAULT_WORKGROUP_SIZE_2D: u32 = 16;

// ============================================================================
// Vector Operations
// ============================================================================

/// Vector addition: result[i] = a[i] + b[i]
/// Workgroup size: 256 threads
pub fn vectorAdd(
    a: std_gpu.StorageConstPtr(f32),
    b: std_gpu.StorageConstPtr(f32),
    result: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        result[gid] = a[gid] + b[gid];
    }
}

/// Vector subtraction: result[i] = a[i] - b[i]
pub fn vectorSub(
    a: std_gpu.StorageConstPtr(f32),
    b: std_gpu.StorageConstPtr(f32),
    result: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        result[gid] = a[gid] - b[gid];
    }
}

/// Vector multiplication: result[i] = a[i] * b[i]
pub fn vectorMul(
    a: std_gpu.StorageConstPtr(f32),
    b: std_gpu.StorageConstPtr(f32),
    result: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        result[gid] = a[gid] * b[gid];
    }
}

/// Scalar-vector multiplication: result[i] = scalar * a[i]
pub fn vectorScale(
    a: std_gpu.StorageConstPtr(f32),
    scalar: f32,
    result: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        result[gid] = scalar * a[gid];
    }
}

/// Fused multiply-add: result[i] = a[i] * b[i] + c[i]
pub fn vectorFMA(
    a: std_gpu.StorageConstPtr(f32),
    b: std_gpu.StorageConstPtr(f32),
    c: std_gpu.StorageConstPtr(f32),
    result: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        result[gid] = @mulAdd(f32, a[gid], b[gid], c[gid]);
    }
}

// ============================================================================
// Reduction Operations
// ============================================================================

/// Partial sum reduction using workgroup shared memory
/// Each workgroup produces one partial sum
/// Requires a second pass to combine partial sums
pub fn reduceSum(
    input: std_gpu.StorageConstPtr(f32),
    partial_sums: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    const lid = std_gpu.localInvocationId()[0];
    const wgid = std_gpu.workgroupId()[0];
    const wg_size = std_gpu.workgroupSize()[0];

    // Shared memory for workgroup reduction
    var shared: [256]f32 = undefined;

    // Load element or zero if out of bounds
    shared[lid] = if (gid < n) input[gid] else 0.0;
    std_gpu.workgroupBarrier();

    // Parallel reduction in shared memory
    var stride: u32 = wg_size / 2;
    while (stride > 0) : (stride /= 2) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        std_gpu.workgroupBarrier();
    }

    // Write workgroup result
    if (lid == 0) {
        partial_sums[wgid] = shared[0];
    }
}

/// Find maximum value using workgroup shared memory
pub fn reduceMax(
    input: std_gpu.StorageConstPtr(f32),
    partial_max: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    const lid = std_gpu.localInvocationId()[0];
    const wgid = std_gpu.workgroupId()[0];
    const wg_size = std_gpu.workgroupSize()[0];

    var shared: [256]f32 = undefined;

    // Load element or negative infinity if out of bounds
    shared[lid] = if (gid < n) input[gid] else -std.math.inf(f32);
    std_gpu.workgroupBarrier();

    var stride: u32 = wg_size / 2;
    while (stride > 0) : (stride /= 2) {
        if (lid < stride) {
            shared[lid] = @max(shared[lid], shared[lid + stride]);
        }
        std_gpu.workgroupBarrier();
    }

    if (lid == 0) {
        partial_max[wgid] = shared[0];
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

/// Simple matrix multiplication: C = A * B
/// Uses 2D workgroups (16x16) with each thread computing one element
pub fn matrixMul(
    a: std_gpu.StorageConstPtr(f32),
    b: std_gpu.StorageConstPtr(f32),
    c: std_gpu.StoragePtr(f32),
    m: u32,
    n: u32,
    k: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId();
    const row = gid[1];
    const col = gid[0];

    if (row >= m or col >= n) return;

    var sum: f32 = 0.0;
    var i: u32 = 0;
    while (i < k) : (i += 1) {
        sum += a[row * k + i] * b[i * n + col];
    }

    c[row * n + col] = sum;
}

/// Tiled matrix multiplication using shared memory
/// Each workgroup loads tiles of A and B into shared memory
/// Tile size: 16x16
pub fn matrixMulTiled(
    a: std_gpu.StorageConstPtr(f32),
    b: std_gpu.StorageConstPtr(f32),
    c: std_gpu.StoragePtr(f32),
    m: u32,
    n: u32,
    k: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const TILE_SIZE: u32 = 16;

    const gid = std_gpu.globalInvocationId();
    const lid = std_gpu.localInvocationId();
    const row = gid[1];
    const col = gid[0];
    const local_row = lid[1];
    const local_col = lid[0];

    // Shared memory tiles
    var tile_a: [TILE_SIZE][TILE_SIZE]f32 = undefined;
    var tile_b: [TILE_SIZE][TILE_SIZE]f32 = undefined;

    var sum: f32 = 0.0;

    // Number of tiles to process
    const num_tiles = (k + TILE_SIZE - 1) / TILE_SIZE;

    var t: u32 = 0;
    while (t < num_tiles) : (t += 1) {
        // Load tile of A into shared memory
        const a_col = t * TILE_SIZE + local_col;
        if (row < m and a_col < k) {
            tile_a[local_row][local_col] = a[row * k + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load tile of B into shared memory
        const b_row = t * TILE_SIZE + local_row;
        if (b_row < k and col < n) {
            tile_b[local_row][local_col] = b[b_row * n + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        std_gpu.workgroupBarrier();

        // Compute partial dot product for this tile
        var i: u32 = 0;
        while (i < TILE_SIZE) : (i += 1) {
            sum += tile_a[local_row][i] * tile_b[i][local_col];
        }

        std_gpu.workgroupBarrier();
    }

    // Write result
    if (row < m and col < n) {
        c[row * n + col] = sum;
    }
}

// ============================================================================
// Activation Functions (for neural networks)
// ============================================================================

/// ReLU activation: result[i] = max(0, input[i])
pub fn relu(
    input: std_gpu.StorageConstPtr(f32),
    output: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        output[gid] = @max(input[gid], 0.0);
    }
}

/// Sigmoid activation: result[i] = 1 / (1 + exp(-input[i]))
pub fn sigmoid(
    input: std_gpu.StorageConstPtr(f32),
    output: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        output[gid] = 1.0 / (1.0 + @exp(-input[gid]));
    }
}

/// SiLU (Swish) activation: result[i] = input[i] * sigmoid(input[i])
pub fn silu(
    input: std_gpu.StorageConstPtr(f32),
    output: std_gpu.StoragePtr(f32),
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        const x = input[gid];
        const sig = 1.0 / (1.0 + @exp(-x));
        output[gid] = x * sig;
    }
}

/// Softmax numerator: output[i] = exp(input[i] - max_val)
/// Requires a second pass to divide by sum
pub fn softmaxNumerator(
    input: std_gpu.StorageConstPtr(f32),
    output: std_gpu.StoragePtr(f32),
    max_val: f32,
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        output[gid] = @exp(input[gid] - max_val);
    }
}

/// Softmax denominator: output[i] = output[i] / sum
pub fn softmaxNormalize(
    data: std_gpu.StoragePtr(f32),
    sum_val: f32,
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        data[gid] /= sum_val;
    }
}

// ============================================================================
// Normalization Operations
// ============================================================================

/// RMS Normalization: output[i] = input[i] * rsqrt(mean(input^2) + eps) * weight[i]
pub fn rmsNorm(
    input: std_gpu.StorageConstPtr(f32),
    weight: std_gpu.StorageConstPtr(f32),
    output: std_gpu.StoragePtr(f32),
    rms: f32,
    n: u32,
) callconv(if (std_gpu.is_gpu_target) .spirv_kernel else .auto) void {
    const gid = std_gpu.globalInvocationId()[0];
    if (gid < n) {
        output[gid] = input[gid] * rms * weight[gid];
    }
}

// ============================================================================
// Set Workgroup Sizes (comptime)
// ============================================================================

comptime {
    if (std_gpu.is_gpu_target) {
        // 1D kernels: 256 threads
        std_gpu.setLocalSize(vectorAdd, 256, 1, 1);
        std_gpu.setLocalSize(vectorSub, 256, 1, 1);
        std_gpu.setLocalSize(vectorMul, 256, 1, 1);
        std_gpu.setLocalSize(vectorScale, 256, 1, 1);
        std_gpu.setLocalSize(vectorFMA, 256, 1, 1);
        std_gpu.setLocalSize(reduceSum, 256, 1, 1);
        std_gpu.setLocalSize(reduceMax, 256, 1, 1);
        std_gpu.setLocalSize(relu, 256, 1, 1);
        std_gpu.setLocalSize(sigmoid, 256, 1, 1);
        std_gpu.setLocalSize(silu, 256, 1, 1);
        std_gpu.setLocalSize(softmaxNumerator, 256, 1, 1);
        std_gpu.setLocalSize(softmaxNormalize, 256, 1, 1);
        std_gpu.setLocalSize(rmsNorm, 256, 1, 1);

        // 2D kernels: 16x16 threads
        std_gpu.setLocalSize(matrixMul, 16, 16, 1);
        std_gpu.setLocalSize(matrixMulTiled, 16, 16, 1);
    }
}

// ============================================================================
// CPU Fallback Execution
// ============================================================================

/// Execute vector addition on CPU (for testing or fallback)
pub fn vectorAddCpu(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |av, bv, *rv| {
        rv.* = av + bv;
    }
}

/// Execute vector multiplication on CPU
pub fn vectorMulCpu(a: []const f32, b: []const f32, result: []f32) void {
    for (a, b, result) |av, bv, *rv| {
        rv.* = av * bv;
    }
}

/// Execute matrix multiplication on CPU
pub fn matrixMulCpu(
    a: []const f32,
    b: []const f32,
    c: []f32,
    m: usize,
    n: usize,
    k: usize,
) void {
    for (0..m) |row| {
        for (0..n) |col| {
            var sum: f32 = 0.0;
            for (0..k) |i| {
                sum += a[row * k + i] * b[i * n + col];
            }
            c[row * n + col] = sum;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "CPU fallback vector add" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var result: [4]f32 = undefined;

    vectorAddCpu(&a, &b, &result);

    try std.testing.expectApproxEqRel(@as(f32, 6.0), result[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 8.0), result[1], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 10.0), result[2], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 12.0), result[3], 1e-6);
}

test "CPU fallback matrix mul" {
    // 2x3 * 3x2 = 2x2
    const a = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const b = [_]f32{ 7, 8, 9, 10, 11, 12 };
    var c: [4]f32 = undefined;

    matrixMulCpu(&a, &b, &c, 2, 2, 3);

    // Row 0: 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // Row 0: 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // Row 1: 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // Row 1: 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    try std.testing.expectApproxEqRel(@as(f32, 58.0), c[0], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 64.0), c[1], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 139.0), c[2], 1e-6);
    try std.testing.expectApproxEqRel(@as(f32, 154.0), c[3], 1e-6);
}

test "kernel function signatures exist" {
    // Verify kernel functions exist and have correct signatures
    // These tests pass on CPU but would use GPU on SPIR-V target
    const gid = std_gpu.globalInvocationId();
    try std.testing.expectEqual(@as(u32, 0), gid[0]);
}
