//! CPU Fallback AI Operations
//!
//! Provides a CPU-based implementation of the AiOps VTable interface.
//! Performs real computations on host memory using standard math.
//! Pointers are interpreted as f32 arrays laid out in row-major order.

const std = @import("std");
const ai_ops = @import("../ai_ops.zig");
const activations = @import("../../../foundation/mod.zig").simd.activations;

const AiOps = ai_ops.AiOps;
const AiOpsError = ai_ops.AiOpsError;
const DeviceBuffer = ai_ops.DeviceBuffer;
const Transpose = ai_ops.Transpose;

inline fn ptrToF32Slice(ptr: *anyopaque) [*]f32 {
    return @ptrCast(@alignCast(ptr));
}

inline fn ptrToConstF32Slice(ptr: *const anyopaque) [*]const f32 {
    return @ptrCast(@alignCast(ptr));
}

inline fn ptrToU8Slice(ptr: *anyopaque) [*]u8 {
    return @ptrCast(@alignCast(ptr));
}

inline fn ptrToConstU8Slice(ptr: *const anyopaque) [*]const u8 {
    return @ptrCast(@alignCast(ptr));
}

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

        const a = ptrToConstF32Slice(a_ptr);
        const b = ptrToConstF32Slice(b_ptr);
        const c = ptrToF32Slice(c_ptr);

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

        const a_base = ptrToConstF32Slice(a_ptr);
        const b_base = ptrToConstF32Slice(b_ptr);
        const c_base = ptrToF32Slice(c_ptr);

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

    /// CPU softmax: delegates to SIMD-accelerated foundation implementation.
    fn cpuSoftmax(_: *anyopaque, data_ptr: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        if (len == 0) return;
        const n_val: usize = @intCast(len);
        const data = ptrToF32Slice(data_ptr);
        activations.softmaxInPlace(data[0..n_val]);
    }

    /// CPU rmsnorm: delegates to SIMD-accelerated foundation implementation.
    fn cpuRmsnorm(
        _: *anyopaque,
        x_ptr: *anyopaque,
        weight_ptr: *const anyopaque,
        len: u32,
        eps: f32,
        _: ?*anyopaque,
    ) AiOpsError!void {
        if (len == 0) return;
        const n_val: usize = @intCast(len);
        const x = ptrToF32Slice(x_ptr);
        const weight = ptrToConstF32Slice(weight_ptr);
        activations.rmsNormInPlace(x[0..n_val], weight[0..n_val], eps);
    }

    /// CPU SiLU: delegates to SIMD-accelerated foundation implementation.
    fn cpuSilu(_: *anyopaque, data_ptr: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n_val: usize = @intCast(len);
        const data = ptrToF32Slice(data_ptr);
        activations.siluInPlace(data[0..n_val]);
    }

    /// CPU GELU: delegates to SIMD-accelerated foundation implementation.
    fn cpuGelu(_: *anyopaque, data_ptr: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n_val: usize = @intCast(len);
        const data = ptrToF32Slice(data_ptr);
        activations.geluInPlace(data[0..n_val]);
    }

    /// CPU scale: x = x * scalar.
    fn cpuScale(_: *anyopaque, data_ptr: *anyopaque, scalar: f32, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n_val: usize = @intCast(len);
        const data = ptrToF32Slice(data_ptr);

        for (0..n_val) |i| {
            data[i] *= scalar;
        }
    }

    /// CPU element-wise multiply: a = a * b.
    fn cpuElementwiseMul(_: *anyopaque, a_ptr: *anyopaque, b_ptr: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n_val: usize = @intCast(len);
        const a = ptrToF32Slice(a_ptr);
        const b = ptrToConstF32Slice(b_ptr);

        for (0..n_val) |i| {
            a[i] *= b[i];
        }
    }

    /// CPU element-wise add: a = a + b.
    fn cpuElementwiseAdd(_: *anyopaque, a_ptr: *anyopaque, b_ptr: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n_val: usize = @intCast(len);
        const a = ptrToF32Slice(a_ptr);
        const b = ptrToConstF32Slice(b_ptr);

        for (0..n_val) |i| {
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
        const dst_slice = ptrToU8Slice(dst);
        @memcpy(dst_slice[0..len], src[0..len]);
    }

    /// Copy from "device" (host-backed) to host.
    fn cpuCopyFromDevice(_: *anyopaque, dst: [*]u8, src: *const anyopaque, len: usize) AiOpsError!void {
        const src_slice = ptrToConstU8Slice(src);
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
