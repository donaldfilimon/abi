//! CPU Fallback AI Operations
//!
//! Provides a CPU-based implementation of the AiOps VTable interface.
//! Performs real computations on host memory using standard math.
//! Pointers are interpreted as f32 arrays laid out in row-major order.

const std = @import("std");
const ai_ops = @import("../ai_ops.zig");

const AiOps = ai_ops.AiOps;
const AiOpsError = ai_ops.AiOpsError;
const DeviceBuffer = ai_ops.DeviceBuffer;
const Transpose = ai_ops.Transpose;

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
        const n_val: usize = @intCast(len);
        const data: [*]f32 = @ptrCast(@alignCast(data_ptr));

        // Find max for numerical stability
        var max_val: f32 = data[0];
        for (1..n_val) |i| {
            if (data[i] > max_val) max_val = data[i];
        }

        // Exponentiate and sum
        var sum: f32 = 0.0;
        for (0..n_val) |i| {
            data[i] = @exp(data[i] - max_val);
            sum += data[i];
        }

        // Normalize
        const inv_sum = 1.0 / sum;
        for (0..n_val) |i| {
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
        const n_val: usize = @intCast(len);
        const x: [*]f32 = @ptrCast(@alignCast(x_ptr));
        const weight: [*]const f32 = @ptrCast(@alignCast(weight_ptr));

        // Compute mean of squares
        var sum_sq: f32 = 0.0;
        for (0..n_val) |i| {
            sum_sq += x[i] * x[i];
        }
        const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(n_val)) + eps);
        const inv_rms = 1.0 / rms;

        // Normalize and apply weight
        for (0..n_val) |i| {
            x[i] = x[i] * inv_rms * weight[i];
        }
    }

    /// CPU SiLU: x = x * sigmoid(x) = x / (1 + exp(-x)).
    fn cpuSilu(_: *anyopaque, data_ptr: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n_val: usize = @intCast(len);
        const data: [*]f32 = @ptrCast(@alignCast(data_ptr));

        for (0..n_val) |i| {
            const val = data[i];
            data[i] = val / (1.0 + @exp(-val));
        }
    }

    /// CPU GELU: x = x * 0.5 * (1 + erf(x / sqrt(2))).
    /// Uses tanh approximation: GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    fn cpuGelu(_: *anyopaque, data_ptr: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n_val: usize = @intCast(len);
        const data: [*]f32 = @ptrCast(@alignCast(data_ptr));
        const sqrt_2_over_pi: f32 = 0.7978845608028654; // sqrt(2/pi)

        for (0..n_val) |i| {
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
        const n_val: usize = @intCast(len);
        const data: [*]f32 = @ptrCast(@alignCast(data_ptr));

        for (0..n_val) |i| {
            data[i] *= scalar;
        }
    }

    /// CPU element-wise multiply: a = a * b.
    fn cpuElementwiseMul(_: *anyopaque, a_ptr: *anyopaque, b_ptr: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n_val: usize = @intCast(len);
        const a: [*]f32 = @ptrCast(@alignCast(a_ptr));
        const b: [*]const f32 = @ptrCast(@alignCast(b_ptr));

        for (0..n_val) |i| {
            a[i] *= b[i];
        }
    }

    /// CPU element-wise add: a = a + b.
    fn cpuElementwiseAdd(_: *anyopaque, a_ptr: *anyopaque, b_ptr: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const n_val: usize = @intCast(len);
        const a: [*]f32 = @ptrCast(@alignCast(a_ptr));
        const b: [*]const f32 = @ptrCast(@alignCast(b_ptr));

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
