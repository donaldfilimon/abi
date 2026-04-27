//! AiOps Adapter Helpers
//!
//! Utilities for creating AiOps wrappers from concrete implementation types.

const std = @import("std");
const ai_ops = @import("../ai_ops.zig");

const AiOps = ai_ops.AiOps;
const AiOpsError = ai_ops.AiOpsError;
const DeviceBuffer = ai_ops.DeviceBuffer;
const Transpose = ai_ops.Transpose;
const PointerCast = @import("../pointer_cast.zig");

// Centralized pointer-cast helper for AiOps adapters (comptime-parameterized).
pub fn asAiOpsImplPtr(comptime Impl: type, ptr: *anyopaque) *Impl {
    return @ptrCast(*Impl, ptr);
}

// (Moved) Centralized pointer-cast helper will be defined inside the generated struct below.

/// Create an AiOps wrapper from a concrete implementation type.
/// The implementation type must have methods matching the VTable signatures.
pub fn createAiOps(comptime Impl: type, impl: *Impl) AiOps {
    // Helper to cast opaque pointer to concrete Impl (captured by outer scope).
    const implCastHelper = fn (ptr: *anyopaque) *Impl {
        return @ptrCast(*Impl, ptr);
    };
    const gen = struct {
        // Use local helper to cast pointer to Impl
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
            const self: *Impl = implCastHelper(ptr);
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
            const self: *Impl = implCastHelper(ptr);
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
            const self: *Impl = implCastHelper(ptr);
            return self.softmax(data, len, stream);
        }

        fn rmsnorm(ptr: *anyopaque, x: *anyopaque, weight: *const anyopaque, len: u32, eps: f32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = implCastHelper(ptr);
            return self.rmsnorm(x, weight, len, eps, stream);
        }

        fn silu(ptr: *anyopaque, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = implCastHelper(ptr);
            return self.silu(data, len, stream);
        }

        fn gelu(ptr: *anyopaque, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = implCastHelper(ptr);
            return self.gelu(data, len, stream);
        }

        fn scale(ptr: *anyopaque, data: *anyopaque, scalar: f32, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = implCastHelper(ptr);
            return self.scale(data, scalar, len, stream);
        }

        fn elementwiseMul(ptr: *anyopaque, a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = implCastHelper(ptr);
            return self.elementwiseMul(a, b, len, stream);
        }

        fn elementwiseAdd(ptr: *anyopaque, a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self: *Impl = implCastHelper(ptr);
            return self.elementwiseAdd(a, b, len, stream);
        }

        fn allocDevice(ptr: *anyopaque, allocator: std.mem.Allocator, size: usize) AiOpsError!DeviceBuffer {
            const self: *Impl = implCastHelper(ptr);
            return self.allocDevice(allocator, size);
        }

        fn copyToDevice(ptr: *anyopaque, dst: *anyopaque, src: [*]const u8, len: usize) AiOpsError!void {
            const self: *Impl = implCastHelper(ptr);
            return self.copyToDevice(dst, src, len);
        }

        fn copyFromDevice(ptr: *anyopaque, dst: [*]u8, src: *const anyopaque, len: usize) AiOpsError!void {
            const self: *Impl = implCastHelper(ptr);
            return self.copyFromDevice(dst, src, len);
        }

        fn freeDevice(ptr: *anyopaque, mem: *anyopaque) void {
            const self: *Impl = implCastHelper(ptr);
            self.freeDevice(mem);
        }

        fn isAvailable(ptr: *anyopaque) bool {
            const self: *Impl = asAiOpsImplPtr(Impl, ptr);
            return self.isAvailable();
        }

        fn deinitFn(ptr: *anyopaque) void {
            const self: *Impl = asAiOpsImplPtr(Impl, ptr);
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
