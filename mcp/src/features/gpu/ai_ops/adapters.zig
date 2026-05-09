//! AiOps Adapter Helpers
//!
//! Utilities for creating AiOps wrappers from concrete implementation types.

const std = @import("std");
const ai_ops = @import("../ai_ops.zig");

const AiOps = ai_ops.AiOps;
const AiOpsError = ai_ops.AiOpsError;
const DeviceBuffer = ai_ops.DeviceBuffer;
const Transpose = ai_ops.Transpose;

/// Helper to cast an opaque pointer back to its concrete implementation type.
inline fn aiOpsImplCast(comptime T: type, ptr: *anyopaque) *T {
    return @ptrCast(@alignCast(ptr));
}

/// Create an AiOps wrapper from a concrete implementation type.
///
/// The Impl type must provide methods matching the AiOps.VTable interface
/// but with the first parameter being `*Impl` instead of `*anyopaque`.
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
            const self = aiOpsImplCast(Impl, ptr);
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
            const self = aiOpsImplCast(Impl, ptr);
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
            const self = aiOpsImplCast(Impl, ptr);
            return self.softmax(data, len, stream);
        }

        fn rmsnorm(ptr: *anyopaque, x: *anyopaque, weight: *const anyopaque, len: u32, eps: f32, stream: ?*anyopaque) AiOpsError!void {
            const self = aiOpsImplCast(Impl, ptr);
            return self.rmsnorm(x, weight, len, eps, stream);
        }

        fn silu(ptr: *anyopaque, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self = aiOpsImplCast(Impl, ptr);
            return self.silu(data, len, stream);
        }

        fn gelu(ptr: *anyopaque, data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self = aiOpsImplCast(Impl, ptr);
            return self.gelu(data, len, stream);
        }

        fn scale(ptr: *anyopaque, data: *anyopaque, scalar: f32, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self = aiOpsImplCast(Impl, ptr);
            return self.scale(data, scalar, len, stream);
        }

        fn elementwiseMul(ptr: *anyopaque, a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self = aiOpsImplCast(Impl, ptr);
            return self.elementwiseMul(a, b, len, stream);
        }

        fn elementwiseAdd(ptr: *anyopaque, a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            const self = aiOpsImplCast(Impl, ptr);
            return self.elementwiseAdd(a, b, len, stream);
        }

        fn allocDevice(ptr: *anyopaque, allocator: std.mem.Allocator, size: usize) AiOpsError!DeviceBuffer {
            const self = aiOpsImplCast(Impl, ptr);
            return self.allocDevice(allocator, size);
        }

        fn copyToDevice(ptr: *anyopaque, dst: *anyopaque, src: [*]const u8, len: usize) AiOpsError!void {
            const self = aiOpsImplCast(Impl, ptr);
            return self.copyToDevice(dst, src, len);
        }

        fn copyFromDevice(ptr: *anyopaque, dst: [*]u8, src: *const anyopaque, len: usize) AiOpsError!void {
            const self = aiOpsImplCast(Impl, ptr);
            return self.copyFromDevice(dst, src, len);
        }

        fn freeDevice(ptr: *anyopaque, mem: *anyopaque) void {
            const self = aiOpsImplCast(Impl, ptr);
            self.freeDevice(mem);
        }

        fn isAvailable(ptr: *anyopaque) bool {
            const self = aiOpsImplCast(Impl, ptr);
            return self.isAvailable();
        }

        fn deinitFn(ptr: *anyopaque) void {
            const self = aiOpsImplCast(Impl, ptr);
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

test "createAiOps routes through centralized opaque cast helper" {
    const MockImpl = struct {
        scale_called: bool = false,
        is_available_called: bool = false,
        deinit_called: bool = false,
        last_scale_scalar: f32 = 0.0,
        last_scale_len: u32 = 0,

        fn sgemm(
            self: *@This(),
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
            _ = self;
            _ = trans_a;
            _ = trans_b;
            _ = m;
            _ = n;
            _ = k;
            _ = alpha;
            _ = a;
            _ = lda;
            _ = b;
            _ = ldb;
            _ = beta;
            _ = c;
            _ = ldc;
        }

        fn sgemmStridedBatched(
            self: *@This(),
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
            _ = self;
            _ = trans_a;
            _ = trans_b;
            _ = m;
            _ = n;
            _ = k;
            _ = alpha;
            _ = a;
            _ = lda;
            _ = stride_a;
            _ = b;
            _ = ldb;
            _ = stride_b;
            _ = beta;
            _ = c;
            _ = ldc;
            _ = stride_c;
            _ = batch_count;
        }

        fn softmax(self: *@This(), data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            _ = self;
            _ = data;
            _ = len;
            _ = stream;
        }

        fn rmsnorm(self: *@This(), x: *anyopaque, weight: *const anyopaque, len: u32, eps: f32, stream: ?*anyopaque) AiOpsError!void {
            _ = self;
            _ = x;
            _ = weight;
            _ = len;
            _ = eps;
            _ = stream;
        }

        fn silu(self: *@This(), data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            _ = self;
            _ = data;
            _ = len;
            _ = stream;
        }

        fn gelu(self: *@This(), data: *anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            _ = self;
            _ = data;
            _ = len;
            _ = stream;
        }

        fn scale(self: *@This(), data: *anyopaque, scalar: f32, len: u32, stream: ?*anyopaque) AiOpsError!void {
            _ = data;
            _ = stream;
            self.scale_called = true;
            self.last_scale_scalar = scalar;
            self.last_scale_len = len;
        }

        fn elementwiseMul(self: *@This(), a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            _ = self;
            _ = a;
            _ = b;
            _ = len;
            _ = stream;
        }

        fn elementwiseAdd(self: *@This(), a: *anyopaque, b: *const anyopaque, len: u32, stream: ?*anyopaque) AiOpsError!void {
            _ = self;
            _ = a;
            _ = b;
            _ = len;
            _ = stream;
        }

        fn allocDevice(self: *@This(), allocator: std.mem.Allocator, size: usize) AiOpsError!DeviceBuffer {
            _ = self;
            _ = allocator;
            _ = size;
            return error.NotSupported;
        }

        fn copyToDevice(self: *@This(), dst: *anyopaque, src: [*]const u8, len: usize) AiOpsError!void {
            _ = self;
            _ = dst;
            _ = src;
            _ = len;
        }

        fn copyFromDevice(self: *@This(), dst: [*]u8, src: *const anyopaque, len: usize) AiOpsError!void {
            _ = self;
            _ = dst;
            _ = src;
            _ = len;
        }

        fn freeDevice(self: *@This(), mem: *anyopaque) void {
            _ = self;
            _ = mem;
        }

        fn isAvailable(self: *@This()) bool {
            self.is_available_called = true;
            return true;
        }

        fn deinit(self: *@This()) void {
            self.deinit_called = true;
        }
    };

    var impl = MockImpl{};
    const ops = createAiOps(MockImpl, &impl);

    var dummy: u32 = 0;
    try ops.vtable.scale(ops.ptr, &dummy, 0.25, 1, null);
    try std.testing.expect(impl.scale_called);
    try std.testing.expectEqual(@as(f32, 0.25), impl.last_scale_scalar);
    try std.testing.expectEqual(@as(u32, 1), impl.last_scale_len);

    try std.testing.expect(ops.vtable.isAvailable(ops.ptr));
    try std.testing.expect(impl.is_available_called);

    ops.vtable.deinit(ops.ptr);
    try std.testing.expect(impl.deinit_called);
}
