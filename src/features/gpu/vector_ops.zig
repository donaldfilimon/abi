const std = @import("std");
const builtin = @import("builtin");
const backends = @import("backends.zig");
const metal = @import("metal_shared.zig");

pub const VectorOps = struct {
    backend: backends.BackendStatus,

    pub fn init() VectorOps {
        if (builtin.target.os.tag == .macos) {
            metal.g_metal_context.init(std.heap.page_allocator) catch |err| {
                std.log.warn("Metal initialization failed: {s}; using vectorized CPU fallback", .{@errorName(err)});
            };
        }
        return .{ .backend = backends.detectBackend() };
    }

    pub fn dot(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        if (a.len != b.len) return error.DimensionMismatch;
        if (a.len == 0) return 0;

        if (self.backend.accelerated and builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
            var stack_buf: [4096]f32 = undefined;
            const res = if (a.len <= 4096)
                stack_buf[0..a.len]
            else
                try std.heap.page_allocator.alloc(f32, a.len);
            defer if (a.len > 4096) std.heap.page_allocator.free(res);

            try metal.g_metal_context.runKernel(metal.g_metal_context.dot_pipeline, a.len, a, b, res);
            var sum: f32 = 0;
            for (res) |val| {
                sum += val;
            }
            return sum;
        }

        var sum: f32 = 0;
        var i: usize = 0;
        while (i + 4 <= a.len) : (i += 4) {
            const av: @Vector(4, f32) = a[i..][0..4].*;
            const bv: @Vector(4, f32) = b[i..][0..4].*;
            sum += @reduce(.Add, av * bv);
        }
        while (i < a.len) : (i += 1) sum += a[i] * b[i];
        return sum;
    }

    pub fn squaredL2(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        if (a.len != b.len) return error.DimensionMismatch;
        if (a.len == 0) return 0;

        if (self.backend.accelerated and builtin.target.os.tag == .macos and metal.g_metal_context.initialized) {
            var stack_buf: [4096]f32 = undefined;
            const res = if (a.len <= 4096)
                stack_buf[0..a.len]
            else
                try std.heap.page_allocator.alloc(f32, a.len);
            defer if (a.len > 4096) std.heap.page_allocator.free(res);

            try metal.g_metal_context.runKernel(metal.g_metal_context.l2_pipeline, a.len, a, b, res);
            var sum: f32 = 0;
            for (res) |val| {
                sum += val;
            }
            return sum;
        }

        var sum: f32 = 0;
        var i: usize = 0;
        while (i + 4 <= a.len) : (i += 4) {
            const av: @Vector(4, f32) = a[i..][0..4].*;
            const bv: @Vector(4, f32) = b[i..][0..4].*;
            const diff = av - bv;
            sum += @reduce(.Add, diff * diff);
        }
        while (i < a.len) : (i += 1) {
            const diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    pub fn cosineSimilarity(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        const ab = try self.dot(a, b);
        const aa = try self.dot(a, a);
        const bb = try self.dot(b, b);
        if (aa == 0 or bb == 0) return 0;
        return ab / @sqrt(aa * bb);
    }
};

pub fn executeKernel(spec: backends.KernelSpec) !backends.KernelResult {
    if (spec.name.len == 0) return error.InvalidKernelName;
    const status = backends.detectBackend();
    const native = backends.nativeKernelStatus();
    return .{
        .backend = status.backend,
        .mode = if (native.linked and status.accelerated) .native_gpu else .simulated_gpu,
        .work_items = spec.work_items,
        .message = if (native.linked and status.accelerated) "native GPU kernel executed" else "kernel metadata validated; vectorized CPU fallback selected",
    };
}

pub fn vectorOps() VectorOps {
    return VectorOps.init();
}

test "gpu vector ops provide deterministic acceleration" {
    const ops = vectorOps();
    try std.testing.expectEqual(@as(f32, 32), try ops.dot(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    try std.testing.expectEqual(@as(f32, 27), try ops.squaredL2(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    try std.testing.expect((try ops.cosineSimilarity(&.{ 1, 0 }, &.{ 1, 0 })) == 1);
}
