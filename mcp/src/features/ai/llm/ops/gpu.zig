//! GPU-accelerated LLM operations.
//!
//! Provides unified GPU inference path for LLM operations with automatic
//! fallback to CPU when GPU is unavailable. Supports backend-agnostic
//! AiOps interface (Metal/MPS on Darwin, CUDA on Linux/Windows).

const std = @import("std");
const time = @import("../../../../foundation/mod.zig").time;
const build_options = @import("build_options");
const matmul = @import("matmul.zig");
const attention = @import("attention.zig");
const rmsnorm = @import("rmsnorm.zig");
const activations = @import("activations.zig");

// Centralized GPU interface
const gpu_feature = @import("../../../gpu/mod.zig");
const ai_ops = gpu_feature.ai_ops;
const AiOps = ai_ops.AiOps;

/// GPU operation context for LLM inference.
pub const GpuOpsContext = struct {
    allocator: std.mem.Allocator,
    ops: ?AiOps = null,
    stats: GpuStats = .{},

    pub fn init(allocator: std.mem.Allocator) GpuOpsContext {
        if (!build_options.feat_gpu) return .{ .allocator = allocator };

        var ops: ?AiOps = null;

        if (@import("builtin").os.tag == .macos) {
            if (ai_ops.MacosAiOps.init(allocator)) |macos| {
                ops = macos.asAiOps().*;
            } else |err| {
                std.log.warn("MacosAiOps init failed: {any}", .{err});
            }
        } else if (build_options.gpu_cuda) {
            // CUDA AI ops will use the same VTable pattern as the macOS backend.
        }

        return .{
            .allocator = allocator,
            .ops = ops,
        };
    }

    pub fn deinit(self: *GpuOpsContext) void {
        if (self.ops) |ops| {
            // We need to free the AiOps struct itself if we allocated it
            // but in GpuOpsContext we just store it by value (copied from asAiOps().*)
            // The underlying implementation deinit should be called.
            ops.deinit();
        }
        self.* = undefined;
    }

    pub fn isGpuAvailable(self: *const GpuOpsContext) bool {
        if (self.ops) |ops| return ops.isAvailable();
        return false;
    }

    pub fn matrixMultiply(
        self: *GpuOpsContext,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: u32,
        k: u32,
        n: u32,
    ) void {
        var timer = time.Timer.start() catch null;
        if (self.ops) |ops| {
            ops.sgemm(
                .no_trans,
                .no_trans,
                @intCast(m),
                @intCast(n),
                @intCast(k),
                1.0,
                @ptrCast(a.ptr),
                @intCast(k),
                @ptrCast(b.ptr),
                @intCast(n),
                0.0,
                @ptrCast(c.ptr),
                @intCast(n),
            ) catch {
                matmul.matrixMultiply(a, b, c, m, k, n);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            matmul.matrixMultiply(a, b, c, m, k, n);
            self.stats.addOp(if (timer) |*t| t.read() else 0, false);
        }
    }

    pub fn rmsNorm(self: *GpuOpsContext, x: []f32, weight: []const f32, eps: f32) void {
        var timer = time.Timer.start() catch null;
        if (self.ops) |ops| {
            ops.rmsnorm(@ptrCast(x.ptr), @ptrCast(weight.ptr), @intCast(x.len), eps, null) catch {
                rmsnorm.rmsNormInPlace(x, weight, eps);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            rmsnorm.rmsNormInPlace(x, weight, eps);
            self.stats.addOp(if (timer) |*t| t.read() else 0, false);
        }
    }

    pub fn softmax(self: *GpuOpsContext, x: []f32) void {
        var timer = time.Timer.start() catch null;
        if (self.ops) |ops| {
            ops.softmax(@ptrCast(x.ptr), @intCast(x.len), null) catch {
                activations.softmaxInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            activations.softmaxInPlace(x);
            self.stats.addOp(if (timer) |*t| t.read() else 0, false);
        }
    }

    pub fn silu(self: *GpuOpsContext, x: []f32) void {
        var timer = time.Timer.start() catch null;
        if (self.ops) |ops| {
            ops.silu(@ptrCast(x.ptr), @intCast(x.len), null) catch {
                activations.siluInPlace(x);
                self.stats.addOp(if (timer) |*t| t.read() else 0, false);
                return;
            };
            self.stats.addOp(if (timer) |*t| t.read() else 0, true);
        } else {
            activations.siluInPlace(x);
            self.stats.addOp(if (timer) |*t| t.read() else 0, false);
        }
    }
};

pub const GpuStats = struct {
    total_ops: u64 = 0,
    total_time_ns: u64 = 0,
    fallback_ops: u64 = 0,

    pub fn addOp(self: *GpuStats, time_ns: u64, used_gpu: bool) void {
        self.total_ops += 1;
        self.total_time_ns += time_ns;
        if (!used_gpu) self.fallback_ops += 1;
    }

    pub fn gpuUtilization(self: GpuStats) f64 {
        if (self.total_ops == 0) return 0;
        return 1.0 - (@as(f64, @floatFromInt(self.fallback_ops)) / @as(f64, @floatFromInt(self.total_ops)));
    }
};

pub fn createContext(allocator: std.mem.Allocator) GpuOpsContext {
    return GpuOpsContext.init(allocator);
}

test "gpu ops context init" {
    const allocator = std.testing.allocator;
    var ctx = GpuOpsContext.init(allocator);
    defer ctx.deinit();

    var a = [_]f32{ 1, 2, 3, 4 };
    var b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };
    ctx.matrixMultiply(&a, &b, &c, 2, 2, 2);
}

test {
    std.testing.refAllDecls(@This());
}
