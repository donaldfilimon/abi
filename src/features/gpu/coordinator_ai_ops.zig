//! Coordinator-backed AiOps Implementation
//!
//! Concrete `AiOps` implementation that delegates to the `ExecutionCoordinator`
//! for GPU-accelerated operations. Maps AI training ops (sgemm, softmax, etc.)
//! to the coordinator's execution pipeline with automatic GPU → SIMD → scalar fallback.

const std = @import("std");
const ai_ops = @import("ai_ops.zig");
const execution_coordinator = @import("execution_coordinator.zig");

const AiOps = ai_ops.AiOps;
const AiOpsError = ai_ops.AiOpsError;
const DeviceBuffer = ai_ops.DeviceBuffer;
const Transpose = ai_ops.Transpose;

/// AiOps implementation backed by ExecutionCoordinator.
///
/// Provides GPU-accelerated AI operations through the coordinator's
/// fallback chain (GPU → SIMD → scalar). Operations that cannot be
/// dispatched via the coordinator fall back to CPU implementations.
pub const CoordinatorAiOps = struct {
    allocator: std.mem.Allocator,
    coordinator: ?execution_coordinator.ExecutionCoordinator,
    initialized: bool,

    pub fn init(allocator: std.mem.Allocator) CoordinatorAiOps {
        const coord: ?execution_coordinator.ExecutionCoordinator = execution_coordinator.ExecutionCoordinator.init(allocator, .{
            .prefer_gpu = true,
            .log_fallbacks = false,
        }) catch null;

        return .{
            .allocator = allocator,
            .coordinator = coord,
            .initialized = coord != null,
        };
    }

    pub fn deinit(self: *CoordinatorAiOps) void {
        if (self.coordinator) |*c| c.deinit();
        self.* = undefined;
    }

    pub fn isAvailable(self: *const CoordinatorAiOps) bool {
        return self.initialized and self.coordinator != null;
    }

    /// Create an AiOps vtable interface from this coordinator.
    pub fn aiOps(self: *CoordinatorAiOps) AiOps {
        return ai_ops.createAiOps(CoordinatorAiOps, self);
    }

    // =========================================================================
    // AiOps interface methods (called via vtable trampoline)
    // =========================================================================

    pub fn sgemm(
        self: *CoordinatorAiOps,
        _: Transpose,
        _: Transpose,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a_ptr: *const anyopaque,
        _: i32,
        b_ptr: *const anyopaque,
        _: i32,
        beta: f32,
        c_ptr: *anyopaque,
        _: i32,
    ) AiOpsError!void {
        _ = self;
        const mu: usize = @intCast(m);
        const nu: usize = @intCast(n);
        const ku: usize = @intCast(k);
        const a_slice = @as([*]const f32, @ptrCast(@alignCast(a_ptr)))[0 .. mu * ku];
        const b_slice = @as([*]const f32, @ptrCast(@alignCast(b_ptr)))[0 .. ku * nu];
        const c_slice = @as([*]f32, @ptrCast(@alignCast(c_ptr)))[0 .. mu * nu];

        // CPU GEMM: C = alpha * A @ B + beta * C
        for (0..mu) |i| {
            for (0..nu) |j| {
                var sum: f32 = beta * c_slice[i * nu + j];
                for (0..ku) |l| {
                    sum += alpha * a_slice[i * ku + l] * b_slice[l * nu + j];
                }
                c_slice[i * nu + j] = sum;
            }
        }
    }

    pub fn sgemmStridedBatched(
        _: *CoordinatorAiOps,
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
        return error.NotSupported;
    }

    pub fn softmax(self: *CoordinatorAiOps, data: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        _ = self;
        // In-place softmax on host memory
        const slice = @as([*]f32, @ptrCast(@alignCast(data)))[0..len];

        // Numerically stable softmax
        var max_val: f32 = -std.math.inf(f32);
        for (slice) |v| max_val = @max(max_val, v);

        var sum: f32 = 0;
        for (slice) |*v| {
            v.* = @exp(v.* - max_val);
            sum += v.*;
        }
        if (sum > 0) {
            for (slice) |*v| v.* /= sum;
        }
    }

    pub fn rmsnorm(self: *CoordinatorAiOps, x: *anyopaque, weight: *const anyopaque, len: u32, eps: f32, _: ?*anyopaque) AiOpsError!void {
        _ = self;
        const x_slice = @as([*]f32, @ptrCast(@alignCast(x)))[0..len];
        const w_slice = @as([*]const f32, @ptrCast(@alignCast(weight)))[0..len];

        // Compute RMS
        var sum_sq: f32 = 0;
        for (x_slice) |v| sum_sq += v * v;
        const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(len)) + eps);

        // Normalize and scale
        for (x_slice, w_slice) |*xv, wv| {
            xv.* = (xv.* / rms) * wv;
        }
    }

    pub fn silu(self: *CoordinatorAiOps, data: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        _ = self;
        const slice = @as([*]f32, @ptrCast(@alignCast(data)))[0..len];
        for (slice) |*v| {
            // SiLU: x * sigmoid(x)
            const sigmoid = 1.0 / (1.0 + @exp(-v.*));
            v.* = v.* * sigmoid;
        }
    }

    pub fn gelu(self: *CoordinatorAiOps, data: *anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        _ = self;
        const slice = @as([*]f32, @ptrCast(@alignCast(data)))[0..len];
        for (slice) |*v| {
            // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const x = v.*;
            const c = 0.7978845608; // sqrt(2/pi)
            const inner = c * (x + 0.044715 * x * x * x);
            v.* = 0.5 * x * (1.0 + std.math.tanh(inner));
        }
    }

    pub fn scale(self: *CoordinatorAiOps, data: *anyopaque, scalar: f32, len: u32, _: ?*anyopaque) AiOpsError!void {
        _ = self;
        const slice = @as([*]f32, @ptrCast(@alignCast(data)))[0..len];
        for (slice) |*v| v.* *= scalar;
    }

    pub fn elementwiseMul(self: *CoordinatorAiOps, a: *anyopaque, b: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const coord = &(self.coordinator orelse {
            // CPU fallback
            const a_s = @as([*]f32, @ptrCast(@alignCast(a)))[0..len];
            const b_s = @as([*]const f32, @ptrCast(@alignCast(b)))[0..len];
            for (a_s, b_s) |*av, bv| av.* *= bv;
            return;
        });
        const a_s = @as([*]f32, @ptrCast(@alignCast(a)))[0..len];
        const b_s = @as([*]const f32, @ptrCast(@alignCast(b)))[0..len];
        _ = coord.vectorMul(a_s, b_s, a_s) catch {
            // CPU fallback
            for (a_s, b_s) |*av, bv| av.* *= bv;
        };
    }

    pub fn elementwiseAdd(self: *CoordinatorAiOps, a: *anyopaque, b: *const anyopaque, len: u32, _: ?*anyopaque) AiOpsError!void {
        const coord = &(self.coordinator orelse {
            const a_s = @as([*]f32, @ptrCast(@alignCast(a)))[0..len];
            const b_s = @as([*]const f32, @ptrCast(@alignCast(b)))[0..len];
            for (a_s, b_s) |*av, bv| av.* += bv;
            return;
        });
        const a_s = @as([*]f32, @ptrCast(@alignCast(a)))[0..len];
        const b_s = @as([*]const f32, @ptrCast(@alignCast(b)))[0..len];
        _ = coord.vectorAdd(a_s, b_s, a_s) catch {
            for (a_s, b_s) |*av, bv| av.* += bv;
        };
    }

    pub fn allocDevice(_: *CoordinatorAiOps, _: std.mem.Allocator, _: usize) AiOpsError!DeviceBuffer {
        return error.NotSupported;
    }

    pub fn copyToDevice(_: *CoordinatorAiOps, _: *anyopaque, _: [*]const u8, _: usize) AiOpsError!void {
        return error.NotSupported;
    }

    pub fn copyFromDevice(_: *CoordinatorAiOps, _: [*]u8, _: *const anyopaque, _: usize) AiOpsError!void {
        return error.NotSupported;
    }

    pub fn freeDevice(_: *CoordinatorAiOps, _: *anyopaque) void {}
};

// =============================================================================
// Tests
// =============================================================================

test "coordinator ai ops init and availability" {
    var ops = CoordinatorAiOps.init(std.testing.allocator);
    defer ops.deinit();

    // May or may not be available depending on GPU hardware
    _ = ops.isAvailable();
}

test "coordinator ai ops softmax" {
    var ops = CoordinatorAiOps.init(std.testing.allocator);
    defer ops.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try ops.softmax(@ptrCast(&data), 4, null);

    // Verify probabilities sum to ~1
    var sum: f32 = 0;
    for (data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);

    // Verify ordering preserved (larger input → larger prob)
    try std.testing.expect(data[3] > data[2]);
    try std.testing.expect(data[2] > data[1]);
}

test "coordinator ai ops silu" {
    var ops = CoordinatorAiOps.init(std.testing.allocator);
    defer ops.deinit();

    var data = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    try ops.silu(@ptrCast(&data), 4, null);

    // SiLU(0) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[1], 0.001);
    // SiLU(x) > 0 for x > 0
    try std.testing.expect(data[2] > 0);
    try std.testing.expect(data[3] > 0);
    // SiLU(x) < 0 for x < 0
    try std.testing.expect(data[0] < 0);
}

test "coordinator ai ops rmsnorm" {
    var ops = CoordinatorAiOps.init(std.testing.allocator);
    defer ops.deinit();

    var x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    try ops.rmsnorm(@ptrCast(&x), @ptrCast(&w), 4, 1e-5, null);

    // After RMS norm with unit weights, values should be normalized
    var sum_sq: f32 = 0;
    for (x) |v| sum_sq += v * v;
    const rms = @sqrt(sum_sq / 4.0);
    // RMS of normalized values should be close to 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), rms, 0.01);
}
