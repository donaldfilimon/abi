//! GPU Training Bridge
//!
//! Wraps `AiOps` with automatic CPU fallback and performance tracking.
//! Each operation attempts the GPU path first; on failure, falls back to
//! CPU computation and increments the fallback counter.
//!
//! This bridge is the integration point between the training pipeline
//! (forward/backward passes) and the GPU acceleration layer.

const std = @import("std");
const ai_ops = @import("ai_ops.zig");
const coordinator_ai_ops = @import("coordinator_ai_ops.zig");
const time = @import("../../services/shared/time.zig");

const AiOps = ai_ops.AiOps;
const AiOpsError = ai_ops.AiOpsError;

/// Performance statistics for GPU-accelerated training.
pub const GpuTrainingStats = struct {
    /// Total operations dispatched to GPU
    total_gpu_ops: u64 = 0,
    /// Total time spent in GPU operations (ns)
    gpu_time_ns: u64 = 0,
    /// Operations that fell back to CPU
    cpu_fallback_ops: u64 = 0,
    /// GPU utilization estimate (0.0 - 1.0)
    utilization: f32 = 0,
    /// Name of the active backend
    backend_name: []const u8 = "none",
    /// Whether GPU is actually available
    gpu_available: bool = false,

    /// Compute average kernel time in milliseconds.
    pub fn avgKernelTimeMs(self: GpuTrainingStats) f32 {
        if (self.total_gpu_ops == 0) return 0;
        return @as(f32, @floatFromInt(self.gpu_time_ns)) / @as(f32, @floatFromInt(self.total_gpu_ops)) / 1e6;
    }

    /// Compute GPU vs CPU dispatch ratio.
    pub fn gpuRatio(self: GpuTrainingStats) f32 {
        const total = self.total_gpu_ops + self.cpu_fallback_ops;
        if (total == 0) return 0;
        return @as(f32, @floatFromInt(self.total_gpu_ops)) / @as(f32, @floatFromInt(total));
    }
};

/// GPU Training Bridge — wraps AiOps with fallback and stats.
pub const GpuTrainingBridge = struct {
    gpu_ops: ?coordinator_ai_ops.CoordinatorAiOps,
    gpu_available: bool,
    stats: GpuTrainingStats,
    allocator: std.mem.Allocator,

    /// Initialize the bridge. Attempts to create a CoordinatorAiOps backend.
    /// If GPU initialization fails, the bridge operates in CPU-only mode.
    pub fn init(allocator: std.mem.Allocator) GpuTrainingBridge {
        var ops = coordinator_ai_ops.CoordinatorAiOps.init(allocator);
        const available = ops.isAvailable();

        if (!available) {
            ops.deinit();
            return .{
                .gpu_ops = null,
                .gpu_available = false,
                .stats = .{ .backend_name = "cpu", .gpu_available = false },
                .allocator = allocator,
            };
        }

        return .{
            .gpu_ops = ops,
            .gpu_available = true,
            .stats = .{
                .backend_name = "coordinator",
                .gpu_available = true,
            },
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GpuTrainingBridge) void {
        if (self.gpu_ops) |*ops| ops.deinit();
        self.* = undefined;
    }

    /// Get current performance statistics.
    pub fn getStats(self: *const GpuTrainingBridge) GpuTrainingStats {
        return self.stats;
    }

    // =========================================================================
    // Training Operations (GPU with CPU fallback)
    // =========================================================================

    /// Matrix multiply: C = A @ B (row-major)
    /// Falls back to CPU ops.matmul if GPU unavailable.
    pub fn matmul(
        self: *GpuTrainingBridge,
        a: []const f32,
        b: []const f32,
        c: []f32,
        m: u32,
        n: u32,
        k: u32,
    ) void {
        if (self.gpu_ops) |*ops| {
            const start_ns = getTimeNs();
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
                self.stats.cpu_fallback_ops += 1;
                cpuMatmul(a, b, c, m, n, k);
                return;
            };
            self.stats.total_gpu_ops += 1;
            self.stats.gpu_time_ns +|= getTimeNs() -| start_ns;
            return;
        }
        self.stats.cpu_fallback_ops += 1;
        cpuMatmul(a, b, c, m, n, k);
    }

    /// In-place RMS normalization.
    pub fn rmsNorm(self: *GpuTrainingBridge, x: []f32, weight: []const f32, eps: f32) void {
        if (self.gpu_ops) |*ops| {
            const start_ns = getTimeNs();
            ops.rmsnorm(
                @ptrCast(x.ptr),
                @ptrCast(weight.ptr),
                @intCast(x.len),
                eps,
                null,
            ) catch {
                self.stats.cpu_fallback_ops += 1;
                cpuRmsNorm(x, weight, eps);
                return;
            };
            self.stats.total_gpu_ops += 1;
            self.stats.gpu_time_ns +|= getTimeNs() -| start_ns;
            return;
        }
        self.stats.cpu_fallback_ops += 1;
        cpuRmsNorm(x, weight, eps);
    }

    /// In-place softmax.
    pub fn softmax(self: *GpuTrainingBridge, data: []f32) void {
        if (self.gpu_ops) |*ops| {
            const start_ns = getTimeNs();
            ops.softmax(@ptrCast(data.ptr), @intCast(data.len), null) catch {
                self.stats.cpu_fallback_ops += 1;
                cpuSoftmax(data);
                return;
            };
            self.stats.total_gpu_ops += 1;
            self.stats.gpu_time_ns +|= getTimeNs() -| start_ns;
            return;
        }
        self.stats.cpu_fallback_ops += 1;
        cpuSoftmax(data);
    }

    /// In-place SiLU activation.
    pub fn silu(self: *GpuTrainingBridge, data: []f32) void {
        if (self.gpu_ops) |*ops| {
            const start_ns = getTimeNs();
            ops.silu(@ptrCast(data.ptr), @intCast(data.len), null) catch {
                self.stats.cpu_fallback_ops += 1;
                cpuSilu(data);
                return;
            };
            self.stats.total_gpu_ops += 1;
            self.stats.gpu_time_ns +|= getTimeNs() -| start_ns;
            return;
        }
        self.stats.cpu_fallback_ops += 1;
        cpuSilu(data);
    }

    /// In-place element-wise multiply: a = a * b.
    pub fn elementwiseMul(self: *GpuTrainingBridge, a: []f32, b: []const f32) void {
        if (self.gpu_ops) |*ops| {
            const start_ns = getTimeNs();
            ops.elementwiseMul(
                @ptrCast(a.ptr),
                @ptrCast(b.ptr),
                @intCast(a.len),
                null,
            ) catch {
                self.stats.cpu_fallback_ops += 1;
                for (a, b) |*av, bv| av.* *= bv;
                return;
            };
            self.stats.total_gpu_ops += 1;
            self.stats.gpu_time_ns +|= getTimeNs() -| start_ns;
            return;
        }
        self.stats.cpu_fallback_ops += 1;
        for (a, b) |*av, bv| av.* *= bv;
    }

    /// In-place element-wise add: a = a + b.
    pub fn elementwiseAdd(self: *GpuTrainingBridge, a: []f32, b: []const f32) void {
        if (self.gpu_ops) |*ops| {
            const start_ns = getTimeNs();
            ops.elementwiseAdd(
                @ptrCast(a.ptr),
                @ptrCast(b.ptr),
                @intCast(a.len),
                null,
            ) catch {
                self.stats.cpu_fallback_ops += 1;
                for (a, b) |*av, bv| av.* += bv;
                return;
            };
            self.stats.total_gpu_ops += 1;
            self.stats.gpu_time_ns +|= getTimeNs() -| start_ns;
            return;
        }
        self.stats.cpu_fallback_ops += 1;
        for (a, b) |*av, bv| av.* += bv;
    }

    /// Update utilization estimate based on accumulated stats.
    pub fn updateUtilization(self: *GpuTrainingBridge) void {
        self.stats.utilization = self.stats.gpuRatio();
    }
};

// =============================================================================
// CPU Fallback Implementations
// =============================================================================

fn cpuMatmul(a: []const f32, b: []const f32, c: []f32, m: u32, n: u32, k: u32) void {
    // C[i,j] = sum_l A[i,l] * B[l,j]
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |l| {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn cpuRmsNorm(x: []f32, weight: []const f32, eps: f32) void {
    var sum_sq: f32 = 0;
    for (x) |v| sum_sq += v * v;
    const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(x.len)) + eps);
    for (x, weight) |*xv, wv| {
        xv.* = (xv.* / rms) * wv;
    }
}

fn cpuSoftmax(data: []f32) void {
    var max_val: f32 = -std.math.inf(f32);
    for (data) |v| max_val = @max(max_val, v);
    var sum: f32 = 0;
    for (data) |*v| {
        v.* = @exp(v.* - max_val);
        sum += v.*;
    }
    if (sum > 0) {
        for (data) |*v| v.* /= sum;
    }
}

fn cpuSilu(data: []f32) void {
    for (data) |*v| {
        const sigmoid = 1.0 / (1.0 + @exp(-v.*));
        v.* = v.* * sigmoid;
    }
}

fn getTimeNs() u64 {
    var timer = time.Timer.start() catch return 0;
    return timer.read();
}

// =============================================================================
// Tests
// =============================================================================

test "gpu training bridge init (cpu fallback)" {
    var bridge = GpuTrainingBridge.init(std.testing.allocator);
    defer bridge.deinit();

    const stats = bridge.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.total_gpu_ops);
    try std.testing.expectEqual(@as(u64, 0), stats.cpu_fallback_ops);
}

test "gpu training bridge matmul" {
    var bridge = GpuTrainingBridge.init(std.testing.allocator);
    defer bridge.deinit();

    // 2x2 matrix multiply
    const a = [_]f32{ 1, 2, 3, 4 };
    const b = [_]f32{ 5, 6, 7, 8 };
    var c = [_]f32{ 0, 0, 0, 0 };

    bridge.matmul(&a, &b, &c, 2, 2, 2);

    // C = A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //           = [[19, 22], [43, 50]]
    try std.testing.expectApproxEqAbs(@as(f32, 19), c[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 22), c[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 43), c[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 50), c[3], 0.01);
}

test "gpu training bridge softmax" {
    var bridge = GpuTrainingBridge.init(std.testing.allocator);
    defer bridge.deinit();

    var data = [_]f32{ 1.0, 2.0, 3.0 };
    bridge.softmax(&data);

    var sum: f32 = 0;
    for (data) |v| sum += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.001);
}

test "gpu training bridge silu" {
    var bridge = GpuTrainingBridge.init(std.testing.allocator);
    defer bridge.deinit();

    var data = [_]f32{ 0.0, 1.0, -1.0 };
    bridge.silu(&data);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), data[0], 0.001);
    try std.testing.expect(data[1] > 0);
    try std.testing.expect(data[2] < 0);
}

test "gpu training stats" {
    var stats = GpuTrainingStats{
        .total_gpu_ops = 80,
        .cpu_fallback_ops = 20,
        .gpu_time_ns = 1_000_000, // 1ms
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.8), stats.gpuRatio(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0125), stats.avgKernelTimeMs(), 0.001);
}
