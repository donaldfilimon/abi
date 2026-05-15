const builtin = @import("builtin");
const std = @import("std");

pub const Backend = enum {
    simulated,
    metal,
    vulkan,
    cuda,
};

pub const BackendStatus = struct {
    backend: Backend,
    available: bool,
    accelerated: bool,
    message: []const u8,
};

pub const ExecutionMode = enum {
    cpu_fallback,
    simulated_gpu,
    native_gpu,
};

pub const KernelSpec = struct {
    name: []const u8,
    work_items: usize,
};

pub const KernelResult = struct {
    backend: Backend,
    mode: ExecutionMode,
    work_items: usize,
    message: []const u8,
};

pub const VectorOps = struct {
    backend: BackendStatus,

    pub fn init() VectorOps {
        return .{ .backend = detectBackend() };
    }

    pub fn dot(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        _ = self;
        if (a.len != b.len) return error.DimensionMismatch;

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
        _ = self;
        if (a.len != b.len) return error.DimensionMismatch;

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

pub fn backendName(backend: Backend) []const u8 {
    return switch (backend) {
        .simulated => "simulated",
        .metal => "metal",
        .vulkan => "vulkan",
        .cuda => "cuda",
    };
}

pub fn detectBackend() BackendStatus {
    if (builtin.target.os.tag == .macos) {
        return .{
            .backend = .metal,
            .available = true,
            .accelerated = false,
            .message = "Metal-capable platform detected; using simulated backend until native kernels are linked",
        };
    }

    return .{
        .backend = .simulated,
        .available = true,
        .accelerated = false,
        .message = "No native GPU backend linked; using deterministic simulated backend",
    };
}

pub fn isAvailable() bool {
    return detectBackend().available;
}

pub fn preferredBackend() Backend {
    return detectBackend().backend;
}

pub fn executeKernel(spec: KernelSpec) !KernelResult {
    if (spec.name.len == 0) return error.InvalidKernelName;
    const status = detectBackend();
    return .{
        .backend = status.backend,
        .mode = if (status.accelerated) .native_gpu else .simulated_gpu,
        .work_items = spec.work_items,
        .message = if (status.accelerated) "native GPU kernel executed" else "simulated GPU kernel executed deterministically",
    };
}

pub fn vectorOps() VectorOps {
    return VectorOps.init();
}

test "gpu detection always provides a safe backend" {
    const status = detectBackend();
    try std.testing.expect(status.available);
    try std.testing.expect(status.message.len > 0);
}

test "gpu vector ops provide deterministic acceleration" {
    const ops = vectorOps();
    try std.testing.expectEqual(@as(f32, 32), try ops.dot(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    try std.testing.expectEqual(@as(f32, 27), try ops.squaredL2(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    try std.testing.expect((try ops.cosineSimilarity(&.{ 1, 0 }, &.{ 1, 0 })) == 1);
}
