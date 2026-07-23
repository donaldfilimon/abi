pub const backends = @This();
pub const vector_ops = @This();
pub const reporting = @This();

pub const Backend = enum {
    simulated,
    metal,
    vulkan,
    cuda,
    webgpu,
    opengl,
    webgl2,
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

pub const NativeKernelStatus = struct {
    backend: Backend,
    linked: bool,
    message: []const u8,
};

pub const BackendCapabilities = struct {
    backend: Backend,
    available: bool,
    accelerated: bool,
    native_kernels: bool,
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
        for (a, b) |av, bv| sum += av * bv;
        return sum;
    }

    pub fn squaredL2(self: VectorOps, a: []const f32, b: []const f32) !f32 {
        _ = self;
        if (a.len != b.len) return error.DimensionMismatch;
        var sum: f32 = 0;
        for (a, b) |av, bv| {
            const diff = av - bv;
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

    pub fn batchCosineSimilarity(self: VectorOps, query: []const f32, candidates: []const []const f32, out: []f32) !void {
        if (out.len != candidates.len) return error.DimensionMismatch;
        for (candidates, out) |cand, *slot| {
            slot.* = try self.cosineSimilarity(query, cand);
        }
    }

    pub fn softmax(self: VectorOps, values: []const f32, out: []f32) !void {
        _ = self;
        if (values.len == 0) return;
        if (out.len != values.len) return error.DimensionMismatch;
        var max_val: f32 = values[0];
        for (values[1..]) |v| {
            if (v > max_val) max_val = v;
        }
        var sum: f32 = 0;
        for (values, out) |v, *slot| {
            slot.* = @exp(v - max_val);
            sum += slot.*;
        }
        if (sum == 0) {
            @memset(out, 0);
            return;
        }
        for (out) |*slot| slot.* /= sum;
    }
};

pub fn backendName(backend: Backend) []const u8 {
    return switch (backend) {
        .simulated => "simulated",
        .metal => "metal",
        .vulkan => "vulkan",
        .cuda => "cuda",
        .webgpu => "webgpu",
        .opengl => "opengl",
        .webgl2 => "webgl2",
    };
}

pub fn detectBackend() BackendStatus {
    return backendStatus(.simulated);
}

pub fn backendStatus(backend: Backend) BackendStatus {
    const caps = backendCapabilities(backend);
    return .{
        .backend = caps.backend,
        .available = caps.available,
        .accelerated = caps.accelerated,
        .message = caps.message,
    };
}

pub fn backendCapabilities(backend: Backend) BackendCapabilities {
    return .{
        .backend = backend,
        .available = backend == .simulated,
        .accelerated = false,
        .native_kernels = false,
        .message = if (backend == .simulated) "GPU feature is disabled; simulated CPU backend active" else "GPU feature is disabled; backend unavailable",
    };
}

pub fn backendCapabilitiesList() [7]BackendCapabilities {
    return .{
        backendCapabilities(.simulated),
        backendCapabilities(.metal),
        backendCapabilities(.vulkan),
        backendCapabilities(.cuda),
        backendCapabilities(.webgpu),
        backendCapabilities(.opengl),
        backendCapabilities(.webgl2),
    };
}

pub fn threadsPerGroup(backend: Backend) usize {
    _ = backend;
    return 1;
}

pub fn backendStatusReport(allocator: @import("std").mem.Allocator) ![]u8 {
    return try allocator.dupe(u8, "GPU feature is disabled; simulated CPU backend active");
}

pub fn nativeKernelStatus() NativeKernelStatus {
    return .{
        .backend = .simulated,
        .linked = false,
        .message = "GPU feature is disabled; native kernels are unavailable",
    };
}

pub fn isAvailable() bool {
    return true;
}

pub fn preferredBackend() Backend {
    return .simulated;
}

pub const PresenceProbe = struct {
    backend: Backend,
    declared: bool,
    native_linked: bool,
    note: []const u8,
};

pub fn presenceProbe(backend: Backend) PresenceProbe {
    return .{
        .backend = backend,
        .declared = true,
        .native_linked = false,
        .note = "GPU feature is disabled; native dispatch not linked",
    };
}

pub fn executeKernel(spec: KernelSpec) !KernelResult {
    if (spec.name.len == 0) return error.InvalidKernelName;
    return .{
        .backend = .simulated,
        .mode = .cpu_fallback,
        .work_items = spec.work_items,
        .message = "GPU feature is disabled; CPU fallback executed",
    };
}

pub fn vectorOps() VectorOps {
    return VectorOps.init();
}

// Parent GPU compute API mirror (feature disabled): CPU-only, every accelerator
// honestly reported unavailable. Keeps name/shape parity with the real module.
pub const compute_api = @This();

pub const Kernel = enum { mul, l2diff, add };

pub const BackendAvailability = struct {
    backend: Backend,
    available: bool,
    dispatches: bool,
    reason: []const u8,
};

pub fn backendMatrix() [7]BackendAvailability {
    return .{
        .{ .backend = .simulated, .available = true, .dispatches = true, .reason = "GPU feature disabled; vectorized CPU path" },
        .{ .backend = .metal, .available = false, .dispatches = false, .reason = "GPU feature is disabled" },
        .{ .backend = .vulkan, .available = false, .dispatches = false, .reason = "GPU feature is disabled" },
        .{ .backend = .cuda, .available = false, .dispatches = false, .reason = "GPU feature is disabled" },
        .{ .backend = .webgpu, .available = false, .dispatches = false, .reason = "GPU feature is disabled" },
        .{ .backend = .opengl, .available = false, .dispatches = false, .reason = "GPU feature is disabled" },
        .{ .backend = .webgl2, .available = false, .dispatches = false, .reason = "GPU feature is disabled" },
    };
}

pub const GpuCompute = struct {
    backend: Backend = .simulated,
    accelerated: bool = false,

    pub fn init() GpuCompute {
        return .{};
    }

    pub fn backendName(self: GpuCompute) []const u8 {
        return @import("std").mem.span(@tagName(self.backend));
    }

    pub fn isAccelerated(self: GpuCompute) bool {
        _ = self;
        return false;
    }

    pub fn map(self: GpuCompute, kernel: Kernel, a: []const f32, b: []const f32, out: []f32) !void {
        _ = self;
        if (a.len != b.len or a.len != out.len) return error.DimensionMismatch;
        switch (kernel) {
            .mul => for (a, b, out) |x, y, *o| {
                o.* = x * y;
            },
            .l2diff => for (a, b, out) |x, y, *o| {
                const d = x - y;
                o.* = d * d;
            },
            .add => for (a, b, out) |x, y, *o| {
                o.* = x + y;
            },
        }
    }

    pub fn reduce(self: GpuCompute, kernel: Kernel, a: []const f32, b: []const f32) !f32 {
        _ = self;
        if (a.len != b.len) return error.DimensionMismatch;
        var sum: f32 = 0;
        switch (kernel) {
            .mul => for (a, b) |x, y| {
                sum += x * y;
            },
            .l2diff => for (a, b) |x, y| {
                const d = x - y;
                sum += d * d;
            },
            .add => for (a, b) |x, y| {
                sum += x + y;
            },
        }
        return sum;
    }
};

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}

test "gpu stub exposes deterministic CPU fallback" {
    const std = @import("std");
    const status = detectBackend();
    try std.testing.expectEqual(Backend.simulated, status.backend);
    try std.testing.expect(status.available);
    try std.testing.expect(!status.accelerated);

    const caps = backendCapabilitiesList();
    try std.testing.expectEqual(@as(usize, 7), caps.len);
    try std.testing.expectEqual(@as(usize, 1), threadsPerGroup(.metal));

    const kernel = try executeKernel(.{ .name = "disabled.kernel", .work_items = 4 });
    try std.testing.expectEqual(ExecutionMode.cpu_fallback, kernel.mode);
    try std.testing.expectEqual(@as(usize, 4), kernel.work_items);

    const ops = vectorOps();
    try std.testing.expectEqual(@as(f32, 32), try ops.dot(&.{ 1, 2, 3 }, &.{ 4, 5, 6 }));
    try std.testing.expectError(error.DimensionMismatch, ops.dot(&.{1}, &.{ 1, 2 }));

    const report = try backendStatusReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    try std.testing.expect(std.mem.indexOf(u8, report, "GPU feature is disabled") != null);
}
