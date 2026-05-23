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

test {
    const std = @import("std");
    std.testing.refAllDecls(@This());
}
