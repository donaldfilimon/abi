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
    };
}

pub fn detectBackend() BackendStatus {
    return .{
        .backend = .simulated,
        .available = true,
        .accelerated = false,
        .message = "GPU feature is disabled; using simulated backend",
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
