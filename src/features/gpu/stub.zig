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
