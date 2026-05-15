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

test "gpu detection always provides a safe backend" {
    const status = detectBackend();
    try std.testing.expect(status.available);
    try std.testing.expect(status.message.len > 0);
}
