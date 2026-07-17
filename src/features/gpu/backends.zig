const std = @import("std");
const builtin = @import("builtin");
const metal = @import("metal_shared.zig");

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

fn preferredBackendForTarget() Backend {
    if (builtin.target.os.tag == .macos) {
        return .metal;
    }
    return .simulated;
}

pub const PresenceProbe = struct {
    backend: Backend,
    declared: bool,
    native_linked: bool,
    note: []const u8,
};

pub fn presenceProbe(backend: Backend) PresenceProbe {
    const caps = backendCapabilities(backend);
    return .{
        .backend = backend,
        .declared = true,
        .native_linked = caps.native_kernels,
        .note = caps.message,
    };
}

pub fn threadsPerGroup(backend: Backend) usize {
    return switch (backend) {
        .simulated => 4,
        .metal => 256,
        .vulkan => 64,
        .cuda => 32,
        .webgpu => 64,
        .opengl => 16,
        .webgl2 => 1,
    };
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
    return switch (backend) {
        .metal => blk: {
            const platform_available = builtin.target.os.tag == .macos;
            const initialized = if (@inComptime()) false else metal.g_metal_context.initialized;
            break :blk .{
                .backend = .metal,
                .available = platform_available,
                .accelerated = platform_available and initialized,
                .native_kernels = platform_available and initialized,
                .message = if (!platform_available)
                    "Metal is only available on macOS; vectorized CPU fallback active"
                else if (initialized)
                    "Metal GPU acceleration active"
                else
                    "Metal framework linked; vectorized CPU fallback active until native kernels initialize",
            };
        },
        .vulkan => .{
            .backend = .vulkan,
            .available = false,
            .accelerated = false,
            .native_kernels = false,
            .message = "Vulkan backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        .cuda => .{
            .backend = .cuda,
            .available = false,
            .accelerated = false,
            .native_kernels = false,
            .message = "CUDA backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        .webgpu => .{
            .backend = .webgpu,
            .available = false,
            .accelerated = false,
            .native_kernels = false,
            .message = "WebGPU backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        .opengl => .{
            .backend = .opengl,
            .available = false,
            .accelerated = false,
            .native_kernels = false,
            .message = "OpenGL backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        .webgl2 => .{
            .backend = .webgl2,
            .available = false,
            .accelerated = false,
            .native_kernels = false,
            .message = "WebGL2 backend is declared but native dispatch is not linked; vectorized CPU fallback active",
        },
        .simulated => .{
            .backend = .simulated,
            .available = true,
            .accelerated = false,
            .native_kernels = false,
            .message = "Deterministic vectorized CPU fallback backend active",
        },
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

pub fn detectBackend() BackendStatus {
    return backendStatus(preferredBackendForTarget());
}

pub fn nativeKernelStatus() NativeKernelStatus {
    const status = detectBackend();
    const metal_linked = builtin.target.os.tag == .macos;
    return .{
        .backend = status.backend,
        .linked = metal_linked and metal.g_metal_context.initialized,
        .message = if (metal_linked)
            if (metal.g_metal_context.initialized)
                "Metal framework linked and native kernels compiled successfully"
            else
                "Metal framework linked at build time; native dispatch is using vectorized CPU fallback"
        else
            "native GPU kernel dispatch is not linked in this build; vectorized CPU fallback is active",
    };
}

test "gpu detection always provides a safe backend" {
    const status = detectBackend();
    try std.testing.expect(status.available);
    try std.testing.expect(status.message.len > 0);
}

test {
    std.testing.refAllDecls(@This());
}
