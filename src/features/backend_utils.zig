const std = @import("std");
const builtin = @import("builtin");

/// Shared backend enumeration for both accelerator and GPU features
pub const Backend = enum {
    cpu,
    gpu_simulated,
    gpu_metal,
    gpu_vulkan,
    gpu_cuda,
    gpu_webgpu,
    gpu_opengl,
    gpu_webgl2,
    mlir,
};

/// Status of a backend including availability and acceleration info
pub const BackendStatus = struct {
    backend: Backend,
    available: bool,
    accelerated: bool,
    message: []const u8,
};

/// Execution mode for kernels
pub const ExecutionMode = enum {
    cpu_fallback,
    simulated_gpu,
    native_gpu,
};

/// Specification for a kernel execution
pub const KernelSpec = struct {
    name: []const u8,
    work_items: usize,
};

/// Result of kernel execution
pub const KernelResult = struct {
    backend: Backend,
    mode: ExecutionMode,
    work_items: usize,
    message: []const u8,
};

/// Status of native kernel availability
pub const NativeKernelStatus = struct {
    backend: Backend,
    linked: bool,
    message: []const u8,
};

/// Capabilities of a backend
pub const BackendCapabilities = struct {
    backend: Backend,
    available: bool,
    accelerated: bool,
    native_kernels: bool,
    message: []const u8,
};

/// Convert backend to string name
pub fn backendName(backend: Backend) []const u8 {
    return switch (backend) {
        .cpu => "cpu",
        .gpu_simulated => "gpu-simulated",
        .gpu_metal => "gpu-metal",
        .gpu_vulkan => "gpu-vulkan",
        .gpu_cuda => "gpu-cuda",
        .gpu_webgpu => "gpu-webgpu",
        .gpu_opengl => "gpu-opengl",
        .gpu_webgl2 => "gpu-webgl2",
        .mlir => "mlir",
    };
}

/// Check if a backend represents accelerated execution
pub fn isAccelerated(backend: Backend) bool {
    return switch (backend) {
        .gpu_metal, .gpu_vulkan, .gpu_cuda, .gpu_webgpu, .gpu_opengl, .gpu_webgl2 => true,
        else => false,
    };
}

/// Preferred backend for the current target platform
pub fn preferredBackendForTarget() Backend {
    if (builtin.target.os.tag == .macos) {
        return .gpu_metal;
    }
    if (builtin.target.os.tag == .linux or builtin.target.os.tag == .windows) {
        return .gpu_vulkan;
    }
    return .gpu_simulated;
}

/// Detect backend status with platform-specific logic
pub fn detectBackend() BackendStatus {
    const metal = @import("../gpu/metal_shared.zig");
    const caps = backendCapabilities(preferredBackendForTarget());
    return .{
        .backend = caps.backend,
        .available = caps.available,
        .accelerated = caps.accelerated,
        .message = caps.message,
    };
}

/// Get capabilities for a specific backend
pub fn backendCapabilities(backend: Backend) BackendCapabilities {
    return switch (backend) {
        .gpu_metal => blk: {
            const platform_available = builtin.target.os.tag == .macos;
            const initialized = if (@inComptime()) false else metal.g_metal_context.initialized;
            break :blk .{
                .backend = .gpu_metal,
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
        .gpu_vulkan => .{
            .backend = .gpu_vulkan,
            .available = builtin.target.os.tag == .linux or builtin.target.os.tag == .windows,
            .accelerated = false,
            .native_kernels = false,
            .message = "Vulkan backend registered; native kernels are not linked, using vectorized CPU fallback",
        },
        .gpu_cuda => .{
            .backend = .gpu_cuda,
            .available = false,
            .accelerated = false,
            .native_kernels = false,
            .message = "CUDA backend registered; native runtime is not linked, using vectorized CPU fallback",
        },
        .gpu_webgpu => .{
            .backend = .gpu_webgpu,
            .available = false,
            .accelerated = false,
            .native_kernels = false,
            .message = "WebGPU backend registered; browser/runtime adapter is not linked, using vectorized CPU fallback",
        },
        .gpu_opengl => .{
            .backend = .gpu_opengl,
            .available = false,
            .accelerated = false,
            .native_kernels = false,
            .message = "OpenGL backend registered; compute path is not linked, using vectorized CPU fallback",
        },
        .gpu_webgl2 => .{
            .backend = .gpu_webgl2,
            .available = false,
            .accelerated = false,
            .native_kernels = false,
            .message = "WebGL2 backend registered; browser adapter is not linked, using vectorized CPU fallback",
        },
        .gpu_simulated => .{
            .backend = .gpu_simulated,
            .available = true,
            .accelerated = false,
            .native_kernels = false,
            .message = "Deterministic vectorized CPU fallback backend active",
        },
        .cpu => .{
            .backend = .cpu,
            .available = true,
            .accelerated = false,
            .native_kernels = false,
            .message = "CPU backend active",
        },
        .mlir => .{
            .backend = .mlir,
            .available = true,
            .accelerated = false,
            .native_kernels = false,
            .message = "MLIR textual lowering selected with CPU execution fallback",
        },
    };
}

/// Get list of all backend capabilities
pub fn backendCapabilitiesList() [9]BackendCapabilities {
    return .{
        backendCapabilities(.cpu),
        backendCapabilities(.gpu_simulated),
        backendCapabilities(.gpu_metal),
        backendCapabilities(.gpu_vulkan),
        backendCapabilities(.gpu_cuda),
        backendCapabilities(.gpu_webgpu),
        backendCapabilities(.gpu_opengl),
        backendCapabilities(.gpu_webgl2),
        backendCapabilities(.mlir),
    };
}
