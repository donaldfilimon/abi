//! C-compatible GPU operation exports.
//! Provides GPU context management for FFI.

const std = @import("std");
const builtin = @import("builtin");
const errors = @import("errors.zig");

/// Opaque GPU handle for C API.
pub const GpuHandle = opaque {};

/// GPU config matching C header (abi_gpu_config_t).
pub const GpuConfig = extern struct {
    backend: c_int = 0, // 0=auto, 1=cuda, 2=vulkan, 3=metal, 4=webgpu
    device_index: c_int = 0,
    enable_profiling: bool = false,
};

/// GPU backend enum.
pub const Backend = enum(c_int) {
    auto = 0,
    cuda = 1,
    vulkan = 2,
    metal = 3,
    webgpu = 4,
    cpu = 5,
};

/// Internal GPU state.
const GpuState = struct {
    allocator: std.mem.Allocator,
    backend: Backend,
    device_index: c_int,
    profiling_enabled: bool,

    pub fn init(allocator: std.mem.Allocator, config: GpuConfig) !*GpuState {
        const state = try allocator.create(GpuState);
        errdefer allocator.destroy(state);

        // Detect best backend
        const backend = detectBackend(@enumFromInt(config.backend));

        state.* = .{
            .allocator = allocator,
            .backend = backend,
            .device_index = config.device_index,
            .profiling_enabled = config.enable_profiling,
        };

        return state;
    }

    pub fn deinit(self: *GpuState) void {
        self.allocator.destroy(self);
    }

    pub fn backendName(self: *const GpuState) [*:0]const u8 {
        return switch (self.backend) {
            .auto => "auto",
            .cuda => "cuda",
            .vulkan => "vulkan",
            .metal => "metal",
            .webgpu => "webgpu",
            .cpu => "cpu",
        };
    }
};

/// Detect best available backend.
fn detectBackend(requested: Backend) Backend {
    if (requested != .auto) {
        return requested;
    }

    // Auto-detect based on platform
    const os = builtin.os.tag;

    if (os == .macos) {
        return .metal;
    }

    // Default to CPU fallback
    return .cpu;
}

/// Check if any GPU is potentially available.
fn isGpuAvailable() bool {
    const os = builtin.os.tag;

    // Metal available on macOS
    if (os == .macos) return true;

    // Other platforms might have Vulkan or CUDA
    if (os == .linux or os == .windows) return true;

    return false;
}

// Global allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};

/// Initialize GPU context.
pub export fn abi_gpu_init(config: *const GpuConfig, out_gpu: *?*GpuHandle) errors.Error {
    const allocator = gpa.allocator();
    const state = GpuState.init(allocator, config.*) catch |err| {
        return errors.fromZigError(err);
    };
    out_gpu.* = @ptrCast(state);
    return errors.OK;
}

/// Shutdown GPU context.
pub export fn abi_gpu_shutdown(handle: ?*GpuHandle) void {
    if (handle) |h| {
        const state: *GpuState = @ptrCast(@alignCast(h));
        state.deinit();
    }
}

/// Check if any GPU backend is available.
pub export fn abi_gpu_is_available() bool {
    return isGpuAvailable();
}

/// Get active GPU backend name.
pub export fn abi_gpu_backend_name(handle: ?*GpuHandle) [*:0]const u8 {
    if (handle) |h| {
        const state: *const GpuState = @ptrCast(@alignCast(h));
        return state.backendName();
    }
    return "none";
}

test "gpu exports" {
    var gpu: ?*GpuHandle = null;
    const config = GpuConfig{
        .backend = 0, // auto
        .device_index = 0,
        .enable_profiling = false,
    };

    try std.testing.expectEqual(errors.OK, abi_gpu_init(&config, &gpu));
    try std.testing.expect(gpu != null);

    const name = abi_gpu_backend_name(gpu);
    try std.testing.expect(std.mem.len(name) > 0);

    _ = abi_gpu_is_available();

    abi_gpu_shutdown(gpu);

    // Null handle should return "none"
    try std.testing.expectEqualStrings("none", std.mem.span(abi_gpu_backend_name(null)));
}
