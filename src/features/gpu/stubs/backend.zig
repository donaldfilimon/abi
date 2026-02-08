const std = @import("std");

pub const Backend = enum {
    cuda,
    vulkan,
    stdgpu,
    metal,
    webgpu,
    opengl,
    opengles,
    webgl2,
    fpga,
    simulated,
};

pub const BackendInfo = struct {
    backend: Backend,
    name: []const u8,
    description: []const u8 = "GPU Disabled",
    enabled: bool = false,
    available: bool,
    availability: []const u8 = "disabled",
    device_count: usize = 0,
    build_flag: []const u8 = "",
};

pub const DetectionLevel = enum { none, loader, device_count };

pub const BackendAvailability = struct {
    enabled: bool = false,
    available: bool = false,
    reason: []const u8 = "gpu disabled",
    device_count: usize = 0,
    level: DetectionLevel = .none,
};

pub const Summary = struct {
    module_enabled: bool = false,
    enabled_backend_count: usize = 0,
    available_backend_count: usize = 0,
    device_count: usize = 0,
    emulated_devices: usize = 0,
};
