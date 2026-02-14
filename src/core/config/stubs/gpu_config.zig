const std = @import("std");

pub const GpuConfig = struct {
    backend: Backend = .auto,
    device_index: u32 = 0,
    memory_limit: ?usize = null,
    async_enabled: bool = false,
    cache_kernels: bool = false,
    recovery: RecoveryConfig = .{},

    pub const Backend = enum {
        auto,
        vulkan,
        cuda,
        metal,
        webgpu,
        opengl,
        fpga,
        tpu,
        cpu,
    };

    pub fn defaults() GpuConfig {
        return .{};
    }

    pub fn autoSelectBackend() Backend {
        return .cpu;
    }
};

pub const RecoveryConfig = struct {
    enabled: bool = false,
    max_retries: u32 = 0,
    fallback_to_cpu: bool = true,
};
