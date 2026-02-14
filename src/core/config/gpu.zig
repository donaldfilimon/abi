//! GPU Configuration
//!
//! Configuration for GPU acceleration including backend selection,
//! device settings, and recovery options.

const std = @import("std");
const build_options = @import("build_options");

/// GPU acceleration configuration.
pub const GpuConfig = struct {
    /// GPU backend to use. Auto-detect by default.
    backend: Backend = .auto,

    /// Preferred device index (0 = first available).
    device_index: u32 = 0,

    /// Maximum GPU memory to use (null = no limit).
    memory_limit: ?usize = null,

    /// Enable async operations.
    async_enabled: bool = true,

    /// Enable kernel caching.
    cache_kernels: bool = true,

    /// Recovery settings for GPU failures.
    recovery: RecoveryConfig = .{},

    pub const Backend = enum {
        auto,
        cuda,
        vulkan,
        stdgpu,
        metal,
        webgpu,
        opengl,
        opengles,
        webgl2,
        fpga,
        tpu,
        cpu,
    };

    pub const RecoveryConfig = struct {
        enabled: bool = true,
        max_retries: u32 = 3,
        fallback_to_cpu: bool = true,
    };

    pub fn defaults() GpuConfig {
        return .{};
    }

    /// Select the best backend based on availability.
    pub fn autoSelectBackend() Backend {
        if (build_options.gpu_cuda) return .cuda;
        if (build_options.gpu_vulkan) return .vulkan;
        if (build_options.gpu_metal) return .metal;
        if (@hasDecl(build_options, "gpu_fpga") and build_options.gpu_fpga) return .fpga;
        if (@hasDecl(build_options, "gpu_tpu") and build_options.gpu_tpu) return .tpu;
        if (build_options.gpu_webgpu) return .webgpu;
        if (build_options.gpu_opengl) return .opengl;
        return .cpu;
    }
};
