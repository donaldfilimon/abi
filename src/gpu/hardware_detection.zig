const std = @import("std");
const builtin = @import("builtin");

/// GPU backend options supported by the framework.
pub const BackendType = enum {
    cuda,
    metal,
    directx12,
    vulkan,
    opencl,
    opengl,
    webgpu,
    cpu_fallback,

    /// Get the priority value for backend selection (higher = better)
    pub fn priority(self: BackendType) u32 {
        return switch (self) {
            .cuda => 100, // Highest priority - NVIDIA CUDA
            .vulkan => 90, // Cross-platform GPU acceleration
            .metal => 85, // Apple GPU acceleration
            .directx12 => 80, // Microsoft DirectX 12
            .opencl => 70, // OpenCL for older GPUs
            .opengl => 60, // OpenGL fallback
            .webgpu => 50, // WebGPU for web compatibility
            .cpu_fallback => 0, // Lowest priority - CPU fallback
        };
    }

    /// Get a display name for the backend
    pub fn displayName(self: BackendType) []const u8 {
        return switch (self) {
            .cuda => "CUDA",
            .vulkan => "Vulkan",
            .metal => "Metal",
            .directx12 => "DirectX 12",
            .opencl => "OpenCL",
            .opengl => "OpenGL",
            .webgpu => "WebGPU",
            .cpu_fallback => "CPU Fallback",
        };
    }

    /// Check if backend supports compute operations
    pub fn supportsCompute(self: BackendType) bool {
        return switch (self) {
            .cuda => true,
            .vulkan => true,
            .metal => true,
            .directx12 => true,
            .opencl => true,
            .opengl => false, // Limited compute support
            .webgpu => true,
            .cpu_fallback => true, // CPU can do compute
        };
    }

    /// Check if backend supports graphics operations
    pub fn supportsGraphics(self: BackendType) bool {
        return switch (self) {
            .cuda => false, // CUDA is compute-only
            .vulkan => true,
            .metal => true,
            .directx12 => true,
            .opencl => false, // Limited graphics support
            .opengl => true,
            .webgpu => true,
            .cpu_fallback => false, // No GPU graphics
        };
    }

    /// Check if backend is cross-platform
    pub fn isCrossPlatform(self: BackendType) bool {
        return switch (self) {
            .cuda => false, // NVIDIA only
            .vulkan => true,
            .metal => false, // Apple only
            .directx12 => false, // Windows only
            .opencl => true,
            .opengl => true,
            .webgpu => true,
            .cpu_fallback => true,
        };
    }

    /// Get the shader language used by this backend
    pub fn shaderLanguage(self: BackendType) []const u8 {
        return switch (self) {
            .cuda => "CUDA C++",
            .vulkan => "SPIR-V/GLSL",
            .metal => "Metal Shading Language",
            .directx12 => "HLSL",
            .opencl => "OpenCL C",
            .opengl => "GLSL",
            .webgpu => "WGSL",
            .cpu_fallback => "None",
        };
    }

    /// Get supported platforms for this backend
    pub fn supportedPlatforms(self: BackendType) []const []const u8 {
        return switch (self) {
            .cuda => &[_][]const u8{ "Windows", "Linux" },
            .vulkan => &[_][]const u8{ "Windows", "Linux", "macOS", "Android", "iOS" },
            .metal => &[_][]const u8{ "macOS", "iOS", "tvOS", "watchOS" },
            .directx12 => &[_][]const u8{"Windows"},
            .opencl => &[_][]const u8{ "Windows", "Linux", "macOS", "Android" },
            .opengl => &[_][]const u8{ "Windows", "Linux", "macOS" },
            .webgpu => &[_][]const u8{ "Web", "Windows", "Linux", "macOS" },
            .cpu_fallback => &[_][]const u8{"Any"},
        };
    }

    /// Check if backend is available on current platform
    pub fn isAvailable(self: BackendType) bool {
        return switch (self) {
            .cuda => builtin.os.tag == .windows or builtin.os.tag == .linux,
            .vulkan => builtin.os.tag == .linux or builtin.os.tag == .windows,
            .metal => builtin.os.tag == .macos or builtin.os.tag == .ios,
            .directx12 => builtin.os.tag == .windows,
            .opencl => true, // OpenCL is widely available
            .opengl => true, // OpenGL is widely available
            .webgpu => true, // WebGPU has broad support
            .cpu_fallback => true, // Always available
        };
    }
};

/// Coarse performance tier classification used by demos and heuristics.
pub const PerformanceTier = enum {
    entry_level,
    mainstream,
    enthusiast,
    workstation,
    ai_optimized,
};

/// Lightweight GPU type bucket for convenience helpers.
pub const GPUType = enum {
    discrete,
    integrated,
};

/// Basic system level capabilities reported alongside detection results.
pub const SystemCapabilities = struct {
    has_discrete_gpu: bool = false,
    has_integrated_gpu: bool = false,
    recommended_backend: BackendType = .cpu_fallback,
    total_vram: u64 = 0,
    shared_memory_limit: u64 = 0,
};

/// Minimal real GPU information record retained for compatibility with
/// higher-level code. Fields map to historic structure layouts.
pub const RealGPUInfo = struct {
    name: []u8,
    vendor: []u8,
    vendor_id: u32 = 0,
    device_id: u32 = 0,
    architecture: []u8,
    memory_size: u64 = 0,
    memory_bandwidth: u64 = 0,
    memory_type: []u8,
    memory_bus_width: u32 = 0,
    compute_units: u32 = 0,
    max_clock_speed: u32 = 0,
    base_clock_speed: u32 = 0,
    memory_clock_speed: u32 = 0,
    shader_cores: u32 = 0,
    tensor_cores: u32 = 0,
    rt_cores: u32 = 0,
    raster_units: u32 = 0,
    texture_units: u32 = 0,
    l1_cache_size: u32 = 0,
    l2_cache_size: u32 = 0,
    shared_memory_size: u32 = 0,
    power_limit: u32 = 0,
    tdp_watts: u32 = 0,
    manufacturing_process: []u8,
    driver_version: []u8,
    opengl_version: []u8,
    vulkan_version: []u8,
    directx_version: []u8,
    cuda_version: []u8,
    opencl_version: []u8,
    current_temperature: f32 = 0.0,
    fan_speed_rpm: u32 = 0,
    power_draw_watts: f32 = 0.0,
    voltage_mv: f32 = 0.0,
    pci_bus_id: u32 = 0,
    pci_device_id: u32 = 0,
    pci_function_id: u32 = 0,
    is_primary: bool = false,
    is_discrete: bool = false,
    is_integrated: bool = false,
    is_mobile: bool = false,
    is_workstation: bool = false,
    is_gaming: bool = false,
    is_ai_optimized: bool = false,
    available_backends: []BackendType,
    performance_tier: PerformanceTier = .entry_level,
    memory_allocator: std.mem.Allocator,

    pub fn deinit(self: *RealGPUInfo) void {
        const allocator = self.memory_allocator;
        allocator.free(self.name);
        allocator.free(self.vendor);
        allocator.free(self.architecture);
        allocator.free(self.memory_type);
        allocator.free(self.manufacturing_process);
        allocator.free(self.driver_version);
        allocator.free(self.opengl_version);
        allocator.free(self.vulkan_version);
        allocator.free(self.directx_version);
        allocator.free(self.cuda_version);
        allocator.free(self.opencl_version);
        allocator.free(self.available_backends);
    }
};

/// Aggregate detection result returned by the detector.
pub const GPUDetectionResult = struct {
    allocator: std.mem.Allocator,
    gpus: []RealGPUInfo,
    discrete_gpus: []RealGPUInfo,
    integrated_gpus: []RealGPUInfo,
    available_backends: []BackendType,
    total_gpus: usize,
    system_capabilities: SystemCapabilities,

    pub fn deinit(self: *GPUDetectionResult) void {
        for (self.gpus) |*gpu| gpu.deinit();
        self.allocator.free(self.gpus);
        self.allocator.free(self.discrete_gpus);
        self.allocator.free(self.integrated_gpus);
        self.allocator.free(self.available_backends);
    }
};

/// Main detector type. The current implementation synthesizes a conservative
/// fallback profile so that higher layers can rely on deterministic data even
/// when platform specific detection hooks are unavailable.
pub const GPUDetector = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) GPUDetector {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *GPUDetector) void {
        _ = self;
    }

    pub fn detectGPUs(self: *GPUDetector) !GPUDetectionResult {
        return createFallbackResult(self.allocator);
    }
};

fn createFallbackResult(allocator: std.mem.Allocator) !GPUDetectionResult {
    // Single placeholder entry so downstream code continues to operate.
    var gpus = try allocator.alloc(RealGPUInfo, 1);
    errdefer allocator.free(gpus);

    var backends = try allocator.alloc(BackendType, 2);
    backends[0] = .webgpu;
    backends[1] = .cpu_fallback;

    gpus[0] = .{
        .name = try allocator.dupe(u8, "Generic GPU"),
        .vendor = try allocator.dupe(u8, "Generic"),
        .architecture = try allocator.dupe(u8, "x86_64"),
        .memory_type = try allocator.dupe(u8, "System Memory"),
        .manufacturing_process = try allocator.dupe(u8, "unknown"),
        .driver_version = try allocator.dupe(u8, builtin.zig_version_string),
        .opengl_version = try allocator.dupe(u8, "n/a"),
        .vulkan_version = try allocator.dupe(u8, "n/a"),
        .directx_version = try allocator.dupe(u8, "n/a"),
        .cuda_version = try allocator.dupe(u8, "n/a"),
        .opencl_version = try allocator.dupe(u8, "n/a"),
        .available_backends = backends,
        .memory_size = 512 * 1024 * 1024,
        .memory_bandwidth = 0,
        .shader_cores = 0,
        .current_temperature = 0.0,
        .fan_speed_rpm = 0,
        .power_draw_watts = 0.0,
        .tdp_watts = 0,
        .is_integrated = true,
        .memory_allocator = allocator,
    };

    const discrete = try allocator.alloc(RealGPUInfo, 0);
    const integrated = try allocator.alloc(RealGPUInfo, 0);

    var available_backends = try allocator.alloc(BackendType, 2);
    available_backends[0] = .webgpu;
    available_backends[1] = .cpu_fallback;

    return GPUDetectionResult{
        .allocator = allocator,
        .gpus = gpus,
        .discrete_gpus = discrete,
        .integrated_gpus = integrated,
        .available_backends = available_backends,
        .total_gpus = gpus.len,
        .system_capabilities = .{
            .has_discrete_gpu = false,
            .has_integrated_gpu = true,
            .recommended_backend = .webgpu,
            .total_vram = gpus[0].memory_size,
            .shared_memory_limit = 0,
        },
    };
}

/// Runtime flag used by higher layers to decide whether to attempt real
/// hardware probing. The stub implementation always returns , which
/// encourages callers to use conservative defaults without failing builds on
/// unsupported targets.
pub fn isHardwareDetectionAvailable() bool {
    return false;
}

/// Return the most desirable backend present in the supplied GPU list.
pub fn determineRecommendedBackend(gpus: []RealGPUInfo) BackendType {
    const priority = [_]BackendType{ .cuda, .metal, .directx12, .vulkan, .opencl, .opengl, .webgpu, .cpu_fallback };
    for (priority) |candidate| {
        for (gpus) |gpu| {
            if (std.mem.indexOfScalar(BackendType, gpu.available_backends, candidate) != null) {
                return candidate;
            }
        }
    }
    return .cpu_fallback;
}

/// Convenience helper used by demos to print a concise summary.
pub fn logGPUDetectionResults(result: *const GPUDetectionResult) void {
    std.log.info("GPU detection (stub): detected {d} GPU(s)", .{result.total_gpus});
    for (result.gpus) |gpu| {
        std.log.info("  - {s} ({s})", .{ gpu.name, gpu.vendor });
        if (gpu.available_backends.len == 0) {
            std.log.info("    backends: none", .{});
        } else {
            for (gpu.available_backends, 0..) |backend, idx| {
                if (idx == 0) {
                    std.log.info("    backends: {s}", .{@tagName(backend)});
                } else {
                    std.log.info("              {s}", .{@tagName(backend)});
                }
            }
        }
    }
}
