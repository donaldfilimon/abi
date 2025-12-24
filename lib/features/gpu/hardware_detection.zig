const std = @import("std");
const builtin = @import("builtin");

/// Accelerator backend options supported by the framework.
/// Covers GPU (CUDA, Vulkan, Metal, DX12, OpenCL, OpenGL, WebGPU),
/// AI accelerators (TPU, NPU), and CPU fallbacks (SIMD, scalar).
pub const BackendType = enum {
    // GPU backends
    cuda,
    vulkan,
    metal,
    directx12,
    opencl,
    opengl,
    webgpu,
    rocm, // AMD ROCm

    // AI/ML accelerators
    tpu, // Google TPU / Coral Edge TPU
    npu, // Neural Processing Units (Apple Neural Engine, Qualcomm Hexagon, etc)
    sycl, // Intel oneAPI SYCL

    // CPU backends
    cpu_simd, // Optimized SIMD (AVX-512, AVX2, NEON)
    cpu_fallback, // Basic scalar CPU

    /// Get the priority value for backend selection (higher = better)
    pub fn priority(self: BackendType) u32 {
        return switch (self) {
            .tpu => 110, // AI-optimized hardware
            .npu => 105, // Neural processing unit
            .cuda => 100, // NVIDIA CUDA
            .rocm => 95, // AMD ROCm
            .vulkan => 90, // Cross-platform GPU
            .metal => 85, // Apple GPU
            .sycl => 82, // Intel oneAPI
            .directx12 => 80, // Windows GPU
            .opencl => 70, // Legacy GPU compute
            .opengl => 60, // Legacy graphics
            .webgpu => 50, // Web compatibility
            .cpu_simd => 20, // Optimized CPU
            .cpu_fallback => 0, // Basic CPU
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
            .rocm => "AMD ROCm",
            .tpu => "TPU",
            .npu => "NPU",
            .sycl => "Intel SYCL",
            .cpu_simd => "CPU SIMD",
            .cpu_fallback => "CPU Fallback",
        };
    }

    /// Check if backend supports compute operations
    pub fn supportsCompute(self: BackendType) bool {
        return switch (self) {
            .cuda, .vulkan, .metal, .directx12, .opencl, .webgpu => true,
            .rocm, .tpu, .npu, .sycl => true,
            .cpu_simd, .cpu_fallback => true,
            .opengl => false,
        };
    }

    /// Check if backend supports neural network training
    pub fn supportsTraining(self: BackendType) bool {
        return switch (self) {
            .cuda, .rocm, .tpu, .npu, .sycl => true,
            .vulkan, .metal, .directx12, .opencl => true,
            .cpu_simd, .cpu_fallback => true,
            .opengl, .webgpu => false,
        };
    }

    /// Check if backend supports tensor operations
    pub fn supportsTensors(self: BackendType) bool {
        return switch (self) {
            .cuda, .rocm, .tpu, .npu, .sycl => true, // Native tensor support
            .vulkan, .metal, .directx12 => true, // Via compute shaders
            .opencl, .cpu_simd, .cpu_fallback => true,
            .opengl, .webgpu => false,
        };
    }

    /// Check if backend supports graphics operations
    pub fn supportsGraphics(self: BackendType) bool {
        return switch (self) {
            .vulkan, .metal, .directx12, .opengl, .webgpu => true,
            .cuda, .rocm, .opencl, .tpu, .npu, .sycl => false,
            .cpu_simd, .cpu_fallback => false,
        };
    }

    /// Check if backend is cross-platform
    pub fn isCrossPlatform(self: BackendType) bool {
        return switch (self) {
            .vulkan, .opencl, .opengl, .webgpu => true,
            .cpu_simd, .cpu_fallback => true,
            .cuda, .rocm => false, // Vendor specific
            .metal, .directx12 => false, // OS specific
            .tpu, .npu, .sycl => false, // Hardware specific
        };
    }

    /// Check if backend is available on current platform
    pub fn isAvailable(self: BackendType) bool {
        return switch (self) {
            .cuda => builtin.os.tag == .windows or builtin.os.tag == .linux,
            .rocm => builtin.os.tag == .linux,
            .vulkan => builtin.os.tag == .linux or builtin.os.tag == .windows,
            .metal => builtin.os.tag == .macos or builtin.os.tag == .ios,
            .directx12 => builtin.os.tag == .windows,
            .sycl => builtin.os.tag == .linux or builtin.os.tag == .windows,
            .tpu => false, // Requires specific hardware detection
            .npu => builtin.os.tag == .macos or builtin.os.tag == .ios, // Apple Neural Engine
            .opencl, .opengl, .webgpu => true,
            .cpu_simd, .cpu_fallback => true,
        };
    }

    /// Get SIMD width for this backend (in f32 elements)
    pub fn simdWidth(self: BackendType) u32 {
        return switch (self) {
            .cpu_simd => switch (builtin.cpu.arch) {
                .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) 16 else if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx2)) 8 else 4,
                .aarch64 => 4, // NEON
                else => 4,
            },
            .cuda, .rocm => 32, // Warp size
            .tpu => 128, // TPU vector width
            .npu => 64, // Typical NPU width
            else => 1,
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
    // Single fallback entry so downstream code continues to operate.
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
/// hardware probing. Returns false to keep defaults conservative on unsupported targets.
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
    std.log.info("GPU detection (fallback): detected {d} GPU(s)", .{result.total_gpus});
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
