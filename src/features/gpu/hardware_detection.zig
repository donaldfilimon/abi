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
        // Try real detection first
        if (isHardwareDetectionAvailable()) {
            return self.detectGPUsReal();
        }
        // Fall back to synthetic detection
        return createFallbackResult(self.allocator);
    }

    /// Attempt real GPU detection using available backends
    fn detectGPUsReal(self: *GPUDetector) !GPUDetectionResult {
        var gpus = std.ArrayList(RealGPUInfo).init(self.allocator);
        defer gpus.deinit();

        var available_backends = std.ArrayList(BackendType).init(self.allocator);
        defer available_backends.deinit();

        // Always add CPU fallback
        try available_backends.append(.cpu_fallback);

        // Try CUDA detection
        if (BackendType.cuda.isAvailable()) {
            try self.detectCUDAGPUs(&gpus, &available_backends);
        }

        // Try Vulkan detection
        const vulkan = @import("libraries/vulkan_integration.zig");
        if (vulkan.VulkanRenderer.isAvailable()) {
            try self.detectVulkanGPUs(&gpus, &available_backends);
        }

        // Try Metal detection (macOS/iOS)
        if (BackendType.metal.isAvailable()) {
            try self.detectMetalGPUs(&gpus, &available_backends);
        }

        // Try DirectX 12 detection (Windows)
        if (BackendType.directx12.isAvailable()) {
            try self.detectDirectX12GPUs(&gpus, &available_backends);
        }

        // Try OpenCL detection (cross-platform)
        if (BackendType.opencl.isAvailable()) {
            try self.detectOpenCLGPUs(&gpus, &available_backends);
        }

        // Convert to slices
        const gpu_slice = try gpus.toOwnedSlice();
        errdefer self.allocator.free(gpu_slice);

        const backend_slice = try available_backends.toOwnedSlice();
        errdefer self.allocator.free(backend_slice);

        // Separate discrete and integrated GPUs
        var discrete_gpus = std.ArrayList(RealGPUInfo).init(self.allocator);
        defer discrete_gpus.deinit();

        var integrated_gpus = std.ArrayList(RealGPUInfo).init(self.allocator);
        defer integrated_gpus.deinit();

        var total_vram: u64 = 0;
        for (gpu_slice) |gpu| {
            if (gpu.is_discrete) {
                try discrete_gpus.append(gpu);
            } else {
                try integrated_gpus.append(gpu);
            }
            total_vram += gpu.memory_size;
        }

        const discrete_slice = try discrete_gpus.toOwnedSlice();
        errdefer self.allocator.free(discrete_slice);

        const integrated_slice = try integrated_gpus.toOwnedSlice();
        errdefer self.allocator.free(integrated_slice);

        const recommended_backend = determineRecommendedBackend(gpu_slice);

        return GPUDetectionResult{
            .allocator = self.allocator,
            .gpus = gpu_slice,
            .discrete_gpus = discrete_slice,
            .integrated_gpus = integrated_slice,
            .available_backends = backend_slice,
            .total_gpus = gpu_slice.len,
            .system_capabilities = .{
                .has_discrete_gpu = discrete_slice.len > 0,
                .has_integrated_gpu = integrated_slice.len > 0,
                .recommended_backend = recommended_backend,
                .total_vram = total_vram,
                .shared_memory_limit = getSystemMemoryLimit(),
            },
        };
    }

    /// Detect CUDA GPUs using the CUDA integration
    fn detectCUDAGPUs(self: *GPUDetector, gpus: *std.ArrayList(RealGPUInfo), available_backends: *std.ArrayList(BackendType)) !void {
        // Import CUDA integration
        const cuda = @import("libraries/cuda_integration.zig");

        // Try to initialize CUDA and get device count
        const init_result = cuda.cuInit(0);
        if (init_result != cuda.CUDA_SUCCESS) {
            return; // CUDA not available
        }

        var device_count: c_int = 0;
        if (cuda.cuDeviceGetCount(&device_count) != cuda.CUDA_SUCCESS or device_count <= 0) {
            return; // No CUDA devices
        }

        // Add CUDA to available backends
        try available_backends.append(.cuda);

        // Enumerate devices
        var i: c_int = 0;
        while (i < device_count) : (i += 1) {
            var device: c_int = 0;
            if (cuda.cuDeviceGet(&device, i) != cuda.CUDA_SUCCESS) {
                continue;
            }

            // Get device name
            var name_buf: [256]u8 = undefined;
            if (cuda.cuDeviceGetName(&name_buf, name_buf.len, device) != cuda.CUDA_SUCCESS) {
                continue;
            }
            const name_len = std.mem.indexOfScalar(u8, &name_buf, 0) orelse name_buf.len;
            const device_name = try self.allocator.dupe(u8, name_buf[0..name_len]);

            // Get memory size
            var total_memory: usize = 0;
            const mem_result = cuda.cuDeviceTotalMem_v2(&total_memory, device);
            if (mem_result != cuda.CUDA_SUCCESS) {
                self.allocator.free(device_name);
                continue;
            }

            // Get some basic attributes
            var multiprocessor_count: c_int = 0;
            var memory_clock_rate: c_int = 0;
            var memory_bus_width: c_int = 0;
            var max_threads_per_block: c_int = 0;
            var integrated: c_int = 0;
            var clock_rate: c_int = 0;
            var clock_rate_khz: c_int = 0;

            const attrs = [_]struct { attr: cuda.CuDeviceAttr, ptr: *c_int }{
                .{ .attr = cuda.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, .ptr = &multiprocessor_count },
                .{ .attr = cuda.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, .ptr = &memory_clock_rate },
                .{ .attr = cuda.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, .ptr = &memory_bus_width },
                .{ .attr = cuda.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, .ptr = &max_threads_per_block },
                .{ .attr = cuda.CU_DEVICE_ATTRIBUTE_INTEGRATED, .ptr = &integrated },
                .{ .attr = cuda.CU_DEVICE_ATTRIBUTE_CLOCK_RATE, .ptr = &clock_rate },
            };

            for (attrs) |attr_query| {
                _ = cuda.cuDeviceGetAttribute(attr_query.ptr, attr_query.attr, device);
            }

            // Convert clock rate from kHz to MHz
            clock_rate_khz = clock_rate;

            // Create backend list for this GPU
            var gpu_backends = try self.allocator.alloc(BackendType, 1);
            gpu_backends[0] = .cuda;

            // Create GPU info
            const gpu_info = RealGPUInfo{
                .name = device_name,
                .vendor = try self.allocator.dupe(u8, "NVIDIA"),
                .architecture = try self.allocator.dupe(u8, "CUDA"),
                .memory_type = try self.allocator.dupe(u8, "GDDR"),
                .manufacturing_process = try self.allocator.dupe(u8, "unknown"),
                .driver_version = try self.allocator.dupe(u8, "CUDA"),
                .opengl_version = try self.allocator.dupe(u8, "n/a"),
                .vulkan_version = try self.allocator.dupe(u8, "n/a"),
                .directx_version = try self.allocator.dupe(u8, "n/a"),
                .cuda_version = try self.allocator.dupe(u8, "available"),
                .opencl_version = try self.allocator.dupe(u8, "n/a"),
                .memory_size = total_memory,
                .compute_units = @intCast(multiprocessor_count),
                .max_clock_speed = @intCast(clock_rate_khz / 1000), // Convert kHz to MHz
                .shader_cores = @intCast(multiprocessor_count * 128), // Estimate
                .memory_clock_speed = @intCast(memory_clock_rate / 1000), // Convert to MHz
                .memory_bus_width = @intCast(memory_bus_width),
                .is_discrete = (integrated == 0),
                .is_integrated = (integrated != 0),
                .available_backends = gpu_backends,
                .memory_allocator = self.allocator,
            };

            try gpus.append(gpu_info);
        }
    }

    /// Detect Vulkan GPUs using the Vulkan integration
    fn detectVulkanGPUs(self: *GPUDetector, gpus: *std.ArrayList(RealGPUInfo), available_backends: *std.ArrayList(BackendType)) !void {
        const vulkan = @import("libraries/vulkan_integration.zig");

        // Try to create a Vulkan renderer to check availability
        const renderer = vulkan.VulkanRenderer.init(self.allocator) catch return;
        defer renderer.deinit();

        // Add Vulkan to available backends
        try available_backends.append(.vulkan);

        // Get device capabilities
        const capabilities = renderer.getDeviceInfo() catch return;

        // Extract memory size from Vulkan capabilities
        var total_memory: u64 = 0;
        for (capabilities.memory_heaps) |heap| {
            if (heap.flags.device_local) {
                total_memory = heap.size;
                break;
            }
        }

        // Extract compute units from Vulkan device limits
        // Use max_compute_work_group_invocations as a proxy for compute units
        const compute_units: u32 = capabilities.limits.max_compute_work_group_invocations / 256;

        // Convert Vulkan capabilities to our GPU info format
        const device_name = try self.allocator.dupe(u8, capabilities.device_name);
        errdefer self.allocator.free(device_name);

        var gpu_backends = try self.allocator.alloc(BackendType, 1);
        gpu_backends[0] = .vulkan;

        const gpu_info = RealGPUInfo{
            .name = device_name,
            .vendor = try self.allocator.dupe(u8, "Unknown"), // Could map vendor_id
            .architecture = try self.allocator.dupe(u8, "Vulkan"),
            .memory_type = try self.allocator.dupe(u8, "GPU Memory"),
            .manufacturing_process = try self.allocator.dupe(u8, "unknown"),
            .driver_version = try self.allocator.dupe(u8, "Vulkan"),
            .opengl_version = try self.allocator.dupe(u8, "n/a"),
            .vulkan_version = try self.allocator.dupe(u8, "available"),
            .directx_version = try self.allocator.dupe(u8, "n/a"),
            .cuda_version = try self.allocator.dupe(u8, "n/a"),
            .opencl_version = try self.allocator.dupe(u8, "n/a"),
            .memory_size = total_memory,
            .compute_units = compute_units,
            .max_clock_speed = 0, // Not directly available in Vulkan caps
            .shader_cores = compute_units, // Use compute_units as estimate
            .memory_clock_speed = 0,
            .memory_bus_width = 0,
            .is_discrete = (capabilities.device_type == .discrete_gpu),
            .is_integrated = (capabilities.device_type == .integrated_gpu),
            .available_backends = gpu_backends,
            .memory_allocator = self.allocator,
        };

        try gpus.append(gpu_info);
    }

    /// Detect Metal GPUs on macOS/iOS
    fn detectMetalGPUs(self: *GPUDetector, gpus: *std.ArrayList(RealGPUInfo), available_backends: *std.ArrayList(BackendType)) !void {
        // Metal is available on macOS/iOS by default
        // We'll create a synthetic GPU entry for Metal since direct Metal API
        // requires Objective-C/Swift bindings which aren't available in pure Zig

        try available_backends.append(.metal);

        // Detect if we have an Apple Silicon GPU (M1, M2, M3, etc.)
        const cpu_arch = builtin.cpu.arch;
        const is_apple_silicon = cpu_arch == .aarch64;

        var gpu_backends = try self.allocator.alloc(BackendType, 1);
        gpu_backends[0] = .metal;

        const device_name = if (is_apple_silicon) "Apple GPU" else "Unknown Metal GPU";

        const gpu_info = RealGPUInfo{
            .name = try self.allocator.dupe(u8, device_name),
            .vendor = try self.allocator.dupe(u8, "Apple"),
            .architecture = try self.allocator.dupe(u8, if (is_apple_silicon) "Apple Silicon" else "Metal"),
            .memory_type = try self.allocator.dupe(u8, "Unified Memory"),
            .manufacturing_process = try self.allocator.dupe(u8, if (is_apple_silicon) "5nm" else "unknown"),
            .driver_version = try self.allocator.dupe(u8, "Metal"),
            .opengl_version = try self.allocator.dupe(u8, "n/a"),
            .vulkan_version = try self.allocator.dupe(u8, "n/a"),
            .directx_version = try self.allocator.dupe(u8, "n/a"),
            .cuda_version = try self.allocator.dupe(u8, "n/a"),
            .opencl_version = try self.allocator.dupe(u8, "n/a"),
            .memory_size = if (is_apple_silicon) 8 * 1024 * 1024 * 1024 else 4 * 1024 * 1024 * 1024, // Conservative estimate
            .compute_units = if (is_apple_silicon) 7 else 4, // Estimated
            .max_clock_speed = if (is_apple_silicon) 1300 else 800, // MHz estimate
            .shader_cores = if (is_apple_silicon) 7 else 4, // Apple GPU cores
            .memory_clock_speed = if (is_apple_silicon) 6400 else 3200, // MHz
            .memory_bus_width = if (is_apple_silicon) 256 else 128, // Bit width
            .is_discrete = !is_apple_silicon,
            .is_integrated = is_apple_silicon,
            .available_backends = gpu_backends,
            .memory_allocator = self.allocator,
        };

        try gpus.append(gpu_info);
    }

    /// Detect DirectX 12 GPUs on Windows
    fn detectDirectX12GPUs(self: *GPUDetector, gpus: *std.ArrayList(RealGPUInfo), available_backends: *std.ArrayList(BackendType)) !void {
        // DirectX 12 is available on Windows 10+
        // We'll create a synthetic GPU entry since direct DirectX API
        // requires Windows COM which is complex to bind in Zig

        try available_backends.append(.directx12);

        var gpu_backends = try self.allocator.alloc(BackendType, 1);
        gpu_backends[0] = .directx12;

        const gpu_info = RealGPUInfo{
            .name = try self.allocator.dupe(u8, "DirectX 12 GPU"),
            .vendor = try self.allocator.dupe(u8, "Unknown"),
            .architecture = try self.allocator.dupe(u8, "DirectX 12"),
            .memory_type = try self.allocator.dupe(u8, "GPU Memory"),
            .manufacturing_process = try self.allocator.dupe(u8, "unknown"),
            .driver_version = try self.allocator.dupe(u8, "DirectX 12"),
            .opengl_version = try self.allocator.dupe(u8, "n/a"),
            .vulkan_version = try self.allocator.dupe(u8, "n/a"),
            .directx_version = try self.allocator.dupe(u8, "12.0"),
            .cuda_version = try self.allocator.dupe(u8, "n/a"),
            .opencl_version = try self.allocator.dupe(u8, "n/a"),
            .memory_size = 4 * 1024 * 1024 * 1024, // Conservative estimate
            .compute_units = 8, // Estimated
            .max_clock_speed = 1500, // MHz estimate
            .shader_cores = 0,
            .memory_clock_speed = 7000, // MHz estimate
            .memory_bus_width = 256, // Bit width estimate
            .is_discrete = true,
            .is_integrated = false,
            .available_backends = gpu_backends,
            .memory_allocator = self.allocator,
        };

        try gpus.append(gpu_info);
    }

    /// Mark OpenCL as available (cross-platform)
    /// Note: Real OpenCL device detection requires C API bindings and is not implemented.
    /// The OpenCL backend is still available for use but without specific device enumeration.
    fn detectOpenCLGPUs(self: *GPUDetector, gpus: *std.ArrayList(RealGPUInfo), available_backends: *std.ArrayList(BackendType)) !void {
        // No synthetic GPU entries created - OpenCL backend available but no device enumeration
        _ = self;
        _ = gpus;
        try available_backends.append(.opencl);
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
/// hardware probing. Returns true when CUDA or other real detection backends are available.
pub fn isHardwareDetectionAvailable() bool {
    // Check if CUDA is available
    if (BackendType.cuda.isAvailable()) {
        return true;
    }
    // Check if Vulkan is available
    const vulkan = @import("libraries/vulkan_integration.zig");
    if (vulkan.VulkanRenderer.isAvailable()) {
        return true;
    }
    // Check if Metal is available (macOS/iOS)
    if (BackendType.metal.isAvailable()) {
        return true;
    }
    // Check if DirectX 12 is available (Windows)
    if (BackendType.directx12.isAvailable()) {
        return true;
    }
    // Check if OpenCL is available (cross-platform)
    if (BackendType.opencl.isAvailable()) {
        return true;
    }
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

/// Query the system's total physical memory
pub fn getSystemMemoryLimit() u64 {
    switch (builtin.os.tag) {
        .windows => {
            const kernel32 = @cImport({
                @cDefine("WIN32_LEAN_AND_MEAN", "");
                @cInclude("windows.h");
            });

            var mem_status: kernel32.MEMORYSTATUSEX = undefined;
            mem_status.dwLength = @sizeOf(kernel32.MEMORYSTATUSEX);
            if (kernel32.GlobalMemoryStatusEx(&mem_status) != 0) {
                return mem_status.ullTotalPhys;
            }
        },
        .linux => {
            const stdlib = @cImport({
                @cInclude("sys/sysinfo.h");
            });

            var info: stdlib.struct_sysinfo = undefined;
            if (stdlib.sysinfo(&info) == 0) {
                return @as(u64, @intCast(info.totalram)) * @as(u64, @intCast(info.mem_unit));
            }
        },
        .macos, .ios => {
            const stdlib = @cImport({
                @cInclude("sys/types.h");
                @cInclude("sys/sysctl.h");
            });

            var mib: [2]c_int = [_]c_int{ stdlib.CTL_HW, stdlib.HW_MEMSIZE };
            var mem_size: u64 = 0;
            var len: usize = @sizeOf(u64);
            if (stdlib.sysctl(&mib, 2, &mem_size, &len, null, 0) == 0) {
                return mem_size;
            }
        },
        else => {},
    }

    // Fallback: return a reasonable default
    return 8 * 1024 * 1024 * 1024; // 8GB
}
