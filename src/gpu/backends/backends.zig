//! Multi-Backend GPU Support
//!
//! This module provides support for multiple GPU backends:
//! - Vulkan (cross-platform, high performance)
//! - Metal (Apple platforms)
//! - DirectX 12 (Windows)
//! - OpenGL (legacy fallback)
//! - CUDA (NVIDIA GPUs)
//! - OpenCL (cross-platform compute)
//! - WebGPU (web and WASM)
//! - CPU Fallback (software rendering)

const std = @import("std");
const gpu_renderer = @import("gpu_renderer.zig");

/// Supported GPU backends
pub const Backend = enum {
    vulkan,
    metal,
    dx12,
    opengl,
    cuda,
    opencl,
    webgpu,
    cpu_fallback,

    pub fn toString(self: Backend) []const u8 {
        return switch (self) {
            .vulkan => "Vulkan",
            .metal => "Metal",
            .dx12 => "DirectX 12",
            .opengl => "OpenGL",
            .cuda => "CUDA",
            .opencl => "OpenCL",
            .webgpu => "WebGPU",
            .cpu_fallback => "CPU Fallback",
        };
    }

    pub fn getPriority(self: Backend) u8 {
        return switch (self) {
            .vulkan => 10, // Highest priority - cross-platform, modern
            .cuda => 9, // NVIDIA optimized
            .metal => 8, // Apple optimized
            .dx12 => 7, // Windows optimized
            .webgpu => 6, // Web standard
            .opencl => 5, // Cross-platform compute
            .opengl => 3, // Legacy but widely supported
            .cpu_fallback => 1, // Always available but slow
        };
    }
};

/// Backend capabilities and features
pub const Capabilities = struct {
    name: []const u8,
    vendor: []const u8,
    version: []const u8,

    // Core capabilities
    compute_shaders: bool = false,
    geometry_shaders: bool = false,
    tessellation_shaders: bool = false,
    ray_tracing: bool = false,

    // Memory features
    unified_memory: bool = false,
    shared_memory: bool = false,
    memory_pools: bool = false,

    // Precision support
    supports_fp16: bool = false,
    supports_fp64: bool = false,
    supports_int8: bool = false,
    supports_int16: bool = false,
    supports_int64: bool = false,

    // Hardware acceleration
    tensor_cores: bool = false,
    rt_cores: bool = false,
    dlss: bool = false,

    // Performance limits
    max_workgroup_size: u32 = 1,
    max_workgroup_count: u32 = 1,
    max_compute_units: u32 = 1,
    max_memory_bandwidth_gb_s: u32 = 0,

    // Memory limits
    max_buffer_size: u64 = 0,
    max_texture_size: u32 = 0,
    max_texture_array_layers: u32 = 0,

    pub fn format(
        self: Capabilities,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("GPU: {s} ({s})\n", .{ self.name, self.vendor });
        try writer.print("Version: {s}\n", .{self.version});
        try writer.print("Compute Shaders: {}\n", .{self.compute_shaders});
        try writer.print("Ray Tracing: {}\n", .{self.ray_tracing});
        try writer.print("Unified Memory: {}\n", .{self.unified_memory});
        try writer.print("Tensor Cores: {}\n", .{self.tensor_cores});
        try writer.print("Max Workgroup Size: {}\n", .{self.max_workgroup_size});
        try writer.print("Max Compute Units: {}\n", .{self.max_compute_units});
        try writer.print("Max Memory Bandwidth: {} GB/s\n", .{self.max_memory_bandwidth_gb_s});
    }
};

/// Backend-specific configuration
pub const BackendConfig = union(Backend) {
    vulkan: VulkanConfig,
    metal: MetalConfig,
    dx12: DX12Config,
    opengl: OpenGLConfig,
    cuda: CUDAConfig,
    opencl: OpenCLConfig,
    webgpu: WebGPUConfig,
    cpu_fallback: CPUConfig,
};

/// Vulkan-specific configuration
pub const VulkanConfig = struct {
    enable_validation: bool = false,
    enable_debug_utils: bool = true,
    preferred_device_type: enum { discrete, integrated, any } = .any,
    required_extensions: []const []const u8 = &.{},
    optional_extensions: []const []const u8 = &.{},
};

/// Metal-specific configuration
pub const MetalConfig = struct {
    enable_validation: bool = false,
    enable_capture: bool = false,
    command_queue_priority: enum { normal, high } = .normal,
};

/// DirectX 12-specific configuration
pub const DX12Config = struct {
    enable_validation: bool = false,
    enable_debug_layer: bool = true,
    shader_model: enum { sm_6_0, sm_6_1, sm_6_2, sm_6_3, sm_6_4, sm_6_5, sm_6_6 } = .sm_6_0,
};

/// OpenGL-specific configuration
pub const OpenGLConfig = struct {
    version_major: u32 = 4,
    version_minor: u32 = 6,
    enable_validation: bool = false,
    core_profile: bool = true,
};

/// CUDA-specific configuration
pub const CUDAConfig = struct {
    device_id: u32 = 0,
    enable_peer_access: bool = false,
    enable_uvm: bool = true,
    kernel_cache_size_mb: u32 = 128,
};

/// OpenCL-specific configuration
pub const OpenCLConfig = struct {
    platform_preference: []const u8 = "",
    device_type: enum { gpu, cpu, accelerator, all } = .gpu,
    enable_profiling: bool = false,
};

/// WebGPU-specific configuration
pub const WebGPUConfig = struct {
    power_preference: enum { high_performance, low_power } = .high_performance,
    force_fallback_adapter: bool = false,
};

/// CPU fallback configuration
pub const CPUConfig = struct {
    thread_count: u32 = 0, // 0 = auto-detect
    enable_simd: bool = true,
    enable_parallel: bool = true,
};

/// Multi-Backend Manager
pub const BackendManager = struct {
    allocator: std.mem.Allocator,
    available_backends: std.ArrayList(Backend),
    current_backend: ?Backend = null,
    backend_configs: std.AutoHashMap(Backend, BackendConfig),

    pub fn init(allocator: std.mem.Allocator) !*BackendManager {
        const self = try allocator.create(BackendManager);
        self.* = .{
            .allocator = allocator,
            .available_backends = std.ArrayList(Backend){},
            .backend_configs = std.AutoHashMap(Backend, BackendConfig){},
        };

        // Detect available backends
        try self.detectAvailableBackends();

        // Set default configurations
        try self.setDefaultConfigs();

        return self;
    }

    pub fn deinit(self: *BackendManager) void {
        self.available_backends.deinit();
        self.backend_configs.deinit();
        self.allocator.destroy(self);
    }

    /// Detect which backends are available on this system
    pub fn detectAvailableBackends(self: *BackendManager) !void {
        // Vulkan detection
        if (self.detectVulkan()) {
            try self.available_backends.append(.vulkan);
        }

        // Metal detection (macOS/iOS only)
        if (self.detectMetal()) {
            try self.available_backends.append(.metal);
        }

        // DirectX 12 detection (Windows only)
        if (self.detectDX12()) {
            try self.available_backends.append(.dx12);
        }

        // OpenGL detection
        if (self.detectOpenGL()) {
            try self.available_backends.append(.opengl);
        }

        // CUDA detection
        if (self.detectCUDA()) {
            try self.available_backends.append(.cuda);
        }

        // OpenCL detection
        if (self.detectOpenCL()) {
            try self.available_backends.append(.opencl);
        }

        // WebGPU is always available (falls back to CPU if needed)
        try self.available_backends.append(.webgpu);

        // CPU fallback is always available
        try self.available_backends.append(.cpu_fallback);

        // Sort by priority
        std.mem.sort(Backend, self.available_backends.items, {}, struct {
            fn lessThan(_: void, a: Backend, b: Backend) bool {
                return a.getPriority() > b.getPriority(); // Higher priority first
            }
        }.lessThan);
    }

    /// Set default configurations for all backends
    pub fn setDefaultConfigs(self: *BackendManager) !void {
        // Vulkan defaults
        try self.backend_configs.put(.vulkan, BackendConfig{
            .vulkan = VulkanConfig{},
        });

        // Metal defaults
        try self.backend_configs.put(.metal, BackendConfig{
            .metal = MetalConfig{},
        });

        // DirectX 12 defaults
        try self.backend_configs.put(.dx12, BackendConfig{
            .dx12 = DX12Config{},
        });

        // OpenGL defaults
        try self.backend_configs.put(.opengl, BackendConfig{
            .opengl = OpenGLConfig{},
        });

        // CUDA defaults
        try self.backend_configs.put(.cuda, BackendConfig{
            .cuda = CUDAConfig{},
        });

        // OpenCL defaults
        try self.backend_configs.put(.opencl, BackendConfig{
            .opencl = OpenCLConfig{},
        });

        // WebGPU defaults
        try self.backend_configs.put(.webgpu, BackendConfig{
            .webgpu = WebGPUConfig{},
        });

        // CPU fallback defaults
        try self.backend_configs.put(.cpu_fallback, BackendConfig{
            .cpu_fallback = CPUConfig{},
        });
    }

    /// Select the best available backend
    pub fn selectBestBackend(self: *BackendManager) ?Backend {
        if (self.available_backends.items.len == 0) {
            return null;
        }

        self.current_backend = self.available_backends.items[0];
        return self.current_backend;
    }

    /// Force a specific backend
    pub fn selectBackend(self: *BackendManager, backend: Backend) !void {
        for (self.available_backends.items) |available| {
            if (available == backend) {
                self.current_backend = backend;
                return;
            }
        }

        return error.BackendNotAvailable;
    }

    /// Get capabilities for a backend
    pub fn getCapabilities(self: *BackendManager, backend: Backend) !Capabilities {
        return switch (backend) {
            .vulkan => self.getVulkanCapabilities(),
            .metal => self.getMetalCapabilities(),
            .dx12 => self.getDX12Capabilities(),
            .opengl => self.getOpenGLCapabilities(),
            .cuda => self.getCUDACapabilities(),
            .opencl => self.getOpenCLCapabilities(),
            .webgpu => self.getWebGPUCapabilities(),
            .cpu_fallback => self.getCPUCapabilities(),
        };
    }

    /// Create a renderer for the current backend
    pub fn createRenderer(self: *BackendManager, config: gpu_renderer.GPUConfig) !*gpu_renderer.GPURenderer {
        const backend = self.current_backend orelse return gpu_renderer.GpuError.InitializationFailed;

        // Modify config based on backend-specific settings
        var modified_config = config;
        if (self.backend_configs.get(backend)) |backend_config| {
            switch (backend_config) {
                .vulkan => |vulkan_config| {
                    modified_config.debug_validation = vulkan_config.enable_validation;
                },
                .webgpu => |webgpu_config| {
                    // WebGPU has its own power preference setting
                    _ = webgpu_config;
                },
                else => {},
            }
        }

        return gpu_renderer.GPURenderer.init(self.allocator, modified_config);
    }

    /// Get backend-specific capabilities (simplified implementations)
    fn getVulkanCapabilities(self: *BackendManager) Capabilities {
        _ = self;
        return .{
            .name = "Vulkan GPU",
            .vendor = "Khronos Group",
            .version = "1.3",
            .compute_shaders = true,
            .ray_tracing = true,
            .unified_memory = false,
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = true,
            .supports_int64 = true,
            .max_workgroup_size = 1024,
            .max_compute_units = 32,
            .max_memory_bandwidth_gb_s = 400,
            .max_buffer_size = 4 * 1024 * 1024 * 1024, // 4GB
            .max_texture_size = 16384,
        };
    }

    fn getMetalCapabilities(self: *BackendManager) Capabilities {
        _ = self;
        return .{
            .name = "Metal GPU",
            .vendor = "Apple",
            .version = "3.0",
            .compute_shaders = true,
            .ray_tracing = true,
            .unified_memory = false,
            .supports_fp16 = true,
            .supports_fp64 = false,
            .supports_int8 = true,
            .supports_int64 = true,
            .max_workgroup_size = 1024,
            .max_compute_units = 64,
            .max_memory_bandwidth_gb_s = 300,
            .max_buffer_size = 2 * 1024 * 1024 * 1024, // 2GB
            .max_texture_size = 16384,
        };
    }

    fn getDX12Capabilities(self: *BackendManager) Capabilities {
        _ = self;
        return .{
            .name = "DirectX 12 GPU",
            .vendor = "Microsoft",
            .version = "12.0",
            .compute_shaders = true,
            .ray_tracing = true,
            .unified_memory = false,
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = true,
            .supports_int64 = true,
            .max_workgroup_size = 1024,
            .max_compute_units = 40,
            .max_memory_bandwidth_gb_s = 350,
            .max_buffer_size = 4 * 1024 * 1024 * 1024, // 4GB
            .max_texture_size = 16384,
        };
    }

    fn getOpenGLCapabilities(self: *BackendManager) Capabilities {
        _ = self;
        return .{
            .name = "OpenGL GPU",
            .vendor = "Khronos Group",
            .version = "4.6",
            .compute_shaders = true,
            .ray_tracing = false,
            .unified_memory = false,
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = false,
            .supports_int64 = true,
            .max_workgroup_size = 1024,
            .max_compute_units = 16,
            .max_memory_bandwidth_gb_s = 100,
            .max_buffer_size = 1 * 1024 * 1024 * 1024, // 1GB
            .max_texture_size = 16384,
        };
    }

    fn getCUDACapabilities(self: *BackendManager) Capabilities {
        _ = self;
        return .{
            .name = "CUDA GPU",
            .vendor = "NVIDIA",
            .version = "12.0",
            .compute_shaders = true,
            .ray_tracing = true,
            .unified_memory = true,
            .tensor_cores = true,
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = true,
            .supports_int64 = true,
            .max_workgroup_size = 1024,
            .max_compute_units = 128,
            .max_memory_bandwidth_gb_s = 1000,
            .max_buffer_size = 16 * 1024 * 1024 * 1024, // 16GB
            .max_texture_size = 32768,
        };
    }

    fn getOpenCLCapabilities(self: *BackendManager) Capabilities {
        _ = self;
        return .{
            .name = "OpenCL Device",
            .vendor = "Khronos Group",
            .version = "3.0",
            .compute_shaders = true,
            .ray_tracing = false,
            .unified_memory = false,
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = true,
            .supports_int64 = true,
            .max_workgroup_size = 256,
            .max_compute_units = 64,
            .max_memory_bandwidth_gb_s = 200,
            .max_buffer_size = 4 * 1024 * 1024 * 1024, // 4GB
            .max_texture_size = 16384,
        };
    }

    fn getWebGPUCapabilities(self: *BackendManager) Capabilities {
        _ = self;
        return .{
            .name = "WebGPU Device",
            .vendor = "WebGPU Working Group",
            .version = "1.0",
            .compute_shaders = true,
            .ray_tracing = false,
            .unified_memory = false,
            .supports_fp16 = false,
            .supports_fp64 = false,
            .supports_int8 = false,
            .supports_int64 = false,
            .max_workgroup_size = 256,
            .max_compute_units = 16,
            .max_memory_bandwidth_gb_s = 50,
            .max_buffer_size = 1 * 1024 * 1024 * 1024, // 1GB
            .max_texture_size = 8192,
        };
    }

    fn getCPUCapabilities(self: *BackendManager) Capabilities {
        _ = self;
        return .{
            .name = "CPU Fallback",
            .vendor = "Software",
            .version = "1.0",
            .compute_shaders = false,
            .ray_tracing = false,
            .unified_memory = true,
            .supports_fp16 = true,
            .supports_fp64 = true,
            .supports_int8 = true,
            .supports_int64 = true,
            .max_workgroup_size = 1,
            .max_compute_units = std.Thread.getCpuCount() catch 4,
            .max_memory_bandwidth_gb_s = 10,
            .max_buffer_size = std.math.maxInt(u64), // Limited by system RAM
            .max_texture_size = 65536,
        };
    }

    // Backend detection functions
    fn detectVulkan(self: *BackendManager) bool {
        _ = self;
        // In a real implementation, this would check for Vulkan loader
        return true; // Assume available for demo
    }

    fn detectMetal(self: *BackendManager) bool {
        _ = self;
        return std.builtin.os.tag == .macos;
    }

    fn detectDX12(self: *BackendManager) bool {
        _ = self;
        return std.builtin.os.tag == .windows;
    }

    fn detectOpenGL(self: *BackendManager) bool {
        _ = self;
        return true; // Assume available
    }

    fn detectCUDA(self: *BackendManager) bool {
        _ = self;
        return false; // Not implemented yet
    }

    fn detectOpenCL(self: *BackendManager) bool {
        _ = self;
        return false; // Not implemented yet
    }
};

/// Backend-specific shader compiler
pub const ShaderCompiler = struct {
    allocator: std.mem.Allocator,
    backend: Backend,

    pub fn init(allocator: std.mem.Allocator, backend: Backend) !*ShaderCompiler {
        const self = try allocator.create(ShaderCompiler);
        self.* = .{
            .allocator = allocator,
            .backend = backend,
        };
        return self;
    }

    pub fn deinit(self: *ShaderCompiler) void {
        self.allocator.destroy(self);
    }

    /// Compile shader source to backend-specific format
    pub fn compileShader(self: *ShaderCompiler, source: []const u8, shader_type: enum { vertex, fragment, compute }) ![]const u8 {
        return switch (self.backend) {
            .vulkan => self.compileSPIRV(source, shader_type),
            .metal => self.compileMSL(source, shader_type),
            .dx12 => self.compileDXIL(source, shader_type),
            .opengl => self.compileGLSL(source, shader_type),
            .cuda => self.compilePTX(source),
            .opencl => self.compileSPIRV(source, shader_type),
            .webgpu => self.compileWGSL(source, shader_type),
            .cpu_fallback => error.ShaderCompilationNotSupported,
        };
    }

    // Compilation implementations (simplified)
    fn compileSPIRV(self: *ShaderCompiler, source: []const u8, shader_type: anytype) ![]const u8 {
        _ = self;
        _ = source;
        _ = shader_type;
        return error.NotImplementedYet;
    }

    fn compileMSL(self: *ShaderCompiler, source: []const u8, shader_type: anytype) ![]const u8 {
        _ = self;
        _ = source;
        _ = shader_type;
        return error.NotImplementedYet;
    }

    fn compileDXIL(self: *ShaderCompiler, source: []const u8, shader_type: anytype) ![]const u8 {
        _ = self;
        _ = source;
        _ = shader_type;
        return error.NotImplementedYet;
    }

    fn compileGLSL(self: *ShaderCompiler, source: []const u8, shader_type: anytype) ![]const u8 {
        _ = self;
        _ = source;
        _ = shader_type;
        return error.NotImplementedYet;
    }

    fn compilePTX(self: *ShaderCompiler, source: []const u8) ![]const u8 {
        _ = self;
        _ = source;
        return error.NotImplementedYet;
    }

    fn compileWGSL(self: *ShaderCompiler, source: []const u8, shader_type: anytype) ![]const u8 {
        _ = self;
        _ = source;
        _ = shader_type;
        return error.NotImplementedYet;
    }
};
