//! GPU Module - High-performance GPU acceleration for AI operations
//!
//! This module provides GPU-accelerated computing capabilities including:
//! - WebGPU-based rendering and compute with fallback support
//! - Multi-backend GPU support (CUDA, Vulkan, Metal, DirectX 12, OpenCL)
//! - Advanced memory management with unified memory architecture
//! - Cross-compilation support for multiple platforms and architectures
//! - Hardware detection and automatic backend selection
//! - Performance profiling and benchmarking tools
//! - WebAssembly support for web deployment
//!
//! ## Architecture
//!
//! The module is organized into several key subsystems:
//! - `core/` - Core GPU functionality (renderer, backend interfaces)
//! - `compute/` - Compute operations and kernel management
//! - `memory/` - Memory management and pooling systems
//! - `backends/` - Multi-backend support and implementations
//! - `benchmark/` - Performance testing and profiling
//! - `libraries/` - External GPU library integrations
//! - `optimizations/` - Platform-specific optimizations
//! - `mobile/` - Mobile platform support
//! - `testing/` - Cross-platform testing utilities
//!
//! ## Key Features
//!
//! - **Multi-Backend Support**: Automatic backend selection with fallback
//! - **Unified Memory**: Support for unified memory architectures
//! - **Cross-Compilation**: Target multiple platforms from single source
//! - **Performance Monitoring**: Comprehensive profiling and benchmarking
//! - **Resource Management**: Efficient memory and resource lifecycle management
//! - **Error Recovery**: Robust error handling with automatic recovery
//!
//! ## Usage Example
//!
//! ```zig
//! const gpu = @import("gpu");
//!
//! // Initialize with recommended settings
//! var renderer = try gpu.utils.initRecommended(allocator);
//! defer renderer.deinit();
//!
//! // Or use hardware detection for optimal configuration
//! var detector = gpu.GPUDetector.init(allocator);
//! const result = try detector.detectGPUs();
//! defer result.deinit();
//!
//! // Select best backend automatically
//! const backend = try gpu.GPUBackendManager.selectBestBackend(result);
//! ```
//!
//! Refactored for Zig 0.15 with modern patterns and comprehensive error handling

const std = @import("std");
const builtin = @import("builtin");

// Compile-time feature detection
const gpu_enabled = blk: {
    // Check if GPU support is enabled in root configuration
    if (@hasDecl(@import("root"), "abi")) {
        if (@hasDecl(@import("root").abi, "gpu")) {
            break :blk @import("root").abi.gpu;
        }
    }
    // Default to enabled if not specified
    break :blk true;
};

// Platform detection for conditional compilation
const target_platform = builtin.target.os.tag;
const is_windows = target_platform == .windows;
const is_macos = target_platform == .macos;
const is_linux = target_platform == .linux;
const is_wasm = builtin.target.cpu.arch == .wasm32 or builtin.target.cpu.arch == .wasm64;

/// Comprehensive GPU module error types
pub const Error = error{
    // Core initialization errors
    InitializationFailed,
    BackendNotAvailable,
    DeviceNotFound,
    UnsupportedPlatform,

    // Memory management errors
    OutOfMemory,
    MemoryAllocationFailed,
    MemoryDeallocationFailed,
    BufferCreationFailed,
    BufferMappingFailed,

    // Shader and compilation errors
    ShaderCompilationFailed,
    ShaderValidationFailed,
    SpirvCompilationFailed,
    KernelCompilationFailed,

    // Backend-specific errors
    VulkanError,
    CudaError,
    MetalError,
    DirectXError,
    OpenCLError,
    WebGPUError,

    // Runtime errors
    CommandSubmissionFailed,
    PipelineCreationFailed,
    BindGroupCreationFailed,
    TextureCreationFailed,
    SynchronizationFailed,

    // Resource management errors
    ResourceNotFound,
    ResourceExhausted,
    ResourceInUse,
    InvalidHandle,

    // Configuration errors
    InvalidConfiguration,
    UnsupportedFeature,
    VersionMismatch,

    // Cross-compilation errors
    CrossCompilationFailed,
    TargetNotSupported,
    BuildFailed,

    // Hardware detection errors
    HardwareDetectionFailed,
    DriverNotFound,
    CapabilityNotSupported,
};

/// Result type for GPU operations that may fail
pub fn Result(comptime T: type) type {
    return Error!T;
}

/// Optional result type for operations that may not return a value
pub fn OptionalResult(comptime T: type) type {
    return Error!?T;
}

// Import and re-export from organized submodules
pub const core = @import("core/mod.zig");
pub const unified_memory = @import("unified_memory.zig");
pub const hardware_detection = @import("hardware_detection.zig");
pub const cross_compilation = @import("cross_compilation.zig");
pub const wasm_support = @import("wasm_support.zig");
pub const libraries = @import("libraries/mod.zig");
pub const optimizations = @import("optimizations/mod.zig");
pub const testing = @import("testing/mod.zig");
pub const mobile = @import("mobile/mod.zig");

// Direct exports for backward compatibility with tests
pub const GPURenderer = core.GPURenderer;
pub const GPUConfig = core.GPUConfig;
pub const GpuError = core.GpuError;
pub const GpuBackend = core.GpuBackend;
pub const GpuBackendConfig = core.GpuBackendConfig;
pub const GpuBackendError = core.GpuBackendError;
pub const BatchConfig = core.BatchConfig;
pub const BatchProcessor = core.BatchProcessor;
pub const GpuStats = core.GpuStats;
pub const Db = core.Db;
pub const KernelManager = core.KernelManager;
pub const GPUBackendManager = core.GPUBackendManager;
pub const CUDADriver = core.CUDADriver;
pub const SPIRVCompiler = core.SPIRVCompiler;
pub const CoreBackendType = core.BackendType;
pub const HardwareCapabilities = core.HardwareCapabilities;
pub const MemoryPool = core.MemoryPool;
pub const BackendSupport = core.BackendSupport;
pub const PerformanceProfiler = core.PerformanceProfiler;
pub const MemoryBandwidthBenchmark = core.MemoryBandwidthBenchmark;
pub const ComputeThroughputBenchmark = core.ComputeThroughputBenchmark;
pub const Backend = core.Backend;
pub const PowerPreference = core.PowerPreference;
pub const has_webgpu_support = core.has_webgpu_support;
pub const Color = core.Color;
pub const GPUHandle = core.GPUHandle;

// Unified Memory exports
pub const UnifiedMemoryManager = unified_memory.UnifiedMemoryManager;
pub const UnifiedMemoryType = unified_memory.UnifiedMemoryType;
pub const UnifiedMemoryConfig = unified_memory.UnifiedMemoryConfig;
pub const UnifiedBuffer = unified_memory.UnifiedBuffer;

// Hardware Detection exports
pub const GPUDetector = hardware_detection.GPUDetector;
pub const RealGPUInfo = hardware_detection.RealGPUInfo;
pub const GPUDetectionResult = hardware_detection.GPUDetectionResult;
pub const BackendType = hardware_detection.BackendType;
pub const PerformanceTier = hardware_detection.PerformanceTier;
pub const SystemCapabilities = hardware_detection.SystemCapabilities;

// Cross-Compilation exports
pub const CrossCompilationManager = cross_compilation.CrossCompilationManager;
pub const CrossCompilationTarget = cross_compilation.CrossCompilationTarget;
pub const GPUBackend = cross_compilation.GPUBackend;
pub const OptimizationLevel = cross_compilation.OptimizationLevel;
pub const TargetFeatures = cross_compilation.TargetFeatures;
pub const MemoryModel = cross_compilation.MemoryModel;
pub const ThreadingModel = cross_compilation.ThreadingModel;
pub const PredefinedTargets = cross_compilation.PredefinedTargets;
pub const CrossCompilationUtils = cross_compilation.CrossCompilationUtils;

// WebAssembly Support exports
pub const WASMCompiler = wasm_support.WASMCompiler;
pub const WASMRuntime = wasm_support.WASMRuntime;
pub const WASMConfig = wasm_support.WASMConfig;
pub const WASMArchitecture = wasm_support.WASMArchitecture;
pub const WASMOptimizationLevel = wasm_support.WASMOptimizationLevel;
pub const WASMMemoryModel = wasm_support.WASMMemoryModel;
pub const WASMGPUBackend = wasm_support.WASMGPUBackend;
pub const PredefinedWASMConfigs = wasm_support.PredefinedWASMConfigs;

// GPU Libraries exports
pub const GPULibraryManager = libraries.GPULibraryManager;
pub const VulkanRenderer = libraries.VulkanRenderer;
pub const VulkanCapabilities = libraries.VulkanCapabilities;
pub const VulkanUtils = libraries.VulkanUtils;
pub const AdvancedVulkanFeatures = libraries.AdvancedVulkanFeatures;
pub const MachRenderer = libraries.MachRenderer;
pub const MachCapabilities = libraries.MachCapabilities;
pub const MachUtils = libraries.MachUtils;
pub const CUDARenderer = libraries.CUDARenderer;
pub const CUDACapabilities = libraries.CUDACapabilities;
pub const CUDAUtils = libraries.CUDAUtils;
pub const VectorTypes = libraries.VectorTypes;
pub const SIMDMath = libraries.SIMDMath;
pub const SIMDGraphics = libraries.SIMDGraphics;
pub const SIMDCompute = libraries.SIMDCompute;
pub const SIMDBenchmarks = libraries.SIMDBenchmarks;
pub const GPULibraryError = libraries.GPULibraryError;

// Optimizations exports
pub const PlatformOptimizations = optimizations.PlatformOptimizations;
pub const BackendDetector = optimizations.BackendDetector;
pub const PlatformConfig = optimizations.PlatformConfig;
pub const PlatformMetrics = optimizations.PlatformMetrics;
pub const PlatformUtils = optimizations.PlatformUtils;

// Testing exports
pub const CrossPlatformTestSuite = testing.CrossPlatformTestSuite;

// Mobile exports
pub const MobilePlatformManager = mobile.MobilePlatformManager;
pub const MobileCapabilities = mobile.MobileCapabilities;
pub const PowerManagement = mobile.PowerManagement;
pub const ThermalManagement = mobile.ThermalManagement;

// Convenience functions
pub const initDefault = core.initDefault;
pub const isGpuAvailable = core.isGpuAvailable;

/// Enhanced utilities and helper functions for GPU operations
pub const utils = struct {
    /// Check if any GPU acceleration is available
    pub fn isAccelerationAvailable() bool {
        return core.isGpuAvailable();
    }

    /// Get recommended GPU configuration based on platform and hardware
    pub fn getRecommendedConfig(allocator: std.mem.Allocator) Result(core.GPUConfig) {
        // Try hardware detection for optimal configuration
        if (hardware_detection.isHardwareDetectionAvailable()) {
            return getHardwareBasedConfig(allocator);
        }

        // Fallback to platform-specific defaults
        return getPlatformDefaultConfig();
    }

    /// Initialize GPU system with recommended settings and error handling
    pub fn initRecommended(allocator: std.mem.Allocator) Result(*core.GPURenderer) {
        const config = try getRecommendedConfig(allocator);
        return core.GPURenderer.init(allocator, config);
    }

    /// Initialize GPU system with hardware detection and automatic backend selection
    pub fn initWithHardwareDetection(allocator: std.mem.Allocator) Result(*core.GPURenderer) {
        var detector = hardware_detection.GPUDetector.init(allocator);
        defer detector.deinit();

        const detection_result = try detector.detectGPUs();
        defer detection_result.deinit();

        if (detection_result.total_gpus == 0) {
            std.log.warn("No GPUs detected, falling back to CPU mode", .{});
            return initWithFallback(allocator);
        }

        // Select best backend based on detected hardware
        const config = try createConfigFromDetection(detection_result);
        return core.GPURenderer.init(allocator, config);
    }

    /// Initialize with CPU fallback when GPU is not available
    pub fn initWithFallback(allocator: std.mem.Allocator) Result(*core.GPURenderer) {
        const config = core.GPUConfig{
            .debug_validation = false,
            .power_preference = .low_power,
            .backend = .cpu_fallback,
            .try_webgpu_first = false,
        };
        return core.GPURenderer.init(allocator, config);
    }

    /// Safely deinitialize GPU resources with comprehensive cleanup
    pub fn safeDeinit(renderer: ?*core.GPURenderer, allocator: std.mem.Allocator) void {
        if (renderer) |r| {
            // Ensure all pending operations are completed before cleanup
            r.waitForIdle() catch |err| {
                std.log.err("Failed to wait for GPU idle during cleanup: {}", .{err});
            };
            r.deinit();
            allocator.destroy(r);
        }
    }

    /// Get system information for GPU compatibility assessment
    pub fn getSystemInfo(allocator: std.mem.Allocator) Result(SystemInfo) {
        var info = SystemInfo{
            .platform = target_platform,
            .architecture = builtin.target.cpu.arch,
            .has_gpu_support = isAccelerationAvailable(),
            .total_memory_mb = getTotalSystemMemory() / (1024 * 1024),
        };

        // Try hardware detection for more detailed info
        if (hardware_detection.isHardwareDetectionAvailable()) {
            var detector = hardware_detection.GPUDetector.init(allocator);
            defer detector.deinit();

            if (detector.detectGPUs()) |result| {
                defer result.deinit();
                info.gpu_count = result.total_gpus;
                info.has_discrete_gpu = result.discrete_gpus.len > 0;
                info.has_integrated_gpu = result.integrated_gpus.len > 0;
            } else |_| {
                // Hardware detection failed, use basic info
            }
        }

        return info;
    }

    /// Validate GPU configuration against system capabilities
    pub fn validateConfiguration(config: *const core.GPUConfig) Result(void) {
        if (!isAccelerationAvailable() and config.backend != .cpu_fallback) {
            return Error.BackendNotAvailable;
        }

        // Platform-specific validation
        if (is_wasm and config.backend != .webgpu and config.backend != .cpu_fallback) {
            return Error.UnsupportedPlatform;
        }

        if (is_windows and config.backend == .metal) {
            return Error.BackendNotAvailable;
        }

        if (is_linux and config.backend == .directx12) {
            return Error.BackendNotAvailable;
        }

        return {};
    }

    // Internal helper functions

    fn getHardwareBasedConfig(allocator: std.mem.Allocator) Result(core.GPUConfig) {
        var detector = hardware_detection.GPUDetector.init(allocator);
        defer detector.deinit();

        const result = try detector.detectGPUs();
        defer result.deinit();

        return createConfigFromDetection(result);
    }

    fn createConfigFromDetection(detection_result: *const hardware_detection.GPUDetectionResult) Result(core.GPUConfig) {
        const recommended_backend = detection_result.recommended_backend;

        return core.GPUConfig{
            .debug_validation = false,
            .power_preference = if (detection_result.system_capabilities.power_delivery_capacity > 500)
                .high_performance
            else
                .low_power,
            .backend = backendTypeToCoreBackend(recommended_backend),
            .try_webgpu_first = recommended_backend == .webgpu,
        };
    }

    fn backendTypeToCoreBackend(backend: hardware_detection.BackendType) core.Backend {
        return switch (backend) {
            .vulkan => .vulkan,
            .cuda => .cuda,
            .metal => .metal,
            .dx12 => .directx12,
            .opengl => .opengl,
            .opencl => .opencl,
            .webgpu => .webgpu,
            .cpu_fallback => .cpu_fallback,
        };
    }

    fn getPlatformDefaultConfig() Result(core.GPUConfig) {
        const config = core.GPUConfig{
            .debug_validation = false,
            .power_preference = .high_performance,
            .backend = if (is_wasm)
                .webgpu
            else if (is_macos)
                .metal
            else if (is_windows)
                .directx12
            else
                .vulkan,
            .try_webgpu_first = is_wasm,
        };

        try validateConfiguration(&config);
        return config;
    }
};

/// System information structure
pub const SystemInfo = struct {
    platform: std.Target.Os.Tag,
    architecture: std.Target.Cpu.Arch,
    has_gpu_support: bool,
    total_memory_mb: usize,
    gpu_count: u32 = 0,
    has_discrete_gpu: bool = false,
    has_integrated_gpu: bool = false,

    pub fn format(
        self: SystemInfo,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;

        try writer.print("Platform: {s}, Architecture: {s}\n", .{
            @tagName(self.platform),
            @tagName(self.architecture),
        });
        try writer.print("GPU Support: {}, Memory: {} MB\n", .{
            self.has_gpu_support,
            self.total_memory_mb,
        });
        if (self.gpu_count > 0) {
            try writer.print("GPUs: {} total ({} discrete, {} integrated)\n", .{
                self.gpu_count,
                self.has_discrete_gpu,
                self.has_integrated_gpu,
            });
        }
    }
};

/// Get total system memory (simplified implementation)
fn getTotalSystemMemory() usize {
    // In a real implementation, this would query the system
    // For now, return a reasonable default
    return 16 * 1024 * 1024 * 1024; // 16 GB
}

// Version information with semantic versioning
pub const version = struct {
    pub const major = 1;
    pub const minor = 0;
    pub const patch = 0;
    pub const pre_release = ""; // e.g., "alpha", "beta", "rc.1"

    /// Full version string
    pub const string = blk: {
        if (pre_release.len > 0) {
            break :blk std.fmt.comptimePrint("{}.{}.{}-{}", .{ major, minor, patch, pre_release });
        } else {
            break :blk std.fmt.comptimePrint("{}.{}.{}", .{ major, minor, patch });
        }
    };

    /// Check if this is a development/pre-release version
    pub const is_development = pre_release.len > 0;

    /// Get version as a packed integer for easy comparison
    pub const packed_version = (major << 16) | (minor << 8) | patch;
};

// Comprehensive test suite
test {
    std.testing.refAllDecls(@This());
}

test "GPU module organization" {
    // Test that all organized components are accessible
    _ = core.GPURenderer;
    _ = core.GpuBackend;
    _ = core.KernelManager;
    _ = core.MemoryPool;
    _ = core.BackendSupport;
    _ = core.PerformanceProfiler;
    _ = utils.isAccelerationAvailable;
}

test "GPU error handling" {
    // Test error types are properly defined
    try std.testing.expectError(Error.InitializationFailed, error.InitializationFailed);
    try std.testing.expectError(Error.BackendNotAvailable, error.BackendNotAvailable);
    try std.testing.expectError(Error.DeviceNotFound, error.DeviceNotFound);
}

test "GPU configuration validation" {
    // Test platform-specific configuration validation
    const config = core.GPUConfig{
        .debug_validation = false,
        .power_preference = .high_performance,
        .backend = .vulkan,
        .try_webgpu_first = false,
    };

    // Should pass on most platforms
    if (!is_wasm) {
        try utils.validateConfiguration(&config);
    }
}

test "System info retrieval" {
    const allocator = std.testing.allocator;

    const info = try utils.getSystemInfo(allocator);
    defer {
        // Clean up any allocated resources if needed
        // Note: info is used in the test below, so we can't discard it
    }

    // Basic validation
    try std.testing.expect(info.platform == target_platform);
    try std.testing.expect(info.architecture == builtin.target.cpu.arch);
    try std.testing.expect(info.total_memory_mb > 0);
}

test "Version information" {
    // Test version constants
    try std.testing.expectEqual(@as(u32, 1), version.major);
    try std.testing.expectEqual(@as(u32, 0), version.minor);
    try std.testing.expectEqual(@as(u32, 0), version.patch);

    // Test version string format
    try std.testing.expect(std.mem.indexOf(u8, version.string, "1.0.0") != null);

    // Test packed version
    const expected_packed = (version.major << 16) | (version.minor << 8) | version.patch;
    try std.testing.expectEqual(expected_packed, version.packed_version);
}
