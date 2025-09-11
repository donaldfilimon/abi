//! GPU Module - High-performance GPU acceleration for AI operations
//!
//! This module provides GPU-accelerated computing capabilities including:
//! - WebGPU-based rendering and compute
//! - GPU backend for vector operations
//! - High-performance neural network computations
//! - Memory-efficient buffer management
//! - Cross-platform compatibility (Desktop + WASM)
//!
//! Organized structure:
//! - core/ - Core GPU functionality (renderer, backend)
//! - compute/ - Compute operations and kernels (CUDA, SPIRV)
//! - memory/ - Memory management and pooling
//! - backends/ - Multi-backend support
//! - benchmark/ - Performance testing and profiling
//! - demo/ - Demo applications and shaders
//!
//! Refactored for Zig 0.15 with modern patterns

const std = @import("std");

// Conditional compilation based on GPU support
const gpu_enabled = @import("root").abi.gpu;

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

// Additional utilities and helpers
pub const utils = struct {
    /// Check if any GPU acceleration is available
    pub fn isAccelerationAvailable() bool {
        return core.isGpuAvailable();
    }

    /// Get recommended GPU configuration
    pub fn getRecommendedConfig(_: std.mem.Allocator) !core.GPUConfig {
        return .{
            .debug_validation = false,
            .power_preference = .high_performance,
            .backend = .auto,
            .try_webgpu_first = true,
        };
    }

    /// Initialize GPU system with recommended settings
    pub fn initRecommended(allocator: std.mem.Allocator) !*core.GPURenderer {
        const config = try getRecommendedConfig(allocator);
        return core.GPURenderer.init(allocator, config);
    }
};

// Version information
pub const version = struct {
    pub const major = 1;
    pub const minor = 0;
    pub const patch = 0;

    pub const string = std.fmt.comptimePrint("{}.{}.{}", .{ major, minor, patch });
};

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
