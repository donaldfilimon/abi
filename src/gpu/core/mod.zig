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
//! - core/ - Core GPU functionality
//! - compute/ - Compute operations and kernels
//! - memory/ - Memory management and pooling
//! - backends/ - Multi-backend support
//! - benchmark/ - Performance testing
//! - demo/ - Demo applications and shaders

const std = @import("std");

// Re-export main GPU components from core
pub const GPURenderer = @import("gpu_renderer.zig").GPURenderer;
pub const GPUConfig = @import("gpu_renderer.zig").GPUConfig;
pub const GpuError = @import("gpu_renderer.zig").GpuError;
pub const Color = @import("gpu_renderer.zig").Color;
pub const GPUHandle = @import("gpu_renderer.zig").GPUHandle;
pub const Backend = @import("gpu_renderer.zig").Backend;
pub const PowerPreference = @import("gpu_renderer.zig").PowerPreference;
pub const has_webgpu_support = @import("gpu_renderer.zig").has_webgpu_support;

// Re-export backend components from core
pub const GpuBackend = @import("backend.zig").GpuBackend;
pub const GpuBackendConfig = @import("backend.zig").GpuBackendConfig;
pub const GpuBackendError = @import("backend.zig").GpuBackend.Error;
pub const BatchConfig = @import("backend.zig").BatchConfig;
pub const BatchProcessor = @import("backend.zig").BatchProcessor;
pub const GpuStats = @import("backend.zig").GpuStats;

// Re-export database interface from core
pub const Db = @import("backend.zig").Db;

// Re-export compute features
pub const KernelManager = @import("../compute/kernels.zig").KernelManager;
pub const GPUBackendManager = @import("../compute/gpu_backend_manager.zig").GPUBackendManager;
pub const CUDADriver = @import("../compute/gpu_backend_manager.zig").CUDADriver;
pub const SPIRVCompiler = @import("../compute/gpu_backend_manager.zig").SPIRVCompiler;
pub const BackendType = @import("../compute/gpu_backend_manager.zig").BackendType;
pub const HardwareCapabilities = @import("../compute/gpu_backend_manager.zig").HardwareCapabilities;

// Re-export memory management
pub const MemoryPool = @import("../memory/memory_pool.zig").MemoryPool;

// Re-export backend support
pub const BackendSupport = @import("../backends/backends.zig").BackendSupport;

// Re-export benchmarking tools
pub const PerformanceProfiler = @import("../benchmark/benchmarks.zig").PerformanceProfiler;
pub const MemoryBandwidthBenchmark = @import("../benchmark/benchmarks.zig").MemoryBandwidthBenchmark;
pub const ComputeThroughputBenchmark = @import("../benchmark/benchmarks.zig").ComputeThroughputBenchmark;

// Re-export commonly used types
pub const Allocator = std.mem.Allocator;

/// Initialize the GPU system with default configuration
pub fn initDefault(allocator: std.mem.Allocator) !*GPURenderer {
    const config = GPUConfig{
        .debug_validation = false,
        .power_preference = .high_performance,
        .backend = .auto,
        .try_webgpu_first = true,
    };
    return GPURenderer.init(allocator, config);
}

/// Check if GPU acceleration is available
pub fn isGpuAvailable() bool {
    // This is a simplified check - in practice you'd initialize and check
    return @import("gpu_renderer.zig").has_webgpu_support;
}

test "GPU module imports" {
    // Test that all main types are accessible
    _ = GPURenderer;
    _ = GPUConfig;
    _ = GpuError;
    _ = GpuBackend;
    _ = GpuBackendConfig;
}
