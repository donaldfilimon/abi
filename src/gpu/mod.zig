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

// Import and re-export from organized submodules
pub const core = @import("core/mod.zig");

// Additional utilities and helpers
pub const utils = struct {
    /// Check if any GPU acceleration is available
    pub fn isAccelerationAvailable() bool {
        return core.isGpuAvailable();
    }

    /// Get recommended GPU configuration
    pub fn getRecommendedConfig(allocator: std.mem.Allocator) !core.GPUConfig {
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
