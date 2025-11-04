//! GPU Libraries Integration Module
//!
//! This module provides integration with advanced GPU libraries for enhanced
//! graphics capabilities and cross-platform compatibility.

pub const vulkan_bindings = @import("vulkan_bindings.zig");
pub const mach_gpu_integration = @import("mach_gpu_integration.zig");
pub const cuda_integration = @import("cuda_integration.zig");
pub const simd_optimizations = @import("simd_optimizations.zig");

// Re-export key types and functions
pub const VulkanRenderer = vulkan_bindings.VulkanRenderer;
pub const VulkanCapabilities = vulkan_bindings.VulkanCapabilities;
pub const VulkanUtils = vulkan_bindings.VulkanUtils;
pub const AdvancedVulkanFeatures = vulkan_bindings.AdvancedVulkanFeatures;

pub const MachRenderer = mach_gpu_integration.MachRenderer;
pub const MachCapabilities = mach_gpu_integration.MachCapabilities;
pub const MachUtils = mach_gpu_integration.MachUtils;

pub const CUDARenderer = cuda_integration.CUDARenderer;
pub const CUDACapabilities = cuda_integration.CUDACapabilities;
pub const CUDAUtils = cuda_integration.CUDAUtils;

pub const VectorTypes = simd_optimizations.VectorTypes;
pub const SIMDMath = simd_optimizations.SIMDMath;
pub const SIMDGraphics = simd_optimizations.SIMDGraphics;
pub const SIMDCompute = simd_optimizations.SIMDCompute;
pub const SIMDBenchmarks = simd_optimizations.SIMDBenchmarks;

const std = @import("std");

/// Unified GPU library manager
pub const GPULibraryManager = struct {
    allocator: std.mem.Allocator,
    vulkan_renderer: ?VulkanRenderer = null,
    mach_renderer: ?MachRenderer = null,
    cuda_renderer: ?CUDARenderer = null,
    available_libraries: AvailableLibraries = .{},

    const Self = @This();

    pub const AvailableLibraries = packed struct {
        vulkan: bool = false,
        mach_gpu: bool = false,
        cuda: bool = false,
        simd: bool = true, // Always available
        _padding: u28 = 0,
    };

    pub fn init(allocator: std.mem.Allocator) !Self {
        var manager = Self{
            .allocator = allocator,
        };

        // Detect available libraries
        manager.available_libraries.vulkan = VulkanUtils.isVulkanAvailable();
        manager.available_libraries.mach_gpu = MachUtils.isMachGPUAvailable();
        manager.available_libraries.cuda = CUDAUtils.isCUDAAvailable();

        return manager;
    }

    pub fn deinit(self: *Self) void {
        if (self.vulkan_renderer) |*renderer| {
            renderer.deinit();
        }
        if (self.mach_renderer) |*renderer| {
            renderer.deinit();
        }
        if (self.cuda_renderer) |*renderer| {
            renderer.deinit();
        }
    }

    /// Initialize Vulkan renderer
    pub fn initVulkan(self: *Self) !void {
        if (!self.available_libraries.vulkan) {
            return error.VulkanNotAvailable;
        }

        self.vulkan_renderer = try VulkanRenderer.init(self.allocator);
        try self.vulkan_renderer.?.initialize();
    }

    /// Initialize Mach/GPU renderer
    pub fn initMachGPU(self: *Self, device_type: mach_gpu_integration.MachDeviceType) !void {
        if (!self.available_libraries.mach_gpu) {
            return error.MachGPUNotAvailable;
        }

        self.mach_renderer = try MachRenderer.init(self.allocator);
        try self.mach_renderer.?.initialize(device_type);
    }

    /// Initialize CUDA renderer
    pub fn initCUDA(self: *Self) !void {
        if (!self.available_libraries.cuda) {
            return error.CUDANotAvailable;
        }

        self.cuda_renderer = try CUDARenderer.init(self.allocator);
        try self.cuda_renderer.?.initialize();
    }

    /// Get available libraries information
    pub fn getAvailableLibraries(self: *Self) AvailableLibraries {
        return self.available_libraries;
    }

    /// Run SIMD benchmarks
    pub fn runSIMDBenchmarks(self: *Self) !void {
        if (!self.available_libraries.simd) {
            return error.SIMDNotAvailable;
        }

        std.log.info("🚀 Running SIMD Performance Benchmarks", .{});

        // Benchmark array operations
        try SIMDBenchmarks.benchmarkSIMDvsScalar(self.allocator, 1000000);
        try SIMDBenchmarks.benchmarkMatrixOperations(self.allocator);
    }

    /// Get comprehensive library status
    pub fn getLibraryStatus(self: *Self) LibraryStatus {
        return LibraryStatus{
            .vulkan = if (self.vulkan_renderer != null) .initialized else if (self.available_libraries.vulkan) .available else .unavailable,
            .mach_gpu = if (self.mach_renderer != null) .initialized else if (self.available_libraries.mach_gpu) .available else .unavailable,
            .cuda = if (self.cuda_renderer != null) .initialized else if (self.available_libraries.cuda) .available else .unavailable,
            .simd = .available,
        };
    }

    pub const LibraryStatus = struct {
        vulkan: LibraryState,
        mach_gpu: LibraryState,
        cuda: LibraryState,
        simd: LibraryState,

        pub const LibraryState = enum {
            unavailable,
            available,
            initialized,
        };
    };
};

/// Error types for GPU library operations
pub const GPULibraryError = error{
    VulkanNotAvailable,
    MachGPUNotAvailable,
    CUDANotAvailable,
    SIMDNotAvailable,
    InitializationFailed,
    InvalidDevice,
    UnsupportedFeature,
    OutOfMemory,
};
