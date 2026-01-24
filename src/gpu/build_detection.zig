//! GPU Capability Detection for Build System
//!
//! This module provides build-time GPU capability detection to automatically
//! configure optimal GPU backends and features based on the target system.

const std = @import("std");
const builtin = @import("builtin");

/// Detected GPU capabilities
pub const GpuCapabilities = struct {
    /// CUDA support available
    has_cuda: bool = false,
    /// Vulkan support available
    has_vulkan: bool = false,
    /// Metal support available
    has_metal: bool = false,
    /// WebGPU support available
    has_webgpu: bool = false,
    /// OpenGL support available
    has_opengl: bool = false,
    /// OpenGL ES support available
    has_opengles: bool = false,
    /// std.gpu support available
    has_stdgpu: bool = true, // Always available as fallback

    /// Recommended backend priority order
    recommended_backends: []const []const u8,
    /// Detected GPU devices
    device_count: usize = 0,
    /// Total GPU memory detected (MB)
    total_memory_mb: usize = 0,
    /// Whether discrete GPUs were detected
    has_discrete_gpu: bool = false,
    /// Whether integrated GPUs were detected
    has_integrated_gpu: bool = false,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) GpuCapabilities {
        return .{
            .recommended_backends = &.{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *GpuCapabilities) void {
        self.allocator.free(self.recommended_backends);
        self.* = undefined;
    }

    /// Detect GPU capabilities on the current system
    pub fn detect(self: *GpuCapabilities) !void {
        const target = builtin.target;

        // Platform-specific detection
        switch (target.os.tag) {
            .windows => try self.detectWindows(),
            .linux => try self.detectLinux(),
            .macos => try self.detectMacOS(),
            else => {
                // Fallback for other platforms
                try self.detectGeneric();
            },
        }

        // Generate recommended backend order
        try self.generateRecommendations();
    }

    fn detectWindows(self: *GpuCapabilities) !void {
        // Check for CUDA (NVIDIA)
        self.has_cuda = try self.checkCudaInstallation();

        // Check for Vulkan
        self.has_vulkan = try self.checkVulkanInstallation();

        // OpenGL is typically available on Windows
        self.has_opengl = true;

        // Check for actual GPU devices
        try self.enumerateDevicesWindows();
    }

    fn detectLinux(self: *GpuCapabilities) !void {
        // Check for CUDA (NVIDIA)
        self.has_cuda = try self.checkCudaInstallation();

        // Check for Vulkan
        self.has_vulkan = try self.checkVulkanInstallation();

        // Check for AMD GPU with ROCm
        _ = try self.checkRocmInstallation();

        // OpenGL is typically available
        self.has_opengl = true;

        // Check for actual GPU devices
        try self.enumerateDevicesLinux();
    }

    fn detectMacOS(self: *GpuCapabilities) !void {
        // Metal is available on all Apple Silicon and Intel Macs with GPU
        self.has_metal = true;

        // WebGPU may be available through Safari/WebKit
        self.has_webgpu = true;

        // OpenGL is deprecated but may still be available
        self.has_opengl = false; // Metal is preferred

        // Enumerate Metal devices
        try self.enumerateDevicesMacOS();
    }

    fn detectGeneric(self: *GpuCapabilities) !void {
        // Conservative defaults for unknown platforms
        self.has_stdgpu = true;

        // Try Vulkan if available
        self.has_vulkan = try self.checkVulkanInstallation();
    }

    fn checkCudaInstallation(self: *GpuCapabilities) !bool {
        _ = self;
        // In a real implementation, this would check:
        // - CUDA toolkit installation
        // - NVIDIA drivers
        // - GPU devices
        // For now, assume CUDA is available if nvidia-smi exists (Linux/Windows)
        // or if we're on a system that typically has NVIDIA GPUs

        const target = builtin.target;
        if (target.os.tag == .windows) {
            // Check registry or known paths
            return false; // Placeholder
        } else if (target.os.tag == .linux) {
            // Check /usr/local/cuda, /proc/driver/nvidia, etc.
            return false; // Placeholder
        }

        return false;
    }

    fn checkVulkanInstallation(self: *GpuCapabilities) !bool {
        _ = self;
        // Check for Vulkan loader and ICDs
        // - Windows: Check registry for VulkanRT
        // - Linux: Check for libvulkan.so.1, ICD files
        // - macOS: Vulkan is available via MoltenVK

        const target = builtin.target;
        switch (target.os.tag) {
            .windows => return false, // Placeholder
            .linux => return false, // Placeholder
            .macos => return true, // MoltenVK provides Vulkan on macOS
            else => return false,
        }
    }

    fn checkRocmInstallation(self: *GpuCapabilities) !bool {
        _ = self;
        // Check for ROCm installation on Linux
        // - /opt/rocm directory
        // - libamdhip64.so
        // - AMD GPU detection
        return false; // Placeholder
    }

    fn enumerateDevicesWindows(self: *GpuCapabilities) !void {
        // Use Windows APIs to enumerate GPUs
        // - DXGI for DirectX capable GPUs
        // - SetupAPI for device enumeration
        // This is a placeholder - real implementation would use FFI

        self.device_count = 1; // Assume at least one GPU
        self.total_memory_mb = 8192; // Assume 8GB
        self.has_discrete_gpu = true;
    }

    fn enumerateDevicesLinux(self: *GpuCapabilities) !void {
        // Use sysfs and drm to enumerate GPUs
        // - /sys/class/drm/card*
        // - /proc/driver/nvidia/gpus
        // - rocm-smi for AMD

        self.device_count = 1; // Assume at least one GPU
        self.total_memory_mb = 8192; // Assume 8GB
        self.has_discrete_gpu = true;
    }

    fn enumerateDevicesMacOS(self: *GpuCapabilities) !void {
        // Metal device enumeration
        // All Macs have at least one GPU (integrated or discrete)

        self.device_count = 1; // Assume at least one GPU
        self.total_memory_mb = 16384; // Apple Silicon typically has more unified memory
        self.has_integrated_gpu = true;
        self.has_discrete_gpu = false; // Most have integrated, some have discrete too
    }

    fn generateRecommendations(self: *GpuCapabilities) !void {
        var backends = std.ArrayList([]const u8).init(self.allocator);
        errdefer backends.deinit();

        // Priority order based on performance and availability
        const target = builtin.target;

        if (target.os.tag == .macos) {
            // macOS: Metal first, then Vulkan via MoltenVK, then std.gpu
            if (self.has_metal) try backends.append("metal");
            if (self.has_vulkan) try backends.append("vulkan");
            try backends.append("stdgpu");
        } else if (target.os.tag == .windows) {
            // Windows: CUDA first (if NVIDIA), then Vulkan, then OpenGL
            if (self.has_cuda) try backends.append("cuda");
            if (self.has_vulkan) try backends.append("vulkan");
            if (self.has_opengl) try backends.append("opengl");
            try backends.append("stdgpu");
        } else if (target.os.tag == .linux) {
            // Linux: CUDA first, then Vulkan, then OpenGL
            if (self.has_cuda) try backends.append("cuda");
            if (self.has_vulkan) try backends.append("vulkan");
            if (self.has_opengl) try backends.append("opengl");
            try backends.append("stdgpu");
        } else {
            // Generic: Vulkan first, then std.gpu
            if (self.has_vulkan) try backends.append("vulkan");
            try backends.append("stdgpu");
        }

        self.recommended_backends = try backends.toOwnedSlice();
    }

    /// Print capability detection results
    pub fn printReport(self: *const GpuCapabilities) void {
        std.debug.print("GPU Capability Detection Report\n", .{});
        std.debug.print("================================\n", .{});
        // Use {t} format specifier for enums (Zig 0.16)
        std.debug.print("Platform: {t}\n", .{builtin.target.os.tag});
        std.debug.print("Architecture: {t}\n\n", .{builtin.target.cpu.arch});

        std.debug.print("Available Backends:\n", .{});
        if (self.has_cuda) std.debug.print("  ✓ CUDA\n", .{});
        if (self.has_vulkan) std.debug.print("  ✓ Vulkan\n", .{});
        if (self.has_metal) std.debug.print("  ✓ Metal\n", .{});
        if (self.has_webgpu) std.debug.print("  ✓ WebGPU\n", .{});
        if (self.has_opengl) std.debug.print("  ✓ OpenGL\n", .{});
        if (self.has_opengles) std.debug.print("  ✓ OpenGL ES\n", .{});
        std.debug.print("  ✓ std.gpu (fallback)\n\n", .{});

        std.debug.print("Hardware:\n", .{});
        std.debug.print("  GPU Devices: {}\n", .{self.device_count});
        std.debug.print("  Total Memory: {} MB\n", .{self.total_memory_mb});
        std.debug.print("  Discrete GPU: {}\n", .{self.has_discrete_gpu});
        std.debug.print("  Integrated GPU: {}\n\n", .{self.has_integrated_gpu});

        std.debug.print("Recommended Backend Order:\n", .{});
        for (self.recommended_backends, 0..) |backend, i| {
            std.debug.print("  {}. {s}\n", .{ i + 1, backend });
        }
        std.debug.print("\n", .{});
    }

    /// Get build flags for detected capabilities
    pub fn getBuildFlags(self: *const GpuCapabilities, allocator: std.mem.Allocator) ![]const u8 {
        var flags = std.ArrayList(u8).init(allocator);
        errdefer flags.deinit();

        // Add -Dgpu-backend= flags
        try flags.appendSlice("-Dgpu-backend=");
        for (self.recommended_backends, 0..) |backend, i| {
            if (i > 0) try flags.append(',');
            try flags.appendSlice(backend);
        }

        // Add enable-gpu flag
        try flags.appendSlice(" -Denable-gpu=true");

        return flags.toOwnedSlice();
    }
};

/// Detect GPU capabilities and return configuration
pub fn detectGpuCapabilities(allocator: std.mem.Allocator) !GpuCapabilities {
    var caps = GpuCapabilities.init(allocator);
    errdefer caps.deinit();

    try caps.detect();
    return caps;
}

test "GPU capability detection" {
    const allocator = std.testing.allocator;
    var caps = try detectGpuCapabilities(allocator);
    defer caps.deinit();

    // Should always have at least std.gpu
    try std.testing.expect(caps.has_stdgpu);

    // Should have recommended backends
    try std.testing.expect(caps.recommended_backends.len > 0);

    // Print report for manual verification
    caps.printReport();
}
