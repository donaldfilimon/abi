//! Enhanced Backend Detection and Selection
//!
//! This module provides comprehensive backend detection for CUDA, OpenCL, DirectML,
//! and other GPU backends with dynamic selection based on hardware capabilities.

const std = @import("std");
const builtin = @import("builtin");
const gpu = @import("../mod.zig");

/// Enhanced backend detection system
pub const BackendDetector = struct {
    allocator: std.mem.Allocator,
    detected_backends: std.ArrayList(BackendInfo),
    recommended_backend: ?BackendType = null,

    const Self = @This();

    pub const BackendType = enum {
        cuda,
        opencl,
        directml,
        vulkan,
        metal,
        d3d12,
        opengl,
        webgpu,
        auto,
    };

    pub const BackendInfo = struct {
        backend_type: BackendType,
        is_available: bool,
        version: BackendVersion,
        capabilities: BackendCapabilities,
        performance_score: f32,
        memory_size: u64,
        compute_units: u32,
        vendor: []const u8,
        device_name: []const u8,

        pub const BackendVersion = struct {
            major: u32,
            minor: u32,
            patch: u32,
        };

        pub const BackendCapabilities = struct {
            supports_compute: bool = false,
            supports_graphics: bool = false,
            supports_raytracing: bool = false,
            supports_mesh_shaders: bool = false,
            supports_variable_rate_shading: bool = false,
            supports_async_compute: bool = false,
            supports_multi_gpu: bool = false,
            supports_shared_memory: bool = false,
            supports_atomic_operations: bool = false,
            supports_double_precision: bool = false,
            supports_half_precision: bool = false,
            supports_int64: bool = false,
            max_work_group_size: u32 = 0,
            max_work_item_dimensions: u32 = 0,
            max_work_item_sizes: [3]u32 = .{ 0, 0, 0 },
            max_memory_alloc_size: u64 = 0,
            max_constant_buffer_size: u64 = 0,
            max_texture_size: u32 = 0,
            max_image_2d_size: [2]u32 = .{ 0, 0 },
            max_image_3d_size: [3]u32 = .{ 0, 0, 0 },
        };
    };

    pub fn init(allocator: std.mem.Allocator) !Self {
        return Self{
            .allocator = allocator,
            .detected_backends = std.ArrayList(BackendInfo).initCapacity(allocator, 8) catch return error.OutOfMemory,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.detected_backends.items) |*backend| {
            self.allocator.free(backend.vendor);
            self.allocator.free(backend.device_name);
        }
        self.detected_backends.deinit(self.allocator);
    }

    /// Detect all available backends
    pub fn detectAllBackends(self: *Self) !void {
        try self.detectAllBackendsWithOptions(.{});
    }

    pub const DetectionOptions = struct {
        force_disable: bool = false,
    };

    pub fn detectAllBackendsWithOptions(self: *Self, options: DetectionOptions) !void {
        self.clearDetectedBackends();
        if (options.force_disable) {
            self.recommended_backend = null;
            return;
        }

        self.appendBackendIfAvailable(self.detectCUDA()) catch return error.OutOfMemory;
        self.appendBackendIfAvailable(self.detectOpenCL()) catch return error.OutOfMemory;
        self.appendBackendIfAvailable(self.detectDirectML()) catch return error.OutOfMemory;
        self.appendBackendIfAvailable(self.detectVulkan()) catch return error.OutOfMemory;
        self.appendBackendIfAvailable(self.detectMetal()) catch return error.OutOfMemory;
        self.appendBackendIfAvailable(self.detectD3D12()) catch return error.OutOfMemory;
        self.appendBackendIfAvailable(self.detectOpenGL()) catch return error.OutOfMemory;
        self.appendBackendIfAvailable(self.detectWebGPU()) catch return error.OutOfMemory;

        self.selectRecommendedBackend();
    }

    /// Detect CUDA backend
    fn detectCUDA(self: *Self) ?BackendInfo {
        const candidates = switch (builtin.os.tag) {
            .windows => &[_][]const u8{"nvcuda.dll"},
            .linux => &[_][]const u8{ "libcuda.so", "libcuda.so.1" },
            else => &[_][]const u8{},
        };
        if (!self.tryOpenLibrary(candidates)) {
            return null;
        }

        return BackendInfo{
            .backend_type = .cuda,
            .is_available = true,
            .version = .{ .major = 12, .minor = 0, .patch = 0 },
            .capabilities = .{
                .supports_compute = true,
                .supports_graphics = false,
                .supports_raytracing = true,
                .supports_mesh_shaders = true,
                .supports_variable_rate_shading = false,
                .supports_async_compute = true,
                .supports_multi_gpu = true,
                .supports_shared_memory = true,
                .supports_atomic_operations = true,
                .supports_double_precision = true,
                .supports_half_precision = true,
                .supports_int64 = true,
                .max_work_group_size = 1024,
                .max_work_item_dimensions = 3,
                .max_work_item_sizes = .{ 1024, 1024, 64 },
                .max_memory_alloc_size = 8 * 1024 * 1024 * 1024, // 8GB
                .max_constant_buffer_size = 64 * 1024, // 64KB
                .max_texture_size = 16384,
                .max_image_2d_size = .{ 16384, 16384 },
                .max_image_3d_size = .{ 16384, 16384, 16384 },
            },
            .performance_score = 90.0,
            .memory_size = 0,
            .compute_units = 0,
            .vendor = self.allocator.dupe(u8, "NVIDIA") catch return null,
            .device_name = self.allocator.dupe(u8, "CUDA Device") catch return null,
        };
    }

    /// Detect OpenCL backend
    fn detectOpenCL(self: *Self) ?BackendInfo {
        const candidates = switch (builtin.os.tag) {
            .windows => &[_][]const u8{"OpenCL.dll"},
            .linux => &[_][]const u8{ "libOpenCL.so", "libOpenCL.so.1" },
            .macos => &[_][]const u8{"/System/Library/Frameworks/OpenCL.framework/OpenCL"},
            else => &[_][]const u8{},
        };
        if (!self.tryOpenLibrary(candidates)) {
            return null;
        }

        return BackendInfo{
            .backend_type = .opencl,
            .is_available = true,
            .version = .{ .major = 3, .minor = 0, .patch = 0 },
            .capabilities = .{
                .supports_compute = true,
                .supports_graphics = false,
                .supports_raytracing = false,
                .supports_mesh_shaders = false,
                .supports_variable_rate_shading = false,
                .supports_async_compute = true,
                .supports_multi_gpu = true,
                .supports_shared_memory = true,
                .supports_atomic_operations = true,
                .supports_double_precision = true,
                .supports_half_precision = true,
                .supports_int64 = true,
                .max_work_group_size = 1024,
                .max_work_item_dimensions = 3,
                .max_work_item_sizes = .{ 1024, 1024, 64 },
                .max_memory_alloc_size = 4 * 1024 * 1024 * 1024, // 4GB
                .max_constant_buffer_size = 64 * 1024, // 64KB
                .max_texture_size = 8192,
                .max_image_2d_size = .{ 8192, 8192 },
                .max_image_3d_size = .{ 2048, 2048, 2048 },
            },
            .performance_score = 70.0,
            .memory_size = 0,
            .compute_units = 0,
            .vendor = self.allocator.dupe(u8, "Unknown") catch return null,
            .device_name = self.allocator.dupe(u8, "OpenCL Device") catch return null,
        };
    }

    /// Detect DirectML backend
    fn detectDirectML(self: *Self) ?BackendInfo {
        const candidates = switch (builtin.os.tag) {
            .windows => &[_][]const u8{"DirectML.dll"},
            else => &[_][]const u8{},
        };
        if (!self.tryOpenLibrary(candidates)) {
            return null;
        }

        return BackendInfo{
            .backend_type = .directml,
            .is_available = true,
            .version = .{ .major = 1, .minor = 8, .patch = 0 },
            .capabilities = .{
                .supports_compute = true,
                .supports_graphics = false,
                .supports_raytracing = false,
                .supports_mesh_shaders = false,
                .supports_variable_rate_shading = false,
                .supports_async_compute = true,
                .supports_multi_gpu = false,
                .supports_shared_memory = false,
                .supports_atomic_operations = false,
                .supports_double_precision = false,
                .supports_half_precision = true,
                .supports_int64 = false,
                .max_work_group_size = 1024,
                .max_work_item_dimensions = 3,
                .max_work_item_sizes = .{ 1024, 1024, 64 },
                .max_memory_alloc_size = 2 * 1024 * 1024 * 1024, // 2GB
                .max_constant_buffer_size = 16 * 1024, // 16KB
                .max_texture_size = 4096,
                .max_image_2d_size = .{ 4096, 4096 },
                .max_image_3d_size = .{ 1024, 1024, 1024 },
            },
            .performance_score = 55.0,
            .memory_size = 0,
            .compute_units = 0,
            .vendor = self.allocator.dupe(u8, "Microsoft") catch return null,
            .device_name = self.allocator.dupe(u8, "DirectML Device") catch return null,
        };
    }

    /// Detect Vulkan backend
    fn detectVulkan(self: *Self) ?BackendInfo {
        const candidates = switch (builtin.os.tag) {
            .windows => &[_][]const u8{"vulkan-1.dll"},
            .linux => &[_][]const u8{ "libvulkan.so.1", "libvulkan.so" },
            .macos => &[_][]const u8{"libvulkan.dylib"},
            else => &[_][]const u8{},
        };
        if (!self.tryOpenLibrary(candidates)) {
            return null;
        }

        return BackendInfo{
            .backend_type = .vulkan,
            .is_available = true,
            .version = .{ .major = 1, .minor = 3, .patch = 0 },
            .capabilities = .{
                .supports_compute = true,
                .supports_graphics = true,
                .supports_raytracing = true,
                .supports_mesh_shaders = true,
                .supports_variable_rate_shading = true,
                .supports_async_compute = true,
                .supports_multi_gpu = true,
                .supports_shared_memory = true,
                .supports_atomic_operations = true,
                .supports_double_precision = true,
                .supports_half_precision = true,
                .supports_int64 = true,
                .max_work_group_size = 1024,
                .max_work_item_dimensions = 3,
                .max_work_item_sizes = .{ 1024, 1024, 64 },
                .max_memory_alloc_size = 16 * 1024 * 1024 * 1024, // 16GB
                .max_constant_buffer_size = 64 * 1024, // 64KB
                .max_texture_size = 16384,
                .max_image_2d_size = .{ 16384, 16384 },
                .max_image_3d_size = .{ 16384, 16384, 16384 },
            },
            .performance_score = 85.0,
            .memory_size = 0,
            .compute_units = 0,
            .vendor = self.allocator.dupe(u8, "Unknown") catch return null,
            .device_name = self.allocator.dupe(u8, "Vulkan Device") catch return null,
        };
    }

    /// Detect Metal backend
    fn detectMetal(self: *Self) ?BackendInfo {
        const candidates = switch (builtin.os.tag) {
            .macos => &[_][]const u8{"/System/Library/Frameworks/Metal.framework/Metal"},
            else => &[_][]const u8{},
        };
        if (!self.tryOpenLibrary(candidates)) {
            return null;
        }

        return BackendInfo{
            .backend_type = .metal,
            .is_available = true,
            .version = .{ .major = 3, .minor = 0, .patch = 0 },
            .capabilities = .{
                .supports_compute = true,
                .supports_graphics = true,
                .supports_raytracing = true,
                .supports_mesh_shaders = true,
                .supports_variable_rate_shading = false,
                .supports_async_compute = true,
                .supports_multi_gpu = false,
                .supports_shared_memory = true,
                .supports_atomic_operations = true,
                .supports_double_precision = true,
                .supports_half_precision = true,
                .supports_int64 = true,
                .max_work_group_size = 1024,
                .max_work_item_dimensions = 3,
                .max_work_item_sizes = .{ 1024, 1024, 64 },
                .max_memory_alloc_size = 8 * 1024 * 1024 * 1024, // 8GB
                .max_constant_buffer_size = 32 * 1024, // 32KB
                .max_texture_size = 16384,
                .max_image_2d_size = .{ 16384, 16384 },
                .max_image_3d_size = .{ 16384, 16384, 16384 },
            },
            .performance_score = 80.0,
            .memory_size = 0,
            .compute_units = 0,
            .vendor = self.allocator.dupe(u8, "Apple") catch return null,
            .device_name = self.allocator.dupe(u8, "Metal Device") catch return null,
        };
    }

    /// Detect D3D12 backend
    fn detectD3D12(self: *Self) ?BackendInfo {
        const candidates = switch (builtin.os.tag) {
            .windows => &[_][]const u8{"d3d12.dll"},
            else => &[_][]const u8{},
        };
        if (!self.tryOpenLibrary(candidates)) {
            return null;
        }

        return BackendInfo{
            .backend_type = .d3d12,
            .is_available = true,
            .version = .{ .major = 12, .minor = 0, .patch = 0 },
            .capabilities = .{
                .supports_compute = true,
                .supports_graphics = true,
                .supports_raytracing = true,
                .supports_mesh_shaders = true,
                .supports_variable_rate_shading = true,
                .supports_async_compute = true,
                .supports_multi_gpu = true,
                .supports_shared_memory = true,
                .supports_atomic_operations = true,
                .supports_double_precision = true,
                .supports_half_precision = true,
                .supports_int64 = true,
                .max_work_group_size = 1024,
                .max_work_item_dimensions = 3,
                .max_work_item_sizes = .{ 1024, 1024, 64 },
                .max_memory_alloc_size = 12 * 1024 * 1024 * 1024, // 12GB
                .max_constant_buffer_size = 64 * 1024, // 64KB
                .max_texture_size = 16384,
                .max_image_2d_size = .{ 16384, 16384 },
                .max_image_3d_size = .{ 16384, 16384, 16384 },
            },
            .performance_score = 82.0,
            .memory_size = 0,
            .compute_units = 0,
            .vendor = self.allocator.dupe(u8, "Microsoft") catch return null,
            .device_name = self.allocator.dupe(u8, "D3D12 Device") catch return null,
        };
    }

    /// Detect OpenGL backend
    fn detectOpenGL(self: *Self) ?BackendInfo {
        const candidates = switch (builtin.os.tag) {
            .windows => &[_][]const u8{"opengl32.dll"},
            .linux => &[_][]const u8{ "libGL.so.1", "libGL.so" },
            .macos => &[_][]const u8{"/System/Library/Frameworks/OpenGL.framework/OpenGL"},
            else => &[_][]const u8{},
        };
        if (!self.tryOpenLibrary(candidates)) {
            return null;
        }

        return BackendInfo{
            .backend_type = .opengl,
            .is_available = true,
            .version = .{ .major = 4, .minor = 6, .patch = 0 },
            .capabilities = .{
                .supports_compute = true,
                .supports_graphics = true,
                .supports_raytracing = false,
                .supports_mesh_shaders = false,
                .supports_variable_rate_shading = false,
                .supports_async_compute = false,
                .supports_multi_gpu = false,
                .supports_shared_memory = false,
                .supports_atomic_operations = true,
                .supports_double_precision = true,
                .supports_half_precision = true,
                .supports_int64 = false,
                .max_work_group_size = 1024,
                .max_work_item_dimensions = 3,
                .max_work_item_sizes = .{ 1024, 1024, 64 },
                .max_memory_alloc_size = 4 * 1024 * 1024 * 1024, // 4GB
                .max_constant_buffer_size = 16 * 1024, // 16KB
                .max_texture_size = 16384,
                .max_image_2d_size = .{ 16384, 16384 },
                .max_image_3d_size = .{ 2048, 2048, 2048 },
            },
            .performance_score = 45.0,
            .memory_size = 0,
            .compute_units = 0,
            .vendor = self.allocator.dupe(u8, "Unknown") catch return null,
            .device_name = self.allocator.dupe(u8, "OpenGL Device") catch return null,
        };
    }

    /// Detect WebGPU backend
    fn detectWebGPU(self: *Self) ?BackendInfo {
        const candidates = switch (builtin.os.tag) {
            .windows => &[_][]const u8{ "wgpu_native.dll", "webgpu.dll" },
            .linux => &[_][]const u8{ "libwgpu_native.so", "libwebgpu.so" },
            .macos => &[_][]const u8{ "libwgpu_native.dylib", "libwebgpu.dylib" },
            else => &[_][]const u8{},
        };
        if (!self.tryOpenLibrary(candidates)) {
            return null;
        }

        return BackendInfo{
            .backend_type = .webgpu,
            .is_available = true,
            .version = .{ .major = 1, .minor = 0, .patch = 0 },
            .capabilities = .{
                .supports_compute = true,
                .supports_graphics = true,
                .supports_raytracing = false,
                .supports_mesh_shaders = false,
                .supports_variable_rate_shading = false,
                .supports_async_compute = true,
                .supports_multi_gpu = false,
                .supports_shared_memory = false,
                .supports_atomic_operations = true,
                .supports_double_precision = false,
                .supports_half_precision = true,
                .supports_int64 = false,
                .max_work_group_size = 256,
                .max_work_item_dimensions = 3,
                .max_work_item_sizes = .{ 256, 256, 64 },
                .max_memory_alloc_size = 1 * 1024 * 1024 * 1024, // 1GB
                .max_constant_buffer_size = 16 * 1024, // 16KB
                .max_texture_size = 4096,
                .max_image_2d_size = .{ 4096, 4096 },
                .max_image_3d_size = .{ 1024, 1024, 1024 },
            },
            .performance_score = 40.0,
            .memory_size = 0,
            .compute_units = 0,
            .vendor = self.allocator.dupe(u8, "WebGPU") catch return null,
            .device_name = self.allocator.dupe(u8, "WebGPU Device") catch return null,
        };
    }

    /// Select the recommended backend based on performance and capabilities
    fn selectRecommendedBackend(self: *Self) void {
        if (self.detected_backends.items.len == 0) {
            self.recommended_backend = null;
            return;
        }

        var best_backend: ?BackendType = null;
        var best_score: f32 = 0.0;

        for (self.detected_backends.items) |backend| {
            if (!backend.is_available) continue;

            var score = backend.performance_score;

            // Bonus for compute capabilities
            if (backend.capabilities.supports_compute) score += 10.0;
            if (backend.capabilities.supports_graphics) score += 5.0;
            if (backend.capabilities.supports_raytracing) score += 15.0;
            if (backend.capabilities.supports_mesh_shaders) score += 10.0;
            if (backend.capabilities.supports_async_compute) score += 5.0;
            if (backend.capabilities.supports_multi_gpu) score += 10.0;

            // Bonus for memory size
            score += @as(f32, @floatFromInt(backend.memory_size / (1024 * 1024 * 1024))) * 0.5;

            // Bonus for compute units
            score += @as(f32, @floatFromInt(backend.compute_units)) * 0.1;

            if (score > best_score) {
                best_score = score;
                best_backend = backend.backend_type;
            }
        }

        self.recommended_backend = best_backend;
    }

    /// Get the recommended backend
    pub fn getRecommendedBackend(self: *Self) ?BackendType {
        return self.recommended_backend;
    }

    /// Get all detected backends
    pub fn getDetectedBackends(self: *Self) []const BackendInfo {
        return self.detected_backends.items;
    }

    /// Get backend info by type
    pub fn getBackendInfo(self: *Self, backend_type: BackendType) ?*BackendInfo {
        for (self.detected_backends.items) |*backend| {
            if (backend.backend_type == backend_type) {
                return backend;
            }
        }
        return null;
    }

    /// Check if a specific backend is available
    pub fn isBackendAvailable(self: *Self, backend_type: BackendType) bool {
        if (self.getBackendInfo(backend_type)) |backend| {
            return backend.is_available;
        }
        return false;
    }

    fn clearDetectedBackends(self: *Self) void {
        for (self.detected_backends.items) |*backend| {
            self.allocator.free(backend.vendor);
            self.allocator.free(backend.device_name);
        }
        self.detected_backends.clearRetainingCapacity();
    }

    fn appendBackendIfAvailable(self: *Self, backend_info: ?BackendInfo) !void {
        const info = backend_info orelse return;
        if (!info.is_available) {
            return;
        }
        try self.detected_backends.append(self.allocator, info);
    }

    fn tryOpenLibrary(self: *Self, names: []const []const u8) bool {
        _ = self;
        const DynLib = std.DynLib;
        for (names) |name| {
            var lib = DynLib.openZ(name) catch continue;
            defer lib.close();
            return true;
        }
        return false;
    }
};

test "gpu backend detection: forced disable yields empty list" {
    const testing = std.testing;
    var detector = try BackendDetector.init(testing.allocator);
    defer detector.deinit();

    try detector.detectAllBackendsWithOptions(.{ .force_disable = true });
    try testing.expectEqual(@as(usize, 0), detector.getDetectedBackends().len);
    try testing.expect(detector.getRecommendedBackend() == null);
}
