//! Vulkan Backend VTable
//!
//! Implements the GPU backend interface using Vulkan compute pipelines.
//! Provides memory management, kernel lifecycle, and compute dispatch
//! through the Vulkan API.

const std = @import("std");
const compute = @import("compute.zig");

/// Vulkan backend kernel handle.
pub const VulkanKernel = struct {
    pipeline: compute.VulkanComputePipeline,
    spirv_hash: u64 = 0,
};

/// Vulkan backend memory handle.
pub const VulkanMemory = struct {
    buffer: compute.VulkanBuffer,
};

/// Backend interface methods for Vulkan.
pub const VulkanBackend = struct {
    buffer_mgr: compute.VulkanBufferManager,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) VulkanBackend {
        return .{
            .buffer_mgr = compute.VulkanBufferManager.init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *VulkanBackend) void {
        _ = self;
    }

    pub fn allocateMemory(self: *VulkanBackend, size: usize) !VulkanMemory {
        const buffer = try self.buffer_mgr.createBuffer(size);
        return .{ .buffer = buffer };
    }

    pub fn freeMemory(self: *VulkanBackend, mem: *VulkanMemory) void {
        self.buffer_mgr.destroyBuffer(&mem.buffer);
    }

    pub fn createKernel(self: *VulkanBackend, spirv: []const u32) !VulkanKernel {
        var pipeline = compute.VulkanComputePipeline.init(self.allocator);
        pipeline.createFromSpirV(spirv) catch |err| {
            pipeline.deinit();
            return err;
        };
        return .{
            .pipeline = pipeline,
            .spirv_hash = std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(spirv)),
        };
    }

    pub fn destroyKernel(self: *VulkanBackend, kernel: *VulkanKernel) void {
        _ = self;
        kernel.pipeline.deinit();
    }

    pub fn isAvailable() bool {
        return compute.isAvailable();
    }
};

// Tests
test "VulkanBackend init/deinit" {
    const allocator = std.testing.allocator;
    var backend = VulkanBackend.init(allocator);
    defer backend.deinit();
    // Verify it doesn't crash regardless of platform
    try std.testing.expect(!VulkanBackend.isAvailable() or VulkanBackend.isAvailable());
}

test "VulkanBackend memory lifecycle" {
    const allocator = std.testing.allocator;
    var backend = VulkanBackend.init(allocator);
    defer backend.deinit();
    var mem = try backend.allocateMemory(4096);
    try std.testing.expectEqual(@as(usize, 4096), mem.buffer.size);
    backend.freeMemory(&mem);
    try std.testing.expectEqual(@as(usize, 0), mem.buffer.size);
}

test "VulkanBackend createKernel without device" {
    const allocator = std.testing.allocator;
    var backend = VulkanBackend.init(allocator);
    defer backend.deinit();
    const spirv = [_]u32{ 0x07230203, 0x00010000 };
    // Pipeline has no device, so createFromSpirV returns DeviceCreationFailed
    try std.testing.expectError(
        error.DeviceCreationFailed,
        backend.createKernel(&spirv),
    );
}

test "VulkanKernel struct defaults" {
    var pipeline = compute.VulkanComputePipeline.init(std.testing.allocator);
    defer pipeline.deinit();
    const kernel = VulkanKernel{ .pipeline = pipeline };
    try std.testing.expectEqual(@as(u64, 0), kernel.spirv_hash);
}

test "VulkanMemory struct defaults" {
    const mem = VulkanMemory{ .buffer = .{} };
    try std.testing.expectEqual(@as(usize, 0), mem.buffer.size);
}

test {
    std.testing.refAllDecls(@This());
}
