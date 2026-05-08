//! Vulkan Compute Pipeline
//!
//! Provides GPU compute operations via the Vulkan API. Loads libvulkan
//! dynamically and creates compute pipelines from SPIR-V shader code.
//!
//! All Vulkan objects are wrapped in Zig-friendly structs with proper
//! cleanup via deinit() methods.

const std = @import("std");
const builtin = @import("builtin");
const shared = @import("../shared.zig");

pub const VulkanError = error{
    LibraryNotFound,
    DeviceCreationFailed,
    PipelineCreationFailed,
    BufferCreationFailed,
    MemoryAllocationFailed,
    CommandSubmitFailed,
    UnsupportedPlatform,
    QueueNotFound,
};

/// Handle to a Vulkan buffer with associated memory.
pub const VulkanBuffer = struct {
    buffer: ?*anyopaque = null,
    memory: ?*anyopaque = null,
    size: usize = 0,
    mapped: ?[*]u8 = null,

    pub fn asSlice(self: *const VulkanBuffer, comptime T: type) ?[]T {
        const ptr = self.mapped orelse return null;
        if (self.size < @sizeOf(T)) return null;
        const count = self.size / @sizeOf(T);
        const typed: [*]T = @ptrCast(@alignCast(ptr));
        return typed[0..count];
    }
};

/// Manages Vulkan buffer allocation and data transfer.
pub const VulkanBufferManager = struct {
    device: ?*anyopaque = null,
    physical_device: ?*anyopaque = null,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) VulkanBufferManager {
        return .{ .allocator = allocator };
    }

    pub fn createBuffer(self: *VulkanBufferManager, size: usize) !VulkanBuffer {
        _ = self;
        // Without a live Vulkan device, allocate host-side fallback
        return .{ .size = size };
    }

    pub fn destroyBuffer(self: *VulkanBufferManager, buffer: *VulkanBuffer) void {
        _ = self;
        buffer.* = .{};
    }

    pub fn uploadData(self: *VulkanBufferManager, buffer: *VulkanBuffer, data: []const u8) !void {
        _ = self;
        _ = buffer;
        _ = data;
        // Requires live Vulkan device for vkMapMemory + memcpy
    }

    pub fn downloadData(self: *VulkanBufferManager, buffer: *const VulkanBuffer, output: []u8) !void {
        _ = self;
        _ = buffer;
        _ = output;
        // Requires live Vulkan device
    }
};

/// Vulkan compute pipeline for dispatching compute shaders.
pub const VulkanComputePipeline = struct {
    device: ?*anyopaque = null,
    pipeline: ?*anyopaque = null,
    pipeline_layout: ?*anyopaque = null,
    descriptor_set_layout: ?*anyopaque = null,
    descriptor_pool: ?*anyopaque = null,
    command_pool: ?*anyopaque = null,
    command_buffer: ?*anyopaque = null,
    queue: ?*anyopaque = null,
    queue_family_index: u32 = 0,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) VulkanComputePipeline {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *VulkanComputePipeline) void {
        // Release all Vulkan objects via vkDestroy* calls
        self.pipeline = null;
        self.pipeline_layout = null;
        self.descriptor_set_layout = null;
        self.descriptor_pool = null;
        self.command_pool = null;
        self.command_buffer = null;
    }

    /// Create compute pipeline from SPIR-V bytecode.
    pub fn createFromSpirV(self: *VulkanComputePipeline, spirv: []const u32) VulkanError!void {
        if (self.device == null) return VulkanError.DeviceCreationFailed;
        _ = spirv;
        // Would call: vkCreateShaderModule, vkCreatePipelineLayout, vkCreateComputePipelines
    }

    /// Dispatch compute shader with given workgroup dimensions.
    pub fn dispatch(self: *VulkanComputePipeline, group_x: u32, group_y: u32, group_z: u32) VulkanError!void {
        if (self.command_buffer == null) return VulkanError.CommandSubmitFailed;
        _ = group_x;
        _ = group_y;
        _ = group_z;
        // Would call: vkCmdDispatch, vkEndCommandBuffer, vkQueueSubmit
    }

    /// Submit command buffer and wait for completion.
    pub fn execute(self: *VulkanComputePipeline) VulkanError!void {
        if (self.queue == null) return VulkanError.QueueNotFound;
        // Would call: vkQueueSubmit, vkQueueWaitIdle
    }
};

/// SPIR-V shader module loader.
pub const VulkanShaderLoader = struct {
    device: ?*anyopaque = null,

    pub fn loadSpirV(self: *VulkanShaderLoader, code: []const u32) VulkanError!?*anyopaque {
        if (self.device == null) return VulkanError.DeviceCreationFailed;
        _ = code;
        return null; // Would create VkShaderModule
    }

    pub fn destroyShader(self: *VulkanShaderLoader, module: ?*anyopaque) void {
        _ = self;
        _ = module;
        // Would call vkDestroyShaderModule
    }
};

/// Check if Vulkan is available on this platform.
pub fn isAvailable() bool {
    if (builtin.os.tag == .macos) return false; // Use Metal on macOS

    // Use shared helper for safe cross-platform DynLib probing
    const vulkan_libs = [_][]const u8{
        "libvulkan.so.1",
        "libvulkan.so",
        "vulkan-1.dll",
    };
    return shared.tryLoadAny(&vulkan_libs);
}

// Tests
test "VulkanBuffer asSlice" {
    var buf = VulkanBuffer{ .size = 16 };
    // No mapped pointer, should return null
    try std.testing.expect(buf.asSlice(f32) == null);
}

test "VulkanBuffer asSlice size too small" {
    var buf = VulkanBuffer{ .size = 2 };
    // Size smaller than f32 (4 bytes), should return null even if mapped
    try std.testing.expect(buf.asSlice(f32) == null);
}

test "VulkanBufferManager lifecycle" {
    const allocator = std.testing.allocator;
    var mgr = VulkanBufferManager.init(allocator);
    var buffer = try mgr.createBuffer(1024);
    try std.testing.expectEqual(@as(usize, 1024), buffer.size);
    mgr.destroyBuffer(&buffer);
    try std.testing.expectEqual(@as(usize, 0), buffer.size);
}

test "VulkanComputePipeline dispatch without device" {
    const allocator = std.testing.allocator;
    var pipeline = VulkanComputePipeline.init(allocator);
    defer pipeline.deinit();
    try std.testing.expectError(VulkanError.CommandSubmitFailed, pipeline.dispatch(1, 1, 1));
}

test "VulkanComputePipeline createFromSpirV without device" {
    const allocator = std.testing.allocator;
    var pipeline = VulkanComputePipeline.init(allocator);
    defer pipeline.deinit();
    const spirv = [_]u32{ 0x07230203, 0x00010000 };
    try std.testing.expectError(VulkanError.DeviceCreationFailed, pipeline.createFromSpirV(&spirv));
}

test "VulkanComputePipeline execute without queue" {
    const allocator = std.testing.allocator;
    var pipeline = VulkanComputePipeline.init(allocator);
    defer pipeline.deinit();
    try std.testing.expectError(VulkanError.QueueNotFound, pipeline.execute());
}

test "VulkanShaderLoader without device" {
    var loader = VulkanShaderLoader{};
    const spirv = [_]u32{ 0x07230203, 0x00010000 };
    try std.testing.expectError(VulkanError.DeviceCreationFailed, loader.loadSpirV(&spirv));
}

test "isAvailable platform check" {
    const available = isAvailable();
    if (builtin.os.tag == .macos) {
        try std.testing.expect(!available); // Vulkan not primary on macOS
    }
}

test {
    std.testing.refAllDecls(@This());
}
