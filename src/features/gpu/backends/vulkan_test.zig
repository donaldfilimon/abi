//! Tests for the Vulkan backend implementation.
//!
//! Extracted from vulkan.zig for better code organization.

const std = @import("std");
const vulkan = @import("vulkan.zig");

const VulkanError = vulkan.VulkanError;
const VkResult = vulkan.VkResult;

test "VulkanError enum covers key cases" {
    // Verify core error values exist in the VulkanError set
    const errors = [_]VulkanError{
        error.InitializationFailed,
        error.DeviceNotFound,
        error.InstanceCreationFailed,
        error.PhysicalDeviceNotFound,
        error.LogicalDeviceCreationFailed,
        error.DeviceCreationFailed,
        error.QueueFamilyNotFound,
        error.MemoryTypeNotFound,
        error.ShaderCompilationFailed,
        error.PipelineCreationFailed,
        error.CommandBufferAllocationFailed,
        error.BufferCreationFailed,
        error.MemoryAllocationFailed,
        error.MemoryCopyFailed,
        error.CommandRecordingFailed,
        error.SubmissionFailed,
        error.InvalidHandle,
        error.SynchronizationFailed,
        error.DeviceLost,
        error.ValidationLayerNotAvailable,
        error.VersionNotSupported,
        error.NotFound,
    };
    try std.testing.expectEqual(@as(usize, 22), errors.len);
}

test "VkResult success value is zero" {
    try std.testing.expectEqual(@as(i32, 0), @intFromEnum(VkResult.success));
}

test "vulkan_initialized starts as false" {
    // Can't call isAvailable directly, but we can verify the initial state
    try std.testing.expect(!vulkan.vulkan_initialized);
}

test "linux vulkan minimum api version is 1.3" {
    const caps = @import("vulkan/capabilities.zig");
    try std.testing.expect(!caps.meetsTargetMinimum(.linux, caps.encodeApiVersion(.{ .major = 1, .minor = 2, .patch = 0 })));
    try std.testing.expect(caps.meetsTargetMinimum(.linux, caps.encodeApiVersion(.{ .major = 1, .minor = 3, .patch = 0 })));
}

test {
    std.testing.refAllDecls(@This());
}
