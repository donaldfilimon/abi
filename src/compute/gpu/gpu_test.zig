//! Tests for GPU module
//!
//! Tests only compiled when enable_gpu is true.

const std = @import("std");
const gpu = @import("gpu/mod.zig");

test "GPUBackend enum has correct variants" {
    try std.testing.expectEqual(gpu.GPUBackend.none, gpu.GPUBackend.none);
    try std.testing.expectEqual(gpu.GPUBackend.cuda, gpu.GPUBackend.cuda);
    try std.testing.expectEqual(gpu.GPUBackend.vulkan, gpu.GPUBackend.vulkan);
    try std.testing.expectEqual(gpu.GPUBackend.metal, gpu.GPUBackend.metal);
    try std.testing.expectEqual(gpu.GPUBackend.webgpu, gpu.GPUBackend.webgpu);
}

test "GPUWorkloadHints default initialization" {
    const hints = gpu.DEFAULT_GPU_HINTS;
    try std.testing.expect(!hints.prefers_gpu);
    try std.testing.expect(!hints.requires_double_precision);
    try std.testing.expect(hints.min_compute_capability == null);
    try std.testing.expect(hints.estimated_memory_bytes == null);
}

test "GPUManager initialization" {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const manager = try gpu.GPUManager.init(gpa.allocator(), gpu.GPUBackend.none);
    defer manager.deinit();

    try std.testing.expectEqual(gpu.GPUBackend.none, manager.backend);
    try std.testing.expectEqual(@as(usize, 0), manager.available_devices.len);
}

test "GPUMemoryRequirements struct layout" {
    const req = gpu.GPUMemoryRequirements{
        .device_memory_bytes = 1024 * 1024 * 1024,
        .host_memory_bytes = 512 * 1024 * 1024,
        .requires_shared_memory = true,
    };

    try std.testing.expectEqual(@as(u64, 1073741824), req.device_memory_bytes);
    try std.testing.expectEqual(@as(u64, 536870912), req.host_memory_bytes);
    try std.testing.expect(req.requires_shared_memory);
}
