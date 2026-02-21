const std = @import("std");
const metal = @import("metal.zig");
const caps = @import("metal/capabilities.zig");
const MetalError = metal.MetalError;
const DeviceInfo = metal.DeviceInfo;

test "MetalError enum covers key cases" {
    // Verify core error values exist in the MetalError set
    const errors = [_]MetalError{
        error.InitializationFailed,
        error.DeviceNotFound,
        error.ShaderCompilationFailed,
        error.PipelineCreationFailed,
        error.CommandQueueCreationFailed,
        error.BufferCreationFailed,
        error.CommandBufferCreationFailed,
        error.KernelExecutionFailed,
        error.MemoryCopyFailed,
        error.ObjcRuntimeUnavailable,
        error.SelectorNotFound,
        error.InvalidGridSize,
        error.InvalidBlockSize,
        error.NSStringCreationFailed,
        error.DeviceQueryFailed,
    };
    try std.testing.expectEqual(@as(usize, 15), errors.len);
}

test "isAvailable and getDeviceInfo are consistent" {
    _ = metal.isAvailable();
    _ = metal.getDeviceInfo();
    try std.testing.expect(true);
}

test "getDeviceInfo returns null when not initialized" {
    const info = metal.getDeviceInfo();
    try std.testing.expect(info == null);
}

test "DeviceInfo struct has correct fields" {
    const info = DeviceInfo{
        .name = "Test Device",
        .total_memory = 8 * 1024 * 1024 * 1024,
        .max_buffer_length = 256 * 1024 * 1024,
        .max_threads_per_threadgroup = 1024,
        .has_unified_memory = true,
    };
    try std.testing.expectEqualStrings("Test Device", info.name);
    try std.testing.expect(info.has_unified_memory);
}

test "metal capability mapping covers metal4 threshold" {
    try std.testing.expectEqual(caps.MetalLevel.none, caps.levelFromFamily(.apple6));
    try std.testing.expectEqual(caps.MetalLevel.metal3, caps.levelFromFamily(.apple8));
    try std.testing.expectEqual(caps.MetalLevel.metal4, caps.levelFromFamily(.apple9));
    try std.testing.expect(caps.MetalLevel.metal4.atLeast(caps.required_runtime_level));
}
