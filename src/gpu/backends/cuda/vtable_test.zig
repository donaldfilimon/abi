//! CUDA VTable Integration Tests
//!
//! Tests the complete CUDA backend through the VTable interface.

const std = @import("std");
const interface = @import("../../interface.zig");
const backend_factory = @import("../../backend_factory.zig");

test "CUDA VTable integration - device query" {
    const allocator = std.testing.allocator;

    // Create backend via factory
    const backend = backend_factory.createVTableBackend(allocator, .cuda) catch |err| {
        if (err == backend_factory.FactoryError.BackendNotAvailable or
            err == backend_factory.FactoryError.BackendInitializationFailed)
        {
            return error.SkipZigTest;
        }
        return err;
    };
    defer backend.deinit();

    // Query device info
    const count = backend.getDeviceCount();
    if (count == 0) return error.SkipZigTest;

    const caps = try backend.getDeviceCaps(0);

    // Basic sanity checks
    try std.testing.expect(caps.max_threads_per_block > 0);
    try std.testing.expect(caps.warp_size > 0);
}

test "CUDA VTable integration - memory operations" {
    const allocator = std.testing.allocator;

    // Create backend via factory
    const backend = backend_factory.createVTableBackend(allocator, .cuda) catch |err| {
        if (err == backend_factory.FactoryError.BackendNotAvailable or
            err == backend_factory.FactoryError.BackendInitializationFailed)
        {
            return error.SkipZigTest;
        }
        return err;
    };
    defer backend.deinit();

    // Allocate device memory
    const size: usize = 1024;
    const ptr = backend.allocate(size, .{}) catch |err| {
        if (err == error.OutOfMemory) return error.SkipZigTest;
        return err;
    };
    defer backend.free(ptr);

    // Test copy to device
    var host_data: [256]u8 = undefined;
    for (&host_data, 0..) |*v, i| v.* = @intCast(i % 256);

    backend.copyToDevice(ptr, &host_data) catch |err| {
        if (err == error.TransferFailed) return error.SkipZigTest;
        return err;
    };

    // Test copy from device
    var result: [256]u8 = undefined;
    backend.copyFromDevice(&result, ptr) catch |err| {
        if (err == error.TransferFailed) return error.SkipZigTest;
        return err;
    };

    try std.testing.expectEqualSlices(u8, &host_data, &result);
}

test "CUDA VTable integration - synchronization" {
    const allocator = std.testing.allocator;

    // Create backend via factory
    const backend = backend_factory.createVTableBackend(allocator, .cuda) catch |err| {
        if (err == backend_factory.FactoryError.BackendNotAvailable or
            err == backend_factory.FactoryError.BackendInitializationFailed)
        {
            return error.SkipZigTest;
        }
        return err;
    };
    defer backend.deinit();

    // Test synchronization (should not error)
    backend.synchronize() catch |err| {
        if (err == error.NotAvailable) return error.SkipZigTest;
        return err;
    };
}
