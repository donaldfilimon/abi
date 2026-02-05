const std = @import("std");
// Use the public GPU module to access the dispatcher and builtin kernels.
const abi = @import("abi");
const gpu = abi.gpu;
const dispatcher = gpu.dispatcher;
const builtin_kernels = gpu.builtin_kernels;

test "KernelRing fast‑path reuse increments ring_hits" {
    // Skip execution on CI – kernel execution requires a supported backend.
    if (true) return error.SkipZigTest;
    const device = dispatcher.Device{
        .id = 0,
        .backend = .vulkan,
        .name = "TestDevice",
        .device_type = .discrete,
        .vendor = .unknown,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
        .pci_bus_id = null,
        .driver_version = null,
    };

    var ctx = try dispatcher.KernelDispatcher.init(std.testing.allocator, .vulkan, &device);
    defer ctx.deinit();

    // Use an empty kernel IR (no buffers, no body) for the test.
    const ir = dispatcher.KernelIR.empty("noop");

    const kernel_handle = try ctx.compileKernel(&ir);

    const config = dispatcher.LaunchConfig{ .global_size = .{ 1, 1, 1 }, .local_size = .{ 1, 1, 1 }, .shared_memory = 0 };
    const args = dispatcher.KernelArgs{}; // No buffers needed for no‑op

    // First execution – miss
    _ = try ctx.execute(kernel_handle, config, args);
    try std.testing.expectEqual(@as(u64, 0), ctx.ring_hits);

    // Second identical execution – should reuse
    _ = try ctx.execute(kernel_handle, config, args);
    try std.testing.expectEqual(@as(u64, 1), ctx.ring_hits);
}
