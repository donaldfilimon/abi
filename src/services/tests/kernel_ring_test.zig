const std = @import("std");
// Use the public GPU module to access the dispatcher and builtin kernels.
const abi = @import("abi");
const gpu = abi.features.gpu;
const dispatcher = gpu.dispatch;
const builtin_kernels = gpu.builtin_kernels;

test "KernelRing fast-path reuse increments ring_hits" {
    if (!@import("build_options").enable_gpu) return error.SkipZigTest;

    const device = dispatcher.Device{
        .id = 0,
        .backend = .stdgpu,
        .name = "SimulatedTestDevice",
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

    var ctx = try dispatcher.KernelDispatcher.init(std.testing.allocator, .stdgpu, &device);
    defer ctx.deinit();

    // Use an empty kernel IR (no buffers, no body) for the test.
    // The "noop" kernel has no CPU fallback, so execute() will return
    // ExecutionFailed after the ring buffer is updated. We intentionally
    // ignore that error since we are only testing ring_hits tracking.
    const ir = dispatcher.KernelIR.empty("noop");

    const kernel_handle = try ctx.compileKernel(&ir);

    const config = dispatcher.LaunchConfig{
        .global_size = .{ 1, 1, 1 },
        .local_size = .{ 1, 1, 1 },
        .shared_memory = 0,
    };
    const args = dispatcher.KernelArgs{}; // No buffers needed for no-op

    // First execution -- ring miss (new descriptor inserted).
    // execute() returns ExecutionFailed because "noop" has no CPU fallback,
    // but the ring buffer push happens before kernel dispatch.
    _ = ctx.execute(kernel_handle, config, args) catch {};
    try std.testing.expectEqual(@as(u64, 0), ctx.ring_hits);

    // Second identical execution -- should reuse the ring slot.
    _ = ctx.execute(kernel_handle, config, args) catch {};
    try std.testing.expectEqual(@as(u64, 1), ctx.ring_hits);
}
