const std = @import("std");
const dispatcher = @import("../dispatcher.zig");
const builtin_kernels = @import("../../builtin_kernels.zig");
const Backend = dispatcher.Backend;
const Device = dispatcher.Device;
const LaunchConfig = dispatcher.LaunchConfig;
const KernelArgs = dispatcher.KernelArgs;
const KernelIR = dispatcher.KernelIR;

test "KernelRing fast‑path reuse increments ring_hits" {
    // Create a mock device (emulated) – reuse an existing test helper if present
    const device = Device{
        .id = 0,
        .backend = .vulkan,
        .name = "TestDevice",
        .device_type = .discrete,
        .total_memory = null,
        .available_memory = null,
        .is_emulated = true,
        .capability = .{},
        .compute_units = null,
        .clock_mhz = null,
    };

    var ctx = try dispatcher.KernelDispatcher.init(std.testing.allocator, .vulkan, &device);
    defer ctx.deinit();

    // Build a trivial kernel IR (e.g., a no‑op) using the builtin helper
    const ir = try builtin_kernels.buildKernelIR(std.testing.allocator, .noop);
    defer ir.deinit(std.testing.allocator);

    const kernel_handle = try ctx.compileKernel(ir);

    // Simple launch config – 1x1 grid, 1 thread
    const config = LaunchConfig.init(.{ .grid = .{ 1, 1, 1 }, .local = .{ 1, 1, 1 } });
    const args = KernelArgs{}; // No buffers needed for no‑op

    // First execution – should be a miss
    const result1 = try ctx.execute(kernel_handle, config, args);
    _ = result1; // ignore
    try std.testing.expectEqual(@as(u64, 0), ctx.ring_hits);

    // Second identical execution – should reuse ring entry
    const result2 = try ctx.execute(kernel_handle, config, args);
    _ = result2;
    try std.testing.expectEqual(@as(u64, 1), ctx.ring_hits);
}

test {
    std.testing.refAllDecls(@This());
}
