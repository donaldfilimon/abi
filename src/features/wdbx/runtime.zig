const std = @import("std");
const build_options = @import("build_options");
const gpu = if (build_options.feat_gpu) @import("../gpu/mod.zig") else @import("../gpu/stub.zig");

pub const AccelerationStatus = struct {
    backend: gpu.Backend,
    mode: gpu.ExecutionMode,
    message: []const u8,
};

pub fn backendName(backend: gpu.Backend) []const u8 {
    return gpu.backendName(backend);
}

pub fn executionModeName(mode: gpu.ExecutionMode) []const u8 {
    return switch (mode) {
        .cpu_fallback => "cpu_fallback",
        .simulated_gpu => "simulated_gpu",
        .native_gpu => "native_gpu",
    };
}

pub fn defaultAcceleration() AccelerationStatus {
    const status = gpu.detectBackend();
    return .{
        .backend = status.backend,
        .mode = if (status.accelerated) .native_gpu else .simulated_gpu,
        .message = status.message,
    };
}

pub fn runAccelerationKernel(name: []const u8, work_items: usize) !AccelerationStatus {
    const result = try gpu.executeKernel(.{ .name = name, .work_items = work_items });
    return .{ .backend = result.backend, .mode = result.mode, .message = result.message };
}

test "runtime acceleration status exports stable mode names" {
    try std.testing.expectEqualStrings("cpu_fallback", executionModeName(.cpu_fallback));
    try std.testing.expectEqualStrings("simulated_gpu", executionModeName(.simulated_gpu));
    try std.testing.expectEqualStrings("native_gpu", executionModeName(.native_gpu));
}

test {
    std.testing.refAllDecls(@This());
}
