const std = @import("std");
const factory = @import("../backend_factory.zig");
const Backend = @import("../backend.zig").Backend;

test "detect all available backends" {
    const allocator = std.testing.allocator;

    const available = try factory.detectAvailableBackends(allocator);
    defer allocator.free(available);

    if (available.len == 0) return error.SkipZigTest;

    try std.testing.expect(available.len >= 1);
}

test "backend priority respects availability" {
    // Only run if specific backends are actually available
    if (!factory.isBackendAvailable(.cuda)) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const best = try factory.selectBestBackendWithFallback(allocator, .{
        .preferred = .cuda,
        .fallback_chain = &.{ .vulkan, .metal, .stdgpu },
    });

    try std.testing.expect(best != null);
}

test "backend detection with feature requirements" {
    const allocator = std.testing.allocator;

    const best = try factory.selectBackendWithFeatures(allocator, .{
        .required_features = &.{ .fp16, .atomics },
        .fallback_to_cpu = false,
    });

    // May be null on systems without FP16 GPU support
    if (best) |backend| {
        try std.testing.expect(backend != .stdgpu);
    }
}

test {
    std.testing.refAllDecls(@This());
}
