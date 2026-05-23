const std = @import("std");
const builtin = @import("builtin");
const backends = @import("backends.zig");

pub fn backendStatusReport(allocator: std.mem.Allocator) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    const caps = backends.backendCapabilitiesList();
    for (caps, 0..) |cap, i| {
        if (i > 0) try out.append(allocator, '\n');
        try out.print(
            allocator,
            "{s}: available={s} accelerated={s} native_kernels={s} - {s}",
            .{
                backends.backendName(cap.backend),
                if (cap.available) "true" else "false",
                if (cap.accelerated) "true" else "false",
                if (cap.native_kernels) "true" else "false",
                cap.message,
            },
        );
    }
    return try out.toOwnedSlice(allocator);
}

pub fn isAvailable() bool {
    return backends.detectBackend().available;
}

pub fn preferredBackend() backends.Backend {
    if (builtin.target.os.tag == .macos) {
        return .metal;
    }
    if (builtin.target.os.tag == .linux or builtin.target.os.tag == .windows) {
        return .vulkan;
    }
    return .simulated;
}

test "gpu backend capability report covers all registered backends" {
    const caps = backends.backendCapabilitiesList();
    try std.testing.expectEqual(@as(usize, 7), caps.len);
    try std.testing.expectEqual(backends.Backend.simulated, caps[0].backend);
    const report = try backendStatusReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    try std.testing.expect(std.mem.indexOf(u8, report, "cuda:") != null);
    try std.testing.expect(std.mem.indexOf(u8, report, "webgpu:") != null);
}
