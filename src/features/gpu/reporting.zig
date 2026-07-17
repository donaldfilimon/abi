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
    return backends.preferredBackend();
}

test "gpu backend capability report covers all registered backends" {
    const caps = backends.backendCapabilitiesList();
    try std.testing.expectEqual(@as(usize, 7), caps.len);
    try std.testing.expectEqual(backends.Backend.simulated, caps[0].backend);
    const report = try backendStatusReport(std.testing.allocator);
    defer std.testing.allocator.free(report);
    for (.{ "cuda:", "webgpu:", "webgl2:", "vulkan:", "opengl:" }) |needle| {
        try std.testing.expect(std.mem.indexOf(u8, report, needle) != null);
    }
    for (caps) |cap| {
        switch (cap.backend) {
            .vulkan, .cuda, .webgpu, .opengl, .webgl2 => {
                try std.testing.expect(!cap.available);
                try std.testing.expect(!cap.accelerated);
                try std.testing.expect(!cap.native_kernels);
            },
            else => {},
        }
    }
}

test "preferred backend agrees with detectBackend" {
    try std.testing.expectEqual(preferredBackend(), backends.detectBackend().backend);
}

test {
    std.testing.refAllDecls(@This());
}
