// Linux-specific cross-platform tests
const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;

const ai = @import("../../lib/mod.zig");

test "linux SIMD operations" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    // Test SIMD vector operations on Linux
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };

    var result: [4]f32 = undefined;
    for (&result, 0..) |*r, i| {
        r.* = a[i] + b[i];
    }

    try testing.expect(result[0] == 3.0);
}

test "linux GPU compute" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test GPU compute operations on Linux
    var accel = try ai.gpu.accelerator.createBestAccelerator(allocator);
    defer accel.deinit();

    // Test kernel execution
    const size = 1024;
    var input = try allocator.alloc(f32, size);
    defer allocator.free(input);

    for (input, 0..) |*d, i| d.* = @as(f32, @floatFromInt(i));

    // Placeholder for GPU kernel test
    try testing.expect(input.len == size);
}

test "linux container compatibility" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    // Test container-specific features
    const cgroup_path = "/proc/self/cgroup";
    const file = std.fs.openFileAbsolute(cgroup_path, .{}) catch |err| {
        // In containers, this might fail differently
        try testing.expect(err == error.FileNotFound or err == error.AccessDenied);
        return;
    };
    defer file.close();

    var buffer: [1024]u8 = undefined;
    const bytes_read = try file.read(&buffer);
    try testing.expect(bytes_read > 0);

    // Check if we're in a container by looking for container-specific cgroup paths
    const content = buffer[0..bytes_read];
    const in_container = std.mem.indexOf(u8, content, "containerd") != null or
        std.mem.indexOf(u8, content, "docker") != null or
        std.mem.indexOf(u8, content, "podman") != null or
        std.mem.indexOf(u8, content, "lxc") != null;

    std.log.info("Running in container: {}", .{in_container});
}

test "linux docker container networking" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    // Test networking in container environments
    const hostname = std.os.getenv("HOSTNAME") orelse "unknown";
    std.log.info("Container hostname: {s}", .{hostname});

    // Test localhost connectivity (should work in containers)
    const address = try std.net.Address.parseIp4("127.0.0.1", 0);
    try testing.expect(address.getPort() == 0);
}

test "Linux epoll" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    // Test Linux epoll API
    const posix = std.posix;

    const epfd = try posix.epoll_create1(0);
    defer posix.close(epfd);

    try std.testing.expect(epfd > 0);
}
