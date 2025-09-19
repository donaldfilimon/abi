// Linux-specific cross-platform tests
const std = @import("std");
const builtin = @import("builtin");

test "Linux file operations" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    // Test Linux-specific file operations

    // Test /proc filesystem access
    const proc_stat = std.fs.openFileAbsolute("/proc/stat", .{}) catch |err| {
        // /proc might not be available in all environments
        if (err == error.FileNotFound) return error.SkipZigTest;
        return err;
    };
    defer proc_stat.close();

    var buffer: [1024]u8 = undefined;
    const bytes_read = try proc_stat.read(&buffer);
    try std.testing.expect(bytes_read > 0);
}

test "Linux epoll" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;

    // Test Linux epoll API
    const os = std.os;

    const epfd = try os.epoll_create1(0);
    defer os.close(epfd);

    try std.testing.expect(epfd > 0);
}
