//! Ralph verification helpers.

const std = @import("std");

pub const VerifyResult = struct {
    passed: bool,
    exit_code: u8,
    stdout: []u8,
    stderr: []u8,
    command: []const u8,

    pub fn deinit(self: *VerifyResult, allocator: std.mem.Allocator) void {
        allocator.free(self.stdout);
        allocator.free(self.stderr);
    }
};

pub fn runVerifyAll(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: []const u8,
) !VerifyResult {
    const result = try std.process.run(allocator, io, .{
        .argv = &.{ "zig", "build", "verify-all" },
        .cwd = .{ .path = cwd },
        .stdout_limit = .limited(64 * 1024 * 1024),
        .stderr_limit = .limited(64 * 1024 * 1024),
    });

    const exit_code: u8 = switch (result.term) {
        .exited => |code| code,
        else => 1,
    };

    return .{
        .passed = exit_code == 0,
        .exit_code = exit_code,
        .stdout = result.stdout,
        .stderr = result.stderr,
        .command = "zig build verify-all",
    };
}
