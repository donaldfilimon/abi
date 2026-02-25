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
    return runGateCommand(allocator, io, cwd, "zig build verify-all");
}

/// Run a configurable gate command (from ralph.yml `gates.per_iteration`).
pub fn runGateCommand(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: []const u8,
    command: []const u8,
) !VerifyResult {
    // Split command into argv (space-separated, simple)
    var argv_list: std.ArrayListUnmanaged([]const u8) = .empty;
    defer argv_list.deinit(allocator);

    var parts = std.mem.splitScalar(u8, command, ' ');
    while (parts.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \t");
        if (trimmed.len > 0) try argv_list.append(allocator, trimmed);
    }

    if (argv_list.items.len == 0) {
        return .{
            .passed = true,
            .exit_code = 0,
            .stdout = try allocator.dupe(u8, ""),
            .stderr = try allocator.dupe(u8, "(no gate command)"),
            .command = command,
        };
    }

    const result = try std.process.run(allocator, io, .{
        .argv = argv_list.items,
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
        .command = command,
    };
}

test {
    std.testing.refAllDecls(@This());
}
