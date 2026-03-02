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

/// Allowed command prefixes for gate execution (defense-in-depth).
const allowed_gate_prefixes = [_][]const u8{ "zig", "abi", "cargo", "make", "npm", "bun" };

/// Check that a gate command is safe to execute.
fn validateGateCommand(command: []const u8) bool {
    // Reject shell metacharacters that could enable injection
    for (command) |c| {
        switch (c) {
            ';', '|', '&', '$', '`', '\\', '>', '<', '(', ')', '{', '}' => return false,
            else => {},
        }
    }
    // Require the command starts with a known build tool
    for (allowed_gate_prefixes) |prefix| {
        if (command.len >= prefix.len and
            std.mem.eql(u8, command[0..prefix.len], prefix) and
            (command.len == prefix.len or command[prefix.len] == ' '))
        {
            return true;
        }
    }
    return false;
}

/// Run a configurable gate command (from ralph.yml `gates.per_iteration`).
pub fn runGateCommand(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: []const u8,
    command: []const u8,
) !VerifyResult {
    if (!validateGateCommand(command)) {
        return .{
            .passed = false,
            .exit_code = 1,
            .stdout = try allocator.dupe(u8, ""),
            .stderr = try allocator.dupe(u8, "gate command rejected: must start with zig/abi/cargo/make/npm/bun and contain no shell metacharacters"),
            .command = command,
        };
    }

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
