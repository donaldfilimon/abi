const std = @import("std");
const builtin = @import("builtin");

pub const CommandResult = struct {
    output: []u8,
    exit_code: i32,

    pub fn ok(self: CommandResult) bool {
        return self.exit_code == 0;
    }
};

pub fn trimSpace(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, " \t\r\n");
}

pub fn fileExists(io: anytype, path: []const u8) bool {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

pub fn dirExists(io: anytype, path: []const u8) bool {
    var dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch return false;
    dir.close(io);
    return true;
}

pub fn captureCommand(allocator: std.mem.Allocator, io: std.Io, cmd: []const u8) !CommandResult {
    const shell = if (builtin.os.tag == .windows) "cmd" else "sh";
    const shell_arg = if (builtin.os.tag == .windows) "/c" else "-c";
    
    // Merge stderr to stdout via shell redirection
    const merged_cmd = try std.fmt.allocPrint(allocator, "{s} 2>&1", .{cmd});
    defer allocator.free(merged_cmd);
    
    const argv = [_][]const u8{ shell, shell_arg, merged_cmd };

    var child = try std.process.spawn(io, .{
        .argv = &argv,
        .stdout = .pipe,
    });

    const max_output_size = 1024 * 1024; // 1MB
    const stdout = try readAllAlloc(io, child.stdout.?, allocator, max_output_size);
    errdefer allocator.free(stdout);

    const term = try child.wait(io);
    const exit_code = switch (term) {
        .exited => |code| @as(i32, @intCast(code)),
        else => -1,
    };

    return .{
        .output = stdout,
        .exit_code = exit_code,
    };
}

fn readAllAlloc(io: std.Io, file: std.Io.File, allocator: std.mem.Allocator, limit: usize) ![]u8 {
    var list = std.ArrayListUnmanaged(u8).empty;
    errdefer list.deinit(allocator);

    var buffer: [4096]u8 = undefined;
    while (true) {
        const amt = try file.readStreaming(io, &.{&buffer});
        if (amt == 0) break;
        try list.appendSlice(allocator, buffer[0..amt]);
        if (list.items.len > limit) return error.StreamTooLong;
    }

    return try list.toOwnedSlice(allocator);
}

pub fn runCommand(io: std.Io, cmd: []const u8) !i32 {
    const shell = if (builtin.os.tag == .windows) "cmd" else "sh";
    const shell_arg = if (builtin.os.tag == .windows) "/c" else "-c";
    const argv = [_][]const u8{ shell, shell_arg, cmd };

    var child = try std.process.spawn(io, .{
        .argv = &argv,
    });
    const term = try child.wait(io);
    return switch (term) {
        .exited => |code| @as(i32, @intCast(code)),
        else => -1,
    };
}

pub fn commandExists(allocator: std.mem.Allocator, io: std.Io, name: []const u8) !bool {
    const cmd = try std.fmt.allocPrint(allocator, "command -v {s} >/dev/null 2>&1", .{name});
    defer allocator.free(cmd);
    return (try runCommand(io, cmd)) == 0;
}

pub fn readFileAlloc(
    allocator: std.mem.Allocator,
    io: anytype,
    path: []const u8,
    max_bytes: usize,
) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_bytes));
}
