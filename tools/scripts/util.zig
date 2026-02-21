const std = @import("std");
const builtin = @import("builtin");

const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
});

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

pub fn captureCommand(allocator: std.mem.Allocator, cmd: []const u8) !CommandResult {
    const cmd_full = try std.fmt.allocPrint(allocator, "{s} 2>&1", .{cmd});
    defer allocator.free(cmd_full);
    const cmd_z = try allocator.dupeZ(u8, cmd_full);
    defer allocator.free(cmd_z);

    const pipe = if (builtin.os.tag == .windows)
        c._popen(cmd_z.ptr, "r")
    else
        c.popen(cmd_z.ptr, "r");

    if (pipe == null) return error.CommandFailed;

    var output = std.ArrayListUnmanaged(u8).empty;
    errdefer output.deinit(allocator);

    var buf: [1024]u8 = undefined;
    while (true) {
        const ptr = c.fgets(&buf, @intCast(buf.len), pipe);
        if (ptr == null) break;
        const len = std.mem.indexOfScalar(u8, &buf, 0) orelse buf.len;
        try output.appendSlice(allocator, buf[0..len]);
    }

    const status_raw = if (builtin.os.tag == .windows) c._pclose(pipe) else c.pclose(pipe);
    const exit_code = normalizeExitCode(status_raw);

    return .{
        .output = try output.toOwnedSlice(allocator),
        .exit_code = exit_code,
    };
}

pub fn runCommand(allocator: std.mem.Allocator, cmd: []const u8) !i32 {
    const cmd_z = try allocator.dupeZ(u8, cmd);
    defer allocator.free(cmd_z);
    return normalizeExitCode(c.system(cmd_z.ptr));
}

pub fn commandExists(allocator: std.mem.Allocator, name: []const u8) !bool {
    const cmd = try std.fmt.allocPrint(allocator, "command -v {s} >/dev/null 2>&1", .{name});
    defer allocator.free(cmd);
    return (try runCommand(allocator, cmd)) == 0;
}

pub fn readFileAlloc(
    allocator: std.mem.Allocator,
    io: anytype,
    path: []const u8,
    max_bytes: usize,
) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_bytes));
}

fn normalizeExitCode(status_raw: c_int) i32 {
    if (builtin.os.tag == .windows) {
        return @as(i32, @intCast(status_raw));
    }
    return @as(i32, @intCast(status_raw >> 8));
}
