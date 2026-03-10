const std = @import("std");

pub fn run(
    io: std.Io,
    argv: []const []const u8,
    stdout: std.process.SpawnOptions.StdIo,
    stderr: std.process.SpawnOptions.StdIo,
) !std.process.Child.Term {
    var child = try std.process.spawn(io, .{
        .argv = argv,
        .stdin = .ignore,
        .stdout = stdout,
        .stderr = stderr,
    });
    return try child.wait(io);
}

pub fn runWithThreadedIo(
    allocator: std.mem.Allocator,
    argv: []const []const u8,
    stdout: std.process.SpawnOptions.StdIo,
    stderr: std.process.SpawnOptions.StdIo,
) !std.process.Child.Term {
    var io_backend = std.Io.Threaded.init(allocator, .{});
    defer io_backend.deinit();
    return try run(io_backend.io(), argv, stdout, stderr);
}

pub fn expectSuccess(term: std.process.Child.Term) !void {
    switch (term) {
        .exited => |code| if (code != 0) return error.CommandFailed,
        else => return error.CommandFailed,
    }
}

pub fn runChecked(
    io: std.Io,
    argv: []const []const u8,
    stdout: std.process.SpawnOptions.StdIo,
    stderr: std.process.SpawnOptions.StdIo,
) !void {
    try expectSuccess(try run(io, argv, stdout, stderr));
}

pub fn runCheckedWithThreadedIo(
    allocator: std.mem.Allocator,
    argv: []const []const u8,
    stdout: std.process.SpawnOptions.StdIo,
    stderr: std.process.SpawnOptions.StdIo,
) !void {
    try expectSuccess(try runWithThreadedIo(allocator, argv, stdout, stderr));
}

test {
    std.testing.refAllDecls(@This());
}
