const std = @import("std");

pub const LoggingSink = enum {
    stdout,
    stderr,
    file,
};

pub const ProfileConfig = struct {
    sink: LoggingSink = .stdout,
    path: []const u8 = "",
};

pub fn writeProfileLine(sink: LoggingSink, line: []const u8) void {
    switch (sink) {
        .stdout => std.debug.print("{s}\n", .{line}),
        .stderr => std.debug.print("{s}\n", .{line}),
        .file => std.debug.print("{s}\n", .{line}),
    }
}
