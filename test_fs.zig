const std = @import("std");

pub fn main() !void {
    const file = try std.fs.createFileAbsolute("/tmp/test.txt", .{});
    file.close();

    // what about relative?
    std.debug.print("Success!\n", .{});
}
