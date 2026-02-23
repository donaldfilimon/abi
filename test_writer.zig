const std = @import("std");

pub fn main() !void {
    const stdout_file = std.Io.File.stdout();
    var bw = std.io.bufferedWriter(stdout_file.writer());
    const any_writer = bw.writer().any();
    try any_writer.print("Hello AnyWriter!\n", .{});
    try bw.flush();
}
