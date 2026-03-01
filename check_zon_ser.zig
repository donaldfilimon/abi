const std = @import("std");
pub fn main() !void {
    const arr = [_]struct{a: i32}{.{.a = 1}, .{.a = 2}};
    var buf: [1024]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buf);
    try std.zon.stringify.serialize(&arr, .{}, &writer);
    std.debug.print("{s}\n", .{writer.buffer[0..writer.end]});
}

