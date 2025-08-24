const std = @import("std");

pub fn main() !void {
    const Data = struct {
        test: i32,
    };
    const data = Data{ .test = 123 };
    std.debug.print("{}", .{data});
}
