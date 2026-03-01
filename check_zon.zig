const std = @import("std");
pub fn main() !void {
    std.debug.print("has std.zon: {}\n", .{@hasDecl(std, "zon")});
}
