const std = @import("std");

pub fn main() !void {
    const decls = std.meta.declarations(std.time);
    inline for (decls) |decl| {
        std.debug.print("{s}\n", .{decl.name});
    }
}
