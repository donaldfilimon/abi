const std = @import("std");
pub fn main() !void {
    inline for (std.meta.declarations(std.time)) |decl| {
        @compileLog(decl.name);
    }
}
