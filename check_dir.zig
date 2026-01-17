const std = @import("std");
pub fn main() void {
    @compileLog(std.meta.declNames(std.Io.Dir));
}
