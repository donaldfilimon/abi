const std = @import("std");
pub fn main() !void {
    @compileLog("std.process.Environ.empty exists:", @hasDecl(std.process.Environ, "empty"));
}
