const std = @import("std");
pub fn main() !void {
    @compileLog("std.Io exists:", @hasDecl(std, "Io"));
}
