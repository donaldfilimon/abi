const std = @import("std");
pub fn main() !void {
    @compileLog("Has Stringify:", @hasDecl(std.json, "Stringify"));
}
