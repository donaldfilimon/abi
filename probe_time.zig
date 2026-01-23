const std = @import("std");
pub fn main() !void {
    @compileLog("Has timestamp:", @hasDecl(std.time, "timestamp"));
    @compileLog("Has milliTimestamp:", @hasDecl(std.time, "milliTimestamp"));
    @compileLog("Has nanoTimestamp:", @hasDecl(std.time, "nanoTimestamp"));
    @compileLog("Has microTimestamp:", @hasDecl(std.time, "microTimestamp"));
}
