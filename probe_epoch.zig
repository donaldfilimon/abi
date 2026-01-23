const std = @import("std");
pub fn main() !void {
    @compileLog("Has now:", @hasDecl(std.time.epoch, "now"));
    @compileLog("Has systemTime:", @hasDecl(std.time.epoch, "systemTime"));
    @compileLog("Has DateTime:", @hasDecl(std.time.epoch, "DateTime"));
}
