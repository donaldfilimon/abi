const std = @import("std");

pub fn tryOpenDriver(names: []const []const u8) bool {
    for (names) |name| {
        var lib = std.DynLib.openZ(name) catch continue;
        defer lib.close();
        return true;
    }
    return false;
}
