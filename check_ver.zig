const std = @import("std");
const builtin = @import("builtin");
pub fn main() void {
    std.debug.print("OS: {s}\n", .{@tagName(builtin.os.tag)});
    std.debug.print("Ver: {}\n", .{builtin.os.version_range.semver.min.major});
}
