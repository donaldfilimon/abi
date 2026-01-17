const std = @import("std");
pub fn main() void {
    const cwd = std.fs.cwd();
    _ = cwd;
}
