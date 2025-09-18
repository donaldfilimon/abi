const std = @import("std");
const wdbx = @import("wdbx/unified.zig");

pub fn main() !void {
    // Use the unified WDBX CLI implementation
    try wdbx.main().?;
}
