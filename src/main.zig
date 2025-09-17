const std = @import("std");
const wdbx = @import("wdbx_unified.zig");

pub fn main() !void {
    // Use the unified WDBX CLI implementation
    try wdbx.main().?;
}
