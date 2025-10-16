// This file is a thin wrapper that forwards to the original `src/comprehensive_cli.zig`
// so that the new CLI structure can be used without relocating the actual source.
const std = @import("std");

pub fn main() !void {
    // Import the original CLI implementation.
    const orig = @import("../../comprehensive_cli.zig");
    try orig.main();
}
