// Backwards‑compatible bootstrap that simply forwards to the new CLI.
const std = @import("std");

pub fn main() !void {
    const cli = @import("../../cli/comprehensive_cli.zig");
    try cli.main();
}
