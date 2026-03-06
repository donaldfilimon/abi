const std = @import("std");
const build_options = @import("build_options.zig");
const abi_simple = @import("abi-simple.zig");

pub fn main(init: std.process.Init.Minimal) void {
    return abi_simple.main(init);
}
