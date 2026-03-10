const std = @import("std");
const build_options = @import("build_options");
const abi_simple = @import("abi-simple");

pub fn main(init: std.process.Init.Minimal) void {
    return abi_simple.main(init);
}
