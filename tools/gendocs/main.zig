//! ABI documentation pipeline entrypoint.
//! Run via: zig build gendocs -- [--check] [--api-only] [--no-wasm]

const std = @import("std");
const gendocs = @import("mod.zig");

pub fn main(init: std.process.Init.Minimal) !void {
    return gendocs.main(init);
}
