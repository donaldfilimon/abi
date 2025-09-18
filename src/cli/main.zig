const std = @import("std");
const abi = @import("abi");
const wdbx = abi.wdbx;

pub fn main() !void {
    // Run WDBX main (if it returns an error, propagate)
    try wdbx.main();
}
