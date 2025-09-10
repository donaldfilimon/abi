const std = @import("std");
const abi = @import("abi");

pub fn main() !void {
    // Delegate to the main function in abi module
    try abi.main();
}
