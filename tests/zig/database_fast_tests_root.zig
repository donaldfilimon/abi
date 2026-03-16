const std = @import("std");
const abi = @import("abi");

test {
    std.testing.refAllDecls(abi.database);
}
