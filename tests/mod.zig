//! Minimal smoke tests for the ABI framework.
const std = @import("std");
const abi = @import("abi");

test "abi module compiles and exposes version" {
    _ = abi;
    try std.testing.expect(abi.version().len > 0);
}
