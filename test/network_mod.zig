//! Focused network integration test root.

const std = @import("std");
const abi = @import("abi");
const build_options = @import("build_options");

const network_tests = @import("integration/network_test.zig");

test {
    _ = abi;
    _ = build_options;
    std.testing.refAllDecls(@This());
}
